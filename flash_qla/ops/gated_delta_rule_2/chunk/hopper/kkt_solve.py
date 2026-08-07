# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""GDN2 kkt_solve: A = (I + StrictLower(M))^{-1} with

    M_ij = sum_c (b_ic k_ic) exp(G_ic - G_jc) k_jc,   G = chunk-local cumsum(g)

Unlike gdn, the per-KEY-CHANNEL decay cannot be conjugated out of the
Gram matrix (see DESIGN_GDR2.md), so it is folded into the k tiles here:

- 16-token DIAGONAL sub-blocks: elementwise over channels — the pairwise
  exponent G_i - G_j is <= 0 on the used triangle for any gate magnitude.
- OFF-DIAGONAL sub-blocks: two bounded factors rebased at the row-block
  start r:  kl_i = (b k)_i exp(G_i - G_r)  (i >= r),
            kr_j = k_j exp(G_r - G_j)      (j <  r),
  both exponents <= 0, then a [16, DK] x [DK, 64] GEMM.

The 16 -> 32 -> 64 blocked triangular inversion is copied from the gdn
kernel unchanged: it is gate-agnostic.
"""

from typing import Optional

import torch
import tilelang
import tilelang.language as T

from flash_qla.utils import prepare_chunk_indices

L2E = 1.442695  # log2(e): T.exp2(x * L2E) == exp(x)


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_kkt_solve_2(
    H,
    Hg,
    DK,
    chunk_size,
    accum_dtype,
    qkva_dtype,
    g_dtype,
    seqlen_dtype,
    is_varlen,
):
    data_batch_size = T.dynamic("data_batch_size")
    real_batch_size = T.dynamic("real_batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    block_S = chunk_size
    SUB = 16
    NSUB = block_S // SUB

    k_shape = (data_batch_size, num_tokens, Hg, DK)
    g_shape = (data_batch_size, num_tokens, H, DK)
    b_shape = (data_batch_size, num_tokens, H, DK)
    a_shape = (data_batch_size, num_tokens, H, chunk_size)

    @T.macro
    def kernel_body(
        bb,
        bc,
        bh,
        bhg,
        batch_idx,
        chunk_idx,
        seq_start_idx,
        seq_end_idx,
        k,
        g,
        b,
        a,
    ):
        left = seq_start_idx + chunk_idx * block_S
        right = left + block_S

        k_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
        b_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
        g_shared = T.alloc_shared((block_S, DK), dtype=accum_dtype)
        kl_shared = T.alloc_shared((SUB, DK), dtype=qkva_dtype)
        kr_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
        arow_fragment = T.alloc_fragment((SUB, block_S), dtype=accum_dtype)
        a64_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
        diag_acc = T.alloc_fragment((SUB, SUB), dtype=accum_dtype)

        a16i_row = T.alloc_fragment((4, 16), dtype=accum_dtype)
        a16i_sum = T.alloc_fragment((4, 16), dtype=accum_dtype)

        a16i_shared = T.alloc_shared((4, 17, 16), dtype=accum_dtype)
        a16o_shared = T.alloc_shared((2, 17, 16), dtype=accum_dtype)
        a16o_fragment = T.alloc_fragment((2, 16, 16), dtype=accum_dtype)

        a32i_fragment = T.alloc_fragment((2, 32, 32), dtype=accum_dtype)
        a32i0_shared = T.alloc_shared((32, 32), dtype=accum_dtype)
        a32i1_shared = T.alloc_shared((32, 32), dtype=accum_dtype)
        a32o_shared = T.alloc_shared((32, 32), dtype=accum_dtype)
        a32o_fragment = T.alloc_fragment((32, 32), dtype=accum_dtype)

        a64_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)

        T.annotate_layout(
            {
                a16i_shared: tilelang.layout.make_linear_layout(a16i_shared),
                a16o_shared: tilelang.layout.make_linear_layout(a16o_shared),
            }
        )

        # Load K (zero-padded past the sequence end)
        if right <= seq_end_idx:
            T.async_copy(k[bb, left:right, bhg, 0:DK], k_shared)
        else:
            for j_s, j_k in T.Parallel(block_S, DK):
                if left + j_s < seq_end_idx:
                    k_shared[j_s, j_k] = k[bb, left + j_s, bhg, j_k]
                else:
                    k_shared[j_s, j_k] = 0

        # Load b (zero-padded) and g (tail-filled with the last valid row,
        # so pairwise differences stay bounded on padded rows)
        for j_s, j_k in T.Parallel(block_S, DK):
            if left + j_s < seq_end_idx:
                b_shared[j_s, j_k] = b[bb, left + j_s, bh, j_k]
                g_shared[j_s, j_k] = g[bb, left + j_s, bh, j_k]
            else:
                b_shared[j_s, j_k] = 0
                g_shared[j_s, j_k] = g[bb, seq_end_idx - 1, bh, j_k]
        if right <= seq_end_idx:
            T.ptx_wait_group(0)

        T.clear(a64_fragment)

        # Off-diagonal sub-blocks: bounded two-factor form, one GEMM per
        # row block. Row block bi covers rows [r, r+16); columns [0, r).
        for bi in T.serial(1, NSUB):
            # kl = (b * k) * exp(G - G_r) on the row block (exponent <= 0)
            for j_s, j_k in T.Parallel(SUB, DK):
                kl_shared[j_s, j_k] = (
                    b_shared[bi * SUB + j_s, j_k].astype(accum_dtype)
                    * k_shared[bi * SUB + j_s, j_k].astype(accum_dtype)
                    * T.exp2(
                        (g_shared[bi * SUB + j_s, j_k] - g_shared[bi * SUB, j_k])
                        * L2E
                    )
                ).astype(qkva_dtype)
            # kr = k * exp(G_r - G) on columns before the block (exponent
            # <= 0); columns at or past the block zeroed.
            for j_s, j_k in T.Parallel(block_S, DK):
                if j_s < bi * SUB:
                    kr_shared[j_s, j_k] = (
                        k_shared[j_s, j_k].astype(accum_dtype)
                        * T.exp2(
                            (g_shared[bi * SUB, j_k] - g_shared[j_s, j_k]) * L2E
                        )
                    ).astype(qkva_dtype)
                else:
                    kr_shared[j_s, j_k] = 0
            T.gemm(
                kl_shared, kr_shared, arow_fragment,
                transpose_B=True, clear_accum=True,
            )
            for j_s, j_t in T.Parallel(SUB, block_S):
                if j_t < bi * SUB:
                    a64_fragment[bi * SUB + j_s, j_t] = arow_fragment[j_s, j_t]

        # Diagonal sub-blocks: elementwise with per-pair exponents — the
        # only form bounded for arbitrary gate magnitudes inside a block.
        for bi in T.serial(NSUB):
            T.clear(diag_acc)
            for j_s, j_t in T.Parallel(SUB, SUB):
                if j_s > j_t:
                    for j_k in T.serial(DK):
                        diag_acc[j_s, j_t] += (
                            b_shared[bi * SUB + j_s, j_k].astype(accum_dtype)
                            * k_shared[bi * SUB + j_s, j_k].astype(accum_dtype)
                            * k_shared[bi * SUB + j_t, j_k].astype(accum_dtype)
                            * T.exp2(
                                (
                                    g_shared[bi * SUB + j_s, j_k]
                                    - g_shared[bi * SUB + j_t, j_k]
                                )
                                * L2E
                            )
                        )
            for j_s, j_t in T.Parallel(SUB, SUB):
                if j_s > j_t:
                    a64_fragment[bi * SUB + j_s, bi * SUB + j_t] = diag_acc[
                        j_s, j_t
                    ]

        # A = I + StrictLower(A)
        for j_s, j_t in T.Parallel(block_S, block_S):
            if j_s < j_t:
                a64_fragment[j_s, j_t] = 0
            elif j_s == j_t:
                a64_fragment[j_s, j_t] = 1

        # ------- blocked triangular inversion: identical to gdn -------
        # Prepare inversion input
        for j_s, j_t in T.Parallel(block_S, block_S):
            if j_s >= 32 and j_t < 32:
                a32o_shared[j_s - 32, j_t] = -a64_fragment[j_s, j_t]
            elif (j_s // 16) == (j_t // 16) + 1:
                a16o_shared[j_s // 32, j_s % 16, j_t % 16] = -a64_fragment[j_s, j_t]
            elif (j_s // 16) == (j_t // 16):
                a16i_shared[j_s // 16, j_s % 16, j_t % 16] = a64_fragment[j_s, j_t]

        # Diagonal 4x16x16
        T.clear(a16i_row)
        for k_s in T.unroll(1, 16):
            for j_s, k_t in T.Parallel(4, 16):
                if k_t < k_s:
                    a16i_row[j_s, k_t] = a16i_shared[j_s, k_s, k_t]
            T.clear(a16i_sum)
            for k_r in T.unroll(k_s):
                for j_s, k_t in T.Parallel(4, 16):
                    a16i_sum[j_s, k_t] -= (
                        a16i_shared[j_s, k_r, k_t] * a16i_row[j_s, k_r]
                    )
            for j_s, k_t in T.Parallel(4, 16):
                if k_t < k_s:
                    a16i_shared[j_s, k_s, k_t] = a16i_sum[j_s, k_t]

        # First level 2x16x16
        T.clear(a16o_fragment)
        for k_r in T.unroll(16):
            for j_s, k_s, k_t in T.Parallel(2, 16, 16):
                a16o_fragment[j_s, k_s, k_t] += (
                    a16i_shared[j_s * 2 + 1, k_s, k_r] * a16o_shared[j_s, k_r, k_t]
                )
        for j_s, k_s, k_t in T.Parallel(2, 16, 16):
            a16o_shared[j_s, k_t, k_s] = a16o_fragment[j_s, k_s, k_t]
        T.clear(a16o_fragment)
        for k_r in T.unroll(16):
            for j_s, k_s, k_t in T.Parallel(2, 16, 16):
                a16o_fragment[j_s, k_s, k_t] += (
                    a16o_shared[j_s, k_r, k_s] * a16i_shared[j_s * 2, k_r, k_t]
                )
        T.copy(a16o_fragment, a16o_shared[:, 0:16, 0:16])

        # Second level 1x32x32
        for j_s, k_s, k_t in T.Parallel(2, 32, 32):
            if k_s < 16 and k_t >= 16:
                a32i_fragment[j_s, k_s, k_t] = 0
        for j_s, k_s, k_t in T.Parallel(2, 32, 32):
            if k_s >= 16 and k_t < 16:
                a32i_fragment[j_s, k_s, k_t] = a16o_shared[j_s, k_s - 16, k_t]
        for j_s, k_s, k_t in T.Parallel(2, 32, 32):
            if k_s // 16 == k_t // 16:
                a32i_fragment[j_s, k_s, k_t] = a16i_shared[
                    j_s * 2 + k_s // 16, k_s % 16, k_t % 16
                ]
        for j_s, k_s, k_t in T.Parallel(2, 32, 32):
            if j_s == 0:
                a32i0_shared[k_s, k_t] = a32i_fragment[j_s, k_s, k_t]
            else:
                a32i1_shared[k_s, k_t] = a32i_fragment[j_s, k_s, k_t]
        T.gemm(a32i1_shared, a32o_shared, a32o_fragment, clear_accum=True)
        T.copy(a32o_fragment, a32o_shared)
        T.gemm(a32o_shared, a32i0_shared, a32o_fragment, clear_accum=True)

        # Combine inversion output
        for j_s, k_s, k_t in T.Parallel(2, 32, 32):
            a64_shared[j_s * 32 + k_s, j_s * 32 + k_t] = a32i_fragment[
                j_s, k_s, k_t
            ]
        for k_s, k_t in T.Parallel(32, 32):
            a64_shared[32 + k_s, k_t] = a32o_fragment[k_s, k_t]
        for k_s, k_t in T.Parallel(32, 32):
            a64_shared[k_s, 32 + k_t] = 0

        # Save A (unmasked)
        if right <= seq_end_idx:
            T.copy(a64_shared, a[bb, left:right, bh, 0:block_S])
        else:
            for j_s, j_t in T.Parallel(block_S, block_S):
                if left + j_s < seq_end_idx:
                    a[bb, left + j_s, bh, j_t] = a64_shared[j_s, j_t]

    if is_varlen:

        @T.prim_func
        def tilelang_kkt_solve_2_kernel(
            k: T.Tensor(k_shape, dtype=qkva_dtype),
            g: T.Tensor(g_shape, dtype=g_dtype),
            b: T.Tensor(b_shape, dtype=qkva_dtype),
            cu_seqlens: T.Tensor([real_batch_size + 1], dtype=seqlen_dtype),
            chunk_indices: T.Tensor([num_chunks, 2], dtype=seqlen_dtype),
            a: T.Tensor(a_shape, dtype=qkva_dtype),
        ):
            with T.Kernel(num_chunks * H, threads=128) as (bch,):
                bc, bh = bch // H, bch % H
                bhg = bh // (H // Hg)

                batch_idx = T.alloc_var("int32")
                chunk_idx = T.alloc_var("int32")
                seq_start_idx = T.alloc_var("int32")
                seq_end_idx = T.alloc_var("int32")

                bb = 0
                batch_idx = chunk_indices[bc, 0]
                chunk_idx = chunk_indices[bc, 1]
                seq_start_idx = cu_seqlens[batch_idx]
                seq_end_idx = cu_seqlens[batch_idx + 1]

                kernel_body(
                    bb,
                    bc,
                    bh,
                    bhg,
                    batch_idx,
                    chunk_idx,
                    seq_start_idx,
                    seq_end_idx,
                    k,
                    g,
                    b,
                    a,
                )

    else:

        @T.prim_func
        def tilelang_kkt_solve_2_kernel(
            k: T.Tensor(k_shape, dtype=qkva_dtype),
            g: T.Tensor(g_shape, dtype=g_dtype),
            b: T.Tensor(b_shape, dtype=qkva_dtype),
            a: T.Tensor(a_shape, dtype=qkva_dtype),
            num_chunks: T.int32,
        ):
            with T.Kernel(num_chunks * H, threads=128) as (bch,):
                bc, bh = bch // H, bch % H
                bhg = bh // (H // Hg)

                batch_idx = T.alloc_var("int32")
                chunk_idx = T.alloc_var("int32")
                seq_start_idx = T.alloc_var("int32")
                seq_end_idx = T.alloc_var("int32")

                bb = bc % data_batch_size
                batch_idx = bb
                chunk_idx = bc // data_batch_size
                seq_start_idx = 0
                seq_end_idx = num_tokens

                kernel_body(
                    bb,
                    bc,
                    bh,
                    bhg,
                    batch_idx,
                    chunk_idx,
                    seq_start_idx,
                    seq_end_idx,
                    k,
                    g,
                    b,
                    a,
                )

    return tilelang_kkt_solve_2_kernel


def kkt_solve(
    k: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    chunk_size: int = 64,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    """A = (I + StrictLower(M))^{-1} per 64-token chunk.

    k: [B, T, Hg, K] (bf16/fp16); g: [B, T, H, K] fp32 CHUNK-LOCAL
    INCLUSIVE cumsum of the log decay; b: [B, T, H, K] (same dtype as k).
    Returns a: [B, T, H, chunk_size] in k.dtype.
    """
    batch_size, num_tokens, Hg, K = k.shape
    H = b.shape[2]
    assert K == 128
    assert chunk_size == 64
    assert g.dtype == torch.float32, "g must be the fp32 chunk-local cumsum"
    assert g.shape == b.shape == (batch_size, num_tokens, H, K)

    if cu_seqlens is None:
        num_chunks = batch_size * tilelang.cdiv(num_tokens, chunk_size)
        seqlen_dtype = "int32"
        is_varlen = False
    else:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        seqlen_dtype = cu_seqlens.dtype
        is_varlen = True

    a = torch.empty(
        (batch_size, num_tokens, H, chunk_size), dtype=k.dtype, device=k.device
    )

    kernel = tilelang_kkt_solve_2(
        H,
        Hg,
        K,
        chunk_size,
        accum_dtype="float32",
        qkva_dtype=k.dtype,
        g_dtype=g.dtype,
        seqlen_dtype=seqlen_dtype,
        is_varlen=is_varlen,
    )
    if is_varlen:
        kernel(k, g, b, cu_seqlens, chunk_indices, a)
    else:
        kernel(k, g, b, a, num_chunks)

    return a
