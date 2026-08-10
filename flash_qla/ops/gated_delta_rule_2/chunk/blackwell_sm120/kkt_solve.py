# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""GDN2 kkt_solve for SM120 (chunk 32): A = (I + StrictLower(M))^{-1},

    M_ij = sum_c (b_ic k_ic) exp(G_ic - G_jc) k_jc,   G = chunk-local cumsum(g)

Per-channel decay handling as in the hopper variant (see DESIGN_GDR2.md):
the 16-token diagonal sub-blocks are computed elementwise (pairwise
exponents, bounded on the used triangle for any gate magnitude); the one
off-diagonal sub-block uses two bounded factors rebased at the row-block
start and a GEMM. The 16 -> 32 blocked triangular inversion is copied
from the gdn sm120 kernel unchanged.
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
    skip_off16=False,
    skip_off8=False,
    skip_diag8=False,
    skip_inv=False,
):
    data_batch_size = T.dynamic("data_batch_size")
    real_batch_size = T.dynamic("real_batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    block_S = chunk_size
    SUB = 16

    k_shape = (data_batch_size, num_tokens, Hg, DK)
    g_shape = (data_batch_size, num_tokens, H, DK)
    b_shape = (data_batch_size, num_tokens, H, DK)
    a_shape = (data_batch_size, num_tokens, H, chunk_size)

    @T.macro
    def kernel_body_32(
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
        q,
        a,
        attn,
    ):
        left = seq_start_idx + chunk_idx * block_S
        right = left + block_S

        k_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
        b_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
        g_shared = T.alloc_shared((block_S, DK), dtype=accum_dtype)
        q_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
        kl_shared = T.alloc_shared((SUB, DK), dtype=qkva_dtype)
        ql_shared = T.alloc_shared((SUB, DK), dtype=qkva_dtype)
        attnoff_fragment = T.alloc_fragment((SUB, block_S), dtype=accum_dtype)
        attnoff_shared = T.alloc_shared((SUB, block_S), dtype=accum_dtype)
        attn_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
        klq8_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
        kr8_shared = T.alloc_shared((SUB, DK), dtype=qkva_dtype)
        off8_fragment = T.alloc_fragment((block_S, SUB), dtype=accum_dtype)
        off8_shared = T.alloc_shared((block_S, SUB), dtype=accum_dtype)
        diag8a_shared = T.alloc_shared((4, 8, 8), dtype=accum_dtype)
        diag8q_shared = T.alloc_shared((4, 8, 8), dtype=accum_dtype)
        kr_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
        arow_fragment = T.alloc_fragment((SUB, block_S), dtype=accum_dtype)
        arow_shared = T.alloc_shared((SUB, block_S), dtype=accum_dtype)
        a32_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
        diag_local = T.alloc_local((1), dtype=accum_dtype)
        diagq_local = T.alloc_local((1), dtype=accum_dtype)
        e_local = T.alloc_local((1), dtype=accum_dtype)

        a16i_row = T.alloc_fragment((2, 16), dtype=accum_dtype)
        a16i_sum = T.alloc_fragment((2, 16), dtype=accum_dtype)

        a16i_shared = T.alloc_shared((2, 17, 16), dtype=accum_dtype)
        a16o_shared = T.alloc_shared((1, 17, 16), dtype=accum_dtype)
        a16o_fragment = T.alloc_fragment((1, 16, 16), dtype=accum_dtype)

        a32_shared = T.alloc_shared((block_S, block_S), dtype=accum_dtype)

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
                q_shared[j_s, j_k] = q[bb, left + j_s, bhg, j_k]
            else:
                b_shared[j_s, j_k] = 0
                g_shared[j_s, j_k] = g[bb, seq_end_idx - 1, bh, j_k]
                q_shared[j_s, j_k] = 0
        if right <= seq_end_idx:
            T.ptx_wait_group(0)

        T.clear(a32_fragment)

        if not skip_off16:
            # Off-diagonal sub-block (rows 16..31 vs cols 0..15): bounded
            # two-factor form rebased at the row-block start r = 16.
            for j_s, j_k in T.Parallel(SUB, DK):
                kl_shared[j_s, j_k] = (
                    b_shared[SUB + j_s, j_k].astype(accum_dtype)
                    * k_shared[SUB + j_s, j_k].astype(accum_dtype)
                    * T.exp2(
                        (g_shared[SUB + j_s, j_k] - g_shared[SUB, j_k]) * L2E
                    )
                ).astype(qkva_dtype)
            for j_s, j_k in T.Parallel(block_S, DK):
                if j_s < SUB:
                    kr_shared[j_s, j_k] = (
                        k_shared[j_s, j_k].astype(accum_dtype)
                        * T.exp2((g_shared[SUB, j_k] - g_shared[j_s, j_k]) * L2E)
                    ).astype(qkva_dtype)
                else:
                    kr_shared[j_s, j_k] = 0
            T.gemm(
                kl_shared, kr_shared, arow_fragment,
                transpose_B=True, clear_accum=True,
            )
            # Fragment-to-fragment moves at shifted indices cross thread
            # ownership and lower to atomics; route through shared instead.
            T.copy(arow_fragment, arow_shared)

        if not skip_off16:
            # Off-diagonal ATTENTION block: same kr factor, q on the rows.
            for j_s, j_k in T.Parallel(SUB, DK):
                ql_shared[j_s, j_k] = (
                    q_shared[SUB + j_s, j_k].astype(accum_dtype)
                    * T.exp2(
                        (g_shared[SUB + j_s, j_k] - g_shared[SUB, j_k]) * L2E
                    )
                ).astype(qkva_dtype)
            T.gemm(
                ql_shared, kr_shared, attnoff_fragment,
                transpose_B=True, clear_accum=True,
            )
            T.copy(attnoff_fragment, attnoff_shared)

        if not skip_off8:
            # 8-token sub-blocking of the same-16-block work: the hi-half x
            # lo-half 8x8 quadrants go to tensor cores — BOTH 16-blocks batched
            # into one m16 gemm per gram (only the two diagonal quadrants of
            # the [16,16] result are valid and read; the cross quadrants are
            # finite garbage by construction). Rebase point: the block's row 8.
            for j_s, j_k in T.Parallel(block_S, DK):
                if j_s < SUB:
                    klq8_shared[j_s, j_k] = (
                        b_shared[(j_s // 8) * SUB + 8 + (j_s % 8), j_k].astype(accum_dtype)
                        * k_shared[(j_s // 8) * SUB + 8 + (j_s % 8), j_k].astype(accum_dtype)
                        * T.exp2(
                            (
                                g_shared[(j_s // 8) * SUB + 8 + (j_s % 8), j_k]
                                - g_shared[(j_s // 8) * SUB + 8, j_k]
                            )
                            * L2E
                        )
                    ).astype(qkva_dtype)
                else:
                    klq8_shared[j_s, j_k] = (
                        q_shared[((j_s - SUB) // 8) * SUB + 8 + (j_s % 8), j_k].astype(accum_dtype)
                        * T.exp2(
                            (
                                g_shared[((j_s - SUB) // 8) * SUB + 8 + (j_s % 8), j_k]
                                - g_shared[((j_s - SUB) // 8) * SUB + 8, j_k]
                            )
                            * L2E
                        )
                    ).astype(qkva_dtype)
            for j_s, j_k in T.Parallel(SUB, DK):
                kr8_shared[j_s, j_k] = (
                    k_shared[(j_s // 8) * SUB + (j_s % 8), j_k].astype(accum_dtype)
                    * T.exp2(
                        (
                            g_shared[(j_s // 8) * SUB + 8, j_k]
                            - g_shared[(j_s // 8) * SUB + (j_s % 8), j_k]
                        )
                        * L2E
                    )
                ).astype(qkva_dtype)
            T.gemm(
                klq8_shared, kr8_shared, off8_fragment,
                transpose_B=True, clear_accum=True,
            )
            T.copy(off8_fragment, off8_shared)

        if not skip_diag8:
            # The four 8x8 diagonals stay elementwise (no bounded factoring
            # exists inside a block) — 47% fewer pairs than 16x16 diagonals,
            # one exp2 feeding both grams, strip-unrolled channel loop.
            for b8 in T.serial(4):
                for j_s, j_t in T.Parallel(8, 8):
                    diag_local[0] = 0.0
                    diagq_local[0] = 0.0
                    if j_s >= j_t:
                        for j_ko in T.serial(8):
                            for j_ki in T.unroll(16):
                                e_local[0] = T.exp2(
                                    (
                                        g_shared[(b8 // 2) * SUB + (b8 % 2) * 8 + j_s,
                                                 j_ko * 16 + j_ki]
                                        - g_shared[(b8 // 2) * SUB + (b8 % 2) * 8 + j_t,
                                                   j_ko * 16 + j_ki]
                                    )
                                    * L2E
                                ) * k_shared[(b8 // 2) * SUB + (b8 % 2) * 8 + j_t,
                                             j_ko * 16 + j_ki].astype(accum_dtype)
                                diagq_local[0] += (
                                    q_shared[(b8 // 2) * SUB + (b8 % 2) * 8 + j_s,
                                             j_ko * 16 + j_ki].astype(accum_dtype)
                                    * e_local[0]
                                )
                                diag_local[0] += (
                                    b_shared[(b8 // 2) * SUB + (b8 % 2) * 8 + j_s,
                                             j_ko * 16 + j_ki].astype(accum_dtype)
                                    * k_shared[(b8 // 2) * SUB + (b8 % 2) * 8 + j_s,
                                               j_ko * 16 + j_ki].astype(accum_dtype)
                                    * e_local[0]
                                )
                    if j_s > j_t:
                        diag8a_shared[b8, j_s, j_t] = diag_local[0]
                    else:
                        diag8a_shared[b8, j_s, j_t] = 0.0
                    diag8q_shared[b8, j_s, j_t] = diagq_local[0]

        # Assemble and store the attention matrix (inclusive tril).
        for j_s, j_t in T.Parallel(block_S, block_S):
            if (j_s // 8) == (j_t // 8):
                attn_fragment[j_s, j_t] = diag8q_shared[
                    j_s // 8, j_s % 8, j_t % 8
                ]
            elif ((j_s // SUB) == (j_t // SUB)) and ((j_s // 8) > (j_t // 8)):
                attn_fragment[j_s, j_t] = off8_shared[
                    SUB + (j_s // SUB) * 8 + (j_s % SUB) - 8,
                    (j_t // SUB) * 8 + (j_t % SUB),
                ]
            elif (j_s // SUB) > (j_t // SUB):
                attn_fragment[j_s, j_t] = attnoff_shared[j_s - SUB, j_t]
            else:
                attn_fragment[j_s, j_t] = 0
        T.copy(attn_fragment, a32_shared)
        if right <= seq_end_idx:
            T.copy(a32_shared, attn[bb, left:right, bh, 0:block_S])
        else:
            for j_s, j_t in T.Parallel(block_S, block_S):
                if left + j_s < seq_end_idx:
                    attn[bb, left + j_s, bh, j_t] = a32_shared[j_s, j_t]

        # Assemble A = I + StrictLower(A): every thread writes only its
        # own a32_fragment elements, reading from shared.
        for j_s, j_t in T.Parallel(block_S, block_S):
            if j_s < j_t:
                a32_fragment[j_s, j_t] = 0
            elif j_s == j_t:
                a32_fragment[j_s, j_t] = 1
            elif (j_s // 8) == (j_t // 8):
                a32_fragment[j_s, j_t] = diag8a_shared[
                    j_s // 8, j_s % 8, j_t % 8
                ]
            elif (j_s // SUB) == (j_t // SUB):
                a32_fragment[j_s, j_t] = off8_shared[
                    (j_s // SUB) * 8 + (j_s % SUB) - 8,
                    (j_t // SUB) * 8 + (j_t % SUB),
                ]
            else:
                a32_fragment[j_s, j_t] = arow_shared[j_s - SUB, j_t]

        if not skip_inv:
            # ------- 16 -> 32 blocked inversion: identical to gdn sm120 -------
            for j_s, j_t in T.Parallel(block_S, block_S):
                if (j_s // 16) == (j_t // 16) + 1:
                    a16o_shared[j_s // 32, j_s % 16, j_t % 16] = -a32_fragment[
                        j_s, j_t
                    ]
                elif (j_s // 16) == (j_t // 16):
                    a16i_shared[j_s // 16, j_s % 16, j_t % 16] = a32_fragment[
                        j_s, j_t
                    ]

            for k_s in T.unroll(1, 16):
                for j_s, k_t in T.Parallel(2, 16):
                    if k_t < k_s:
                        a16i_row[j_s, k_t] = a16i_shared[j_s, k_s, k_t]
                T.clear(a16i_sum)
                for k_r in T.unroll(k_s):
                    for j_s, k_t in T.Parallel(2, 16):
                        a16i_sum[j_s, k_t] -= (
                            a16i_shared[j_s, k_r, k_t] * a16i_row[j_s, k_r]
                        )
                for j_s, k_t in T.Parallel(2, 16):
                    if k_t < k_s:
                        a16i_shared[j_s, k_s, k_t] = a16i_sum[j_s, k_t]

            T.clear(a16o_fragment)
            for k_r in T.unroll(16):
                for j_s, k_s, k_t in T.Parallel(1, 16, 16):
                    a16o_fragment[j_s, k_s, k_t] += (
                        a16i_shared[j_s * 2 + 1, k_s, k_r]
                        * a16o_shared[j_s, k_r, k_t]
                    )
            for j_s, k_s, k_t in T.Parallel(1, 16, 16):
                a16o_shared[j_s, k_t, k_s] = a16o_fragment[j_s, k_s, k_t]
            T.clear(a16o_fragment)
            for k_r in T.unroll(16):
                for j_s, k_s, k_t in T.Parallel(1, 16, 16):
                    a16o_fragment[j_s, k_s, k_t] += (
                        a16o_shared[j_s, k_r, k_s]
                        * a16i_shared[j_s * 2, k_r, k_t]
                    )
            T.copy(a16o_fragment, a16o_shared[:, 0:16, 0:16])

            for k_s, k_t in T.Parallel(16, 16):
                a32_shared[k_s, k_t] = a16i_shared[0, k_s, k_t]
            for k_s, k_t in T.Parallel(16, 16):
                a32_shared[k_s, 16 + k_t] = 0
            for k_s, k_t in T.Parallel(16, 16):
                a32_shared[16 + k_s, k_t] = a16o_fragment[0, k_s, k_t]
            for k_s, k_t in T.Parallel(16, 16):
                a32_shared[16 + k_s, 16 + k_t] = a16i_shared[1, k_s, k_t]

        # Save A (unmasked)
        if right <= seq_end_idx:
            T.copy(a32_shared, a[bb, left:right, bh, 0:block_S])
        else:
            for j_s, j_t in T.Parallel(block_S, block_S):
                if left + j_s < seq_end_idx:
                    a[bb, left + j_s, bh, j_t] = a32_shared[j_s, j_t]

    if is_varlen:

        @T.prim_func
        def tilelang_kkt_solve_2_kernel(
            k: T.Tensor(k_shape, dtype=qkva_dtype),
            g: T.Tensor(g_shape, dtype=g_dtype),
            b: T.Tensor(b_shape, dtype=qkva_dtype),
            q: T.Tensor(k_shape, dtype=qkva_dtype),
            cu_seqlens: T.Tensor([real_batch_size + 1], dtype=seqlen_dtype),
            chunk_indices: T.Tensor([num_chunks, 2], dtype=seqlen_dtype),
            a: T.Tensor(a_shape, dtype=qkva_dtype),
            attn: T.Tensor(a_shape, dtype=qkva_dtype),
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

                kernel_body_32(
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
                    q,
                    a,
                    attn,
                )

    else:

        @T.prim_func
        def tilelang_kkt_solve_2_kernel(
            k: T.Tensor(k_shape, dtype=qkva_dtype),
            g: T.Tensor(g_shape, dtype=g_dtype),
            b: T.Tensor(b_shape, dtype=qkva_dtype),
            q: T.Tensor(k_shape, dtype=qkva_dtype),
            a: T.Tensor(a_shape, dtype=qkva_dtype),
            attn: T.Tensor(a_shape, dtype=qkva_dtype),
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

                kernel_body_32(
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
                    q,
                    a,
                    attn,
                )

    return tilelang_kkt_solve_2_kernel


def kkt_solve(
    k: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    q: torch.Tensor,
    chunk_size: int = 32,
    cu_seqlens: Optional[torch.LongTensor] = None,
    **ablate,
):
    """A = (I + StrictLower(M))^{-1} per 32-token chunk.

    k: [B, T, Hg, K] (bf16/fp16); g: [B, T, H, K] fp32 CHUNK-LOCAL
    INCLUSIVE cumsum of the log decay; b: [B, T, H, K] (same dtype as k).
    Returns a: [B, T, H, chunk_size] in k.dtype.
    """
    k, g, b, q = k.contiguous(), g.contiguous(), b.contiguous(), q.contiguous()
    batch_size, num_tokens, Hg, K = k.shape
    H = b.shape[2]
    assert K == 128
    assert chunk_size == 32
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
    attn = torch.empty_like(a)

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
        **ablate,
    )
    if is_varlen:
        kernel(k, g, b, q, cu_seqlens, chunk_indices, a, attn)
    else:
        kernel(k, g, b, q, a, attn, num_chunks)

    return a, attn
