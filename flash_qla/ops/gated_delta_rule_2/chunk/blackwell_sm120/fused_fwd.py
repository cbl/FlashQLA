# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""GDN2 fused_fwd for SM120 (chunk 32): the output stage.

    o_i = scale * [ sum_{j<=i} attn_ij R_j  +  (e^G * q)_i @ S_chunk ]
    attn_ij = sum_c q_ic k_jc e^{G_ic - G_jc}   (INCLUSIVE lower triangle)

R and the per-chunk entering states S come from prepare_h_2. The attn
matrix uses the same bounded sub-block structure as kkt_solve (16-token
diagonal blocks elementwise, the off-diagonal block as a two-factor
rebased GEMM); the inter-chunk read is a plain GEMM against the chunk
state. Chunks are independent here, so the grid parallelizes over
(chunk, head) — unlike the sequential prepare_h march.
"""

from typing import Optional

import torch
import tilelang
import tilelang.language as T

L2E = 1.442695  # log2(e): T.exp2(x * L2E) == exp(x)


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_fused_fwd_2(
    H,
    Hg,
    DK,
    DV,
    chunk_size,
    scale,
    accum_dtype,
    qkva_dtype,
    g_dtype,
    h_dtype,
    o_dtype,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    block_S = chunk_size
    SUB = 16

    qk_shape = (batch_size, num_tokens, Hg, DK)
    g_shape = (batch_size, num_tokens, H, DK)
    r_shape = (batch_size, num_tokens, H, DV)
    h_shape = (batch_size, num_chunks, H, DK, DV)
    o_shape = (batch_size, num_tokens, H, DV)

    @T.prim_func
    def tilelang_fused_fwd_2_kernel(
        q: T.Tensor(qk_shape, dtype=qkva_dtype),
        k: T.Tensor(qk_shape, dtype=qkva_dtype),
        g: T.Tensor(g_shape, dtype=g_dtype),
        r: T.Tensor(r_shape, dtype=qkva_dtype),
        h: T.Tensor(h_shape, dtype=h_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
    ):
        with T.Kernel(num_chunks * H, threads=128) as (bch,):
            bc, bh = bch // H, bch % H
            bhg = bh // (H // Hg)
            bb = bc % batch_size
            chunk_idx = bc // batch_size
            left = chunk_idx * block_S

            q_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            k_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            g_shared = T.alloc_shared((block_S, DK), dtype=accum_dtype)
            r_shared = T.alloc_shared((block_S, DV), dtype=qkva_dtype)
            h_shared = T.alloc_shared((DK, DV), dtype=h_dtype)
            ql_shared = T.alloc_shared((SUB, DK), dtype=qkva_dtype)
            kr_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            attn_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            off_fragment = T.alloc_fragment((SUB, block_S), dtype=accum_dtype)
            off_shared = T.alloc_shared((SUB, block_S), dtype=accum_dtype)
            o_fragment = T.alloc_fragment((block_S, DV), dtype=accum_dtype)
            diag_local = T.alloc_local((1), dtype=accum_dtype)

            # Loads (zero-padded tail; g tail-filled with last valid row)
            for j_s, j_k in T.Parallel(block_S, DK):
                if left + j_s < num_tokens:
                    q_shared[j_s, j_k] = q[bb, left + j_s, bhg, j_k]
                    k_shared[j_s, j_k] = k[bb, left + j_s, bhg, j_k]
                    g_shared[j_s, j_k] = g[bb, left + j_s, bh, j_k]
                else:
                    q_shared[j_s, j_k] = 0
                    k_shared[j_s, j_k] = 0
                    g_shared[j_s, j_k] = g[bb, num_tokens - 1, bh, j_k]
            for j_s, j_v in T.Parallel(block_S, DV):
                if left + j_s < num_tokens:
                    r_shared[j_s, j_v] = r[bb, left + j_s, bh, j_v]
                else:
                    r_shared[j_s, j_v] = 0
            T.copy(h[bb, chunk_idx, bh, 0:DK, 0:DV], h_shared)

            # Off-diagonal attn block (rows 16..31 vs cols 0..15), bounded
            # two-factor form rebased at the row-block start.
            for j_s, j_k in T.Parallel(SUB, DK):
                ql_shared[j_s, j_k] = (
                    q_shared[SUB + j_s, j_k].astype(accum_dtype)
                    * T.exp2(
                        (g_shared[SUB + j_s, j_k] - g_shared[SUB, j_k]) * L2E
                    )
                ).astype(qkva_dtype)
            for j_s, j_k in T.Parallel(block_S, DK):
                if j_s < SUB:
                    kr_shared[j_s, j_k] = (
                        k_shared[j_s, j_k].astype(accum_dtype)
                        * T.exp2(
                            (g_shared[SUB, j_k] - g_shared[j_s, j_k]) * L2E
                        )
                    ).astype(qkva_dtype)
                else:
                    kr_shared[j_s, j_k] = 0
            T.gemm(
                ql_shared, kr_shared, off_fragment,
                transpose_B=True, clear_accum=True,
            )
            T.copy(off_fragment, off_shared)

            # Diagonal blocks elementwise, INCLUSIVE (i >= j), then
            # assemble attn in shared for the @R gemm.
            for bi in T.serial(2):
                for j_s, j_t in T.Parallel(SUB, SUB):
                    diag_local[0] = 0.0
                    if j_s >= j_t:
                        for j_k in T.serial(DK):
                            diag_local[0] += (
                                q_shared[bi * SUB + j_s, j_k].astype(accum_dtype)
                                * k_shared[bi * SUB + j_t, j_k].astype(accum_dtype)
                                * T.exp2(
                                    (
                                        g_shared[bi * SUB + j_s, j_k]
                                        - g_shared[bi * SUB + j_t, j_k]
                                    )
                                    * L2E
                                )
                            )
                    attn_shared[bi * SUB + j_s, bi * SUB + j_t] = (
                        diag_local[0]
                    ).astype(qkva_dtype)
            for j_s, j_t in T.Parallel(block_S, block_S):
                if (j_s // SUB) > (j_t // SUB):
                    attn_shared[j_s, j_t] = (
                        off_shared[j_s - SUB, j_t]
                    ).astype(qkva_dtype)
                elif (j_s // SUB) < (j_t // SUB):
                    attn_shared[j_s, j_t] = 0

            # o = attn @ R + (e^G * q) @ S  (eq reuses kr_shared)
            T.gemm(attn_shared, r_shared, o_fragment, clear_accum=True)
            for j_s, j_k in T.Parallel(block_S, DK):
                kr_shared[j_s, j_k] = (
                    q_shared[j_s, j_k].astype(accum_dtype)
                    * T.exp2(g_shared[j_s, j_k] * L2E)
                ).astype(qkva_dtype)
            T.gemm(kr_shared, h_shared, o_fragment, clear_accum=False)

            for j_s, j_v in T.Parallel(block_S, DV):
                if left + j_s < num_tokens:
                    o[bb, left + j_s, bh, j_v] = (
                        o_fragment[j_s, j_v] * scale
                    ).astype(o_dtype)

    return tilelang_fused_fwd_2_kernel


def fused_fwd_2(
    q: torch.Tensor,
    k: torch.Tensor,
    g_cs: torch.Tensor,
    r: torch.Tensor,
    h: torch.Tensor,
    scale: Optional[float] = None,
):
    """Outputs from per-chunk states. q/k: [B, T, Hg, K]; g_cs:
    [B, T, H, K] fp32 chunk-local cumsum; r: [B, T, H, V] from
    prepare_h_2; h: [B, N, H, K, V] entering states. Returns
    o [B, T, H, V] in q.dtype."""
    q, k, g_cs, r, h = (x.contiguous() for x in (q, k, g_cs, r, h))
    batch_size, num_tokens, Hg, K = q.shape
    H, V = r.shape[2], r.shape[3]
    assert K == 128 and V == 128
    assert g_cs.dtype == torch.float32
    if scale is None:
        scale = K**-0.5
    num_chunks = h.shape[1]

    o = torch.empty(
        (batch_size, num_tokens, H, V), dtype=q.dtype, device=q.device
    )
    kernel = tilelang_fused_fwd_2(
        H,
        Hg,
        K,
        V,
        32,
        float(scale),
        accum_dtype="float32",
        qkva_dtype=q.dtype,
        g_dtype=g_cs.dtype,
        h_dtype=h.dtype,
        o_dtype=o.dtype,
    )
    kernel(q, k, g_cs, r, h, o)
    return o
