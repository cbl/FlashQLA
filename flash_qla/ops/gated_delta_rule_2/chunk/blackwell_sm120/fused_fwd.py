# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""GDN2 fused_fwd for SM120 (chunk 32): the output stage.

    o = scale * [ attn @ R  +  eq @ S_chunk ]

All elementwise work happens upstream: attn (the inclusive-tril decayed
q-gram) is emitted by kkt_solve alongside A from the SAME exponentials;
eq = e^G * q comes from the precompute; R and the per-chunk entering
states from prepare_h. What remains here is two GEMMs per (chunk, head)
block, fully parallel over chunks.
"""

from typing import Optional

import torch
import tilelang
import tilelang.language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_fused_fwd_2(
    H,
    DK,
    DV,
    chunk_size,
    scale,
    accum_dtype,
    qkva_dtype,
    h_dtype,
    o_dtype,
    include_intra=True,
    include_inter=True,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    block_S = chunk_size

    x_shape = (batch_size, num_tokens, H, DK)
    a_shape = (batch_size, num_tokens, H, chunk_size)
    r_shape = (batch_size, num_tokens, H, DV)
    h_shape = (batch_size, num_chunks, H, DK, DV)
    o_shape = (batch_size, num_tokens, H, DV)

    @T.prim_func
    def tilelang_fused_fwd_2_kernel(
        eq: T.Tensor(x_shape, dtype=qkva_dtype),
        attn: T.Tensor(a_shape, dtype=qkva_dtype),
        r: T.Tensor(r_shape, dtype=qkva_dtype),
        h: T.Tensor(h_shape, dtype=h_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
    ):
        with T.Kernel(batch_size * num_chunks * H, threads=256) as (bch,):
            bc, bh = bch // H, bch % H
            bb = bc % batch_size
            chunk_idx = bc // batch_size
            left = chunk_idx * block_S

            eq_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            attn_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            r_shared = T.alloc_shared((block_S, DV), dtype=qkva_dtype)
            h_shared = T.alloc_shared((DK, DV), dtype=h_dtype)
            o_fragment = T.alloc_fragment((block_S, DV), dtype=accum_dtype)

            for j_s, j_k in T.Parallel(block_S, DK):
                if left + j_s < num_tokens:
                    eq_shared[j_s, j_k] = eq[bb, left + j_s, bh, j_k]
                else:
                    eq_shared[j_s, j_k] = 0
            for j_s, j_t in T.Parallel(block_S, block_S):
                if left + j_s < num_tokens:
                    attn_shared[j_s, j_t] = attn[bb, left + j_s, bh, j_t]
                else:
                    attn_shared[j_s, j_t] = 0
            for j_s, j_v in T.Parallel(block_S, DV):
                if left + j_s < num_tokens:
                    r_shared[j_s, j_v] = r[bb, left + j_s, bh, j_v]
                else:
                    r_shared[j_s, j_v] = 0
            T.copy(h[bb, chunk_idx, bh, 0:DK, 0:DV], h_shared)

            T.clear(o_fragment)
            if include_intra:
                T.gemm(attn_shared, r_shared, o_fragment, clear_accum=False)
            if include_inter:
                T.gemm(eq_shared, h_shared, o_fragment, clear_accum=False)

            for j_s, j_v in T.Parallel(block_S, DV):
                if left + j_s < num_tokens:
                    o[bb, left + j_s, bh, j_v] = (
                        o_fragment[j_s, j_v] * scale
                    ).astype(o_dtype)

    return tilelang_fused_fwd_2_kernel


def fused_fwd_2(
    eq: torch.Tensor,
    attn: torch.Tensor,
    r: torch.Tensor,
    h: torch.Tensor,
    scale: Optional[float] = None,
    include_intra: bool = True,
    include_inter: bool = True,
):
    """Outputs from per-chunk states. eq: [B, T, H, K] (e^G * q, from the
    precompute); attn: [B, T, H, chunk] (from kkt_solve); r: [B, T, H, V]
    and h: [B, N, H, K, V] (from prepare_h_2). Returns o [B, T, H, V]."""
    eq, attn, r, h = (x.contiguous() for x in (eq, attn, r, h))
    batch_size, num_tokens, H, K = eq.shape
    V = r.shape[-1]
    chunk_size = attn.shape[-1]
    assert K == 128 and V == 128
    if scale is None:
        scale = K**-0.5

    o = torch.empty(
        (batch_size, num_tokens, H, V), dtype=eq.dtype, device=eq.device
    )
    kernel = tilelang_fused_fwd_2(
        H,
        K,
        V,
        chunk_size,
        float(scale),
        accum_dtype="float32",
        qkva_dtype=eq.dtype,
        h_dtype=h.dtype,
        o_dtype=o.dtype,
        include_intra=include_intra,
        include_inter=include_inter,
    )
    kernel(eq, attn, r, h, o)
    return o
