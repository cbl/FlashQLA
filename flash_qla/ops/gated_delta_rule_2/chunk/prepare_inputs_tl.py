# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Fused decay-folding precompute (see prepare_inputs.py for semantics).

One kernel, one pass: per (chunk, head) block, load the gate/operand
tiles once, run the 32-step per-channel cumsum in registers, and emit
all five outputs. Replaces ~15 elementwise/cumsum torch launches that
otherwise dominate the forward. GVA is handled by indexing k through the
head-group map instead of materializing a repeat.
"""

import torch
import tilelang
import tilelang.language as T

L2E = 1.442695


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_prepare_inputs_2(
    H,
    Hg,
    DK,
    DV,
    chunk_size,
    accum_dtype,
    qkva_dtype,
    g_in_dtype,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    block_S = chunk_size

    k_shape = (batch_size, num_tokens, Hg, DK)
    x_shape = (batch_size, num_tokens, H, DK)
    v_shape = (batch_size, num_tokens, H, DV)
    gend_shape = (batch_size, num_chunks, H, DK)

    @T.prim_func
    def tilelang_prepare_inputs_2_kernel(
        q: T.Tensor(k_shape, dtype=qkva_dtype),
        k: T.Tensor(k_shape, dtype=qkva_dtype),
        v: T.Tensor(v_shape, dtype=qkva_dtype),
        g: T.Tensor(x_shape, dtype=g_in_dtype),
        b: T.Tensor(x_shape, dtype=qkva_dtype),
        w: T.Tensor(v_shape, dtype=qkva_dtype),
        g_cs: T.Tensor(x_shape, dtype=accum_dtype),
        eq: T.Tensor(x_shape, dtype=qkva_dtype),
        ekb: T.Tensor(x_shape, dtype=qkva_dtype),
        kte: T.Tensor(x_shape, dtype=qkva_dtype),
        mv: T.Tensor(v_shape, dtype=qkva_dtype),
        gend: T.Tensor(gend_shape, dtype=accum_dtype),
    ):
        with T.Kernel(batch_size * num_chunks * H, threads=128) as (bch,):
            bc, bh = bch // H, bch % H
            bhg = bh // (H // Hg)
            bb = bc % batch_size
            chunk_idx = bc // batch_size
            left = chunk_idx * block_S

            gcs_shared = T.alloc_shared((block_S, DK), dtype=accum_dtype)
            acc = T.alloc_local((1), dtype=accum_dtype)

            # Per-channel inclusive cumsum over the chunk, zero-padded
            # past the sequence end (so G_end matches the last valid row).
            for j_k in T.Parallel(DK):
                acc[0] = 0.0
                for j_s in T.serial(block_S):
                    if left + j_s < num_tokens:
                        acc[0] += g[bb, left + j_s, bh, j_k].astype(accum_dtype)
                    gcs_shared[j_s, j_k] = acc[0]

            for j_s, j_k in T.Parallel(block_S, DK):
                if left + j_s < num_tokens:
                    g_cs[bb, left + j_s, bh, j_k] = gcs_shared[j_s, j_k]
                    eq[bb, left + j_s, bh, j_k] = (
                        T.exp2(gcs_shared[j_s, j_k] * L2E)
                        * q[bb, left + j_s, bhg, j_k].astype(accum_dtype)
                    ).astype(qkva_dtype)
                    ekb[bb, left + j_s, bh, j_k] = (
                        T.exp2(gcs_shared[j_s, j_k] * L2E)
                        * b[bb, left + j_s, bh, j_k].astype(accum_dtype)
                        * k[bb, left + j_s, bhg, j_k].astype(accum_dtype)
                    ).astype(qkva_dtype)
                    kte[bb, left + j_s, bh, j_k] = (
                        T.exp2(
                            (gcs_shared[block_S - 1, j_k] - gcs_shared[j_s, j_k])
                            * L2E
                        )
                        * k[bb, left + j_s, bhg, j_k].astype(accum_dtype)
                    ).astype(qkva_dtype)
            for j_s, j_v in T.Parallel(block_S, DV):
                if left + j_s < num_tokens:
                    mv[bb, left + j_s, bh, j_v] = (
                        w[bb, left + j_s, bh, j_v].astype(accum_dtype)
                        * v[bb, left + j_s, bh, j_v].astype(accum_dtype)
                    ).astype(qkva_dtype)
            for j_k in T.Parallel(DK):
                gend[bb, chunk_idx, bh, j_k] = T.exp2(
                    gcs_shared[block_S - 1, j_k] * L2E
                )

    return tilelang_prepare_inputs_2_kernel


def prepare_inputs_2_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    chunk_size: int,
):
    """Drop-in for ``prepare_inputs_2`` (same outputs, one kernel)."""
    q, k, v, g, b, w = (x.contiguous() for x in (q, k, v, g, b, w))
    batch_size, num_tokens, Hg, K = k.shape
    H, V = v.shape[2], v.shape[3]
    num_chunks = tilelang.cdiv(num_tokens, chunk_size)

    g_cs = torch.empty(
        (batch_size, num_tokens, H, K), dtype=torch.float32, device=k.device
    )
    eq = torch.empty_like(g_cs, dtype=q.dtype)
    ekb = torch.empty_like(g_cs, dtype=k.dtype)
    kte = torch.empty_like(ekb)
    mv = torch.empty(
        (batch_size, num_tokens, H, V), dtype=v.dtype, device=k.device
    )
    gend = torch.empty(
        (batch_size, num_chunks, H, K), dtype=torch.float32, device=k.device
    )
    kernel = tilelang_prepare_inputs_2(
        H,
        Hg,
        K,
        V,
        chunk_size,
        accum_dtype="float32",
        qkva_dtype=k.dtype,
        g_in_dtype=g.dtype,
    )
    kernel(q, k, v, g, b, w, g_cs, eq, ekb, kte, mv, gend)
    return g_cs, eq, ekb, kte, mv, gend


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_gcs_2(
    H,
    DK,
    chunk_size,
    accum_dtype,
    g_in_dtype,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    block_S = chunk_size

    x_in = (batch_size, num_tokens, H, DK)
    x_out = (batch_size, num_tokens, H, DK)

    @T.prim_func
    def tilelang_gcs_2_kernel(
        g: T.Tensor(x_in, dtype=g_in_dtype),
        g_cs: T.Tensor(x_out, dtype=accum_dtype),
        num_chunks: T.int32,
    ):
        with T.Kernel(batch_size * num_chunks * H, threads=128) as (bch,):
            bc, bh = bch // H, bch % H
            bb = bc % batch_size
            chunk_idx = bc // batch_size
            left = chunk_idx * block_S

            acc = T.alloc_local((1), dtype=accum_dtype)
            for j_k in T.Parallel(DK):
                acc[0] = 0.0
                for j_s in T.serial(block_S):
                    if left + j_s < num_tokens:
                        acc[0] += g[bb, left + j_s, bh, j_k].astype(accum_dtype)
                        g_cs[bb, left + j_s, bh, j_k] = acc[0]

    return tilelang_gcs_2_kernel


def gcs_2(g: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Chunk-local inclusive cumsum of the log decay, fp32."""
    g = g.contiguous()
    batch_size, num_tokens, H, K = g.shape
    num_chunks = tilelang.cdiv(num_tokens, chunk_size)
    g_cs = torch.empty(
        (batch_size, num_tokens, H, K), dtype=torch.float32, device=g.device
    )
    kernel = tilelang_gcs_2(
        H, K, chunk_size, accum_dtype="float32", g_in_dtype=g.dtype
    )
    kernel(g, g_cs, num_chunks)
    return g_cs
