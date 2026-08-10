# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""GDN2 fused march+output for SM120 (chunk 32): the v2 core.

One kernel marches the state AND emits outputs, mirroring the gdn
forward's structure (fused_gdr_fwd never materializes per-chunk states).
Per chunk, with the state S resident and operands decay-folded ON THE
FLY from raw tensors (ekb/kte/eq/mv never exist in global memory):

    U  = ekb @ S            R = A @ (w*v - U)
    o  = scale * [ attn @ R + eq @ S ]
    S  = e^{G_end} * S + kte^T @ R

Grid: (batch, head, V-slice) — the march is independent per value-column
slice. Flat 128-thread blocks (v0); producer/consumer pipelining is the
next evolution once numerics are pinned.
"""

from typing import Optional

import torch
import tilelang
import tilelang.language as T

L2E = 1.442695


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_fused_march_2(
    H,
    Hg,
    DK,
    DV,
    chunk_size,
    scale,
    accum_dtype,
    qkva_dtype,
    g_dtype,
    ht_dtype,
    o_dtype,
    use_initial_state,
    store_final_state,
    v_split,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    block_S = chunk_size
    DVS = DV // v_split

    qk_shape = (batch_size, num_tokens, Hg, DK)
    g_shape = (batch_size, num_tokens, H, DK)
    bx_shape = (batch_size, num_tokens, H, DK)
    v_shape = (batch_size, num_tokens, H, DV)
    a_shape = (batch_size, num_tokens, H, chunk_size)
    h0_shape = (batch_size, H, DK, DV)
    ht_shape = (batch_size, H, DK, DV)

    @T.prim_func
    def tilelang_fused_march_2_kernel(
        q: T.Tensor(qk_shape, dtype=qkva_dtype),
        k: T.Tensor(qk_shape, dtype=qkva_dtype),
        v: T.Tensor(v_shape, dtype=qkva_dtype),
        g: T.Tensor(g_shape, dtype=g_dtype),
        b: T.Tensor(bx_shape, dtype=qkva_dtype),
        w: T.Tensor(v_shape, dtype=qkva_dtype),
        a: T.Tensor(a_shape, dtype=qkva_dtype),
        attn: T.Tensor(a_shape, dtype=qkva_dtype),
        h0: T.Tensor(h0_shape, dtype=ht_dtype),
        o: T.Tensor(v_shape, dtype=o_dtype),
        ht: T.Tensor(ht_shape, dtype=ht_dtype),
        num_chunks: T.int32,
    ):
        with T.Kernel(batch_size * H * v_split, threads=128) as (bbhv,):
            bb = bbhv // (H * v_split)
            bh = (bbhv % (H * v_split)) // v_split
            vo = (bbhv % v_split) * DVS
            bhg = bh // (H // Hg)

            ekb_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            kte_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            eq_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            mv_shared = T.alloc_shared((block_S, DVS), dtype=qkva_dtype)
            a_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            attn_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            s_shared = T.alloc_shared((DK, DVS), dtype=qkva_dtype)
            ymu_shared = T.alloc_shared((block_S, DVS), dtype=qkva_dtype)
            r_shared = T.alloc_shared((block_S, DVS), dtype=qkva_dtype)
            glast_shared = T.alloc_shared((DK), dtype=accum_dtype, scope="shared")

            s_fragment = T.alloc_fragment((DK, DVS), dtype=accum_dtype)
            u_fragment = T.alloc_fragment((block_S, DVS), dtype=accum_dtype)
            r_fragment = T.alloc_fragment((block_S, DVS), dtype=accum_dtype)
            o_fragment = T.alloc_fragment((block_S, DVS), dtype=accum_dtype)

            last_idx = T.alloc_var("int32")

            if use_initial_state:
                T.copy(h0[bb, bh, 0:DK, vo:vo + DVS], s_fragment)
            else:
                T.clear(s_fragment)

            for i_s in T.serial(num_chunks):
                left = i_s * block_S
                last_idx = left + block_S - 1
                if last_idx >= num_tokens:
                    last_idx = num_tokens - 1

                # G_end (log) for this chunk, per channel.
                for j_k in T.Parallel(DK):
                    glast_shared[j_k] = g[bb, last_idx, bh, j_k]

                # Fold operands on the fly from raw tensors (one pass).
                for j_s, j_k in T.Parallel(block_S, DK):
                    if left + j_s < num_tokens:
                        ekb_shared[j_s, j_k] = (
                            T.exp2(g[bb, left + j_s, bh, j_k] * L2E)
                            * b[bb, left + j_s, bh, j_k].astype(accum_dtype)
                            * k[bb, left + j_s, bhg, j_k].astype(accum_dtype)
                        ).astype(qkva_dtype)
                        kte_shared[j_s, j_k] = (
                            T.exp2(
                                (glast_shared[j_k] - g[bb, left + j_s, bh, j_k])
                                * L2E
                            )
                            * k[bb, left + j_s, bhg, j_k].astype(accum_dtype)
                        ).astype(qkva_dtype)
                        eq_shared[j_s, j_k] = (
                            T.exp2(g[bb, left + j_s, bh, j_k] * L2E)
                            * q[bb, left + j_s, bhg, j_k].astype(accum_dtype)
                        ).astype(qkva_dtype)
                    else:
                        ekb_shared[j_s, j_k] = 0
                        kte_shared[j_s, j_k] = 0
                        eq_shared[j_s, j_k] = 0
                for j_s, j_v in T.Parallel(block_S, DVS):
                    if left + j_s < num_tokens:
                        mv_shared[j_s, j_v] = (
                            w[bb, left + j_s, bh, vo + j_v].astype(accum_dtype)
                            * v[bb, left + j_s, bh, vo + j_v].astype(accum_dtype)
                        ).astype(qkva_dtype)
                    else:
                        mv_shared[j_s, j_v] = 0
                for j_s, j_t in T.Parallel(block_S, block_S):
                    if left + j_s < num_tokens:
                        a_shared[j_s, j_t] = a[bb, left + j_s, bh, j_t]
                        attn_shared[j_s, j_t] = attn[bb, left + j_s, bh, j_t]
                    else:
                        a_shared[j_s, j_t] = 0
                        attn_shared[j_s, j_t] = 0

                # U = ekb @ S ; R = A @ (mv - U)
                T.copy(s_fragment, s_shared)
                T.gemm(ekb_shared, s_shared, u_fragment, clear_accum=True)
                for j_s, j_v in T.Parallel(block_S, DVS):
                    u_fragment[j_s, j_v] = (
                        mv_shared[j_s, j_v].astype(accum_dtype)
                        - u_fragment[j_s, j_v]
                    )
                T.copy(u_fragment, ymu_shared)
                T.gemm(a_shared, ymu_shared, r_fragment, clear_accum=True)
                T.copy(r_fragment, r_shared)

                # o = scale * (attn @ R + eq @ S)
                T.clear(o_fragment)
                T.gemm(attn_shared, r_shared, o_fragment, clear_accum=False)
                T.gemm(eq_shared, s_shared, o_fragment, clear_accum=False)
                for j_s, j_v in T.Parallel(block_S, DVS):
                    if left + j_s < num_tokens:
                        o[bb, left + j_s, bh, vo + j_v] = (
                            o_fragment[j_s, j_v] * scale
                        ).astype(o_dtype)

                # S = e^{G_end} * S + kte^T @ R
                for j_k, j_v in T.Parallel(DK, DVS):
                    s_fragment[j_k, j_v] *= T.exp2(glast_shared[j_k] * L2E)
                T.gemm(
                    kte_shared, r_shared, s_fragment,
                    transpose_A=True, clear_accum=False,
                )

            if store_final_state:
                T.copy(s_fragment, ht[bb, bh, 0:DK, vo:vo + DVS])

    return tilelang_fused_march_2_kernel


def fused_march_2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cs: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    attn: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
):
    """March + output in one pass. q/k: [B, T, Hg, K]; g_cs: [B, T, H, K]
    fp32 chunk-local cumsum; b: [B, T, H, K]; v/w: [B, T, H, V]; a/attn:
    [B, T, H, chunk] from kkt_solve. Returns (o [B, T, H, V], ht)."""
    q, k, v, b, w, a, attn = (
        x.contiguous() for x in (q, k, v, b, w, a, attn)
    )
    g_cs = g_cs.contiguous()
    batch_size, num_tokens, Hg, K = q.shape
    H, V = v.shape[2], v.shape[3]
    chunk_size = a.shape[-1]
    assert K == 128 and V == 128
    assert g_cs.dtype == torch.float32
    if scale is None:
        scale = K**-0.5

    use_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = torch.empty(
            (batch_size, H, K, V), dtype=torch.float32, device=q.device
        )
    else:
        initial_state = initial_state.float().contiguous()
    o = torch.empty(
        (batch_size, num_tokens, H, V), dtype=q.dtype, device=q.device
    )
    ht = torch.empty(
        (batch_size, H, K, V), dtype=torch.float32, device=q.device
    )
    kernel = tilelang_fused_march_2(
        H,
        Hg,
        K,
        V,
        chunk_size,
        float(scale),
        accum_dtype="float32",
        qkva_dtype=q.dtype,
        g_dtype=g_cs.dtype,
        ht_dtype=torch.float32,
        o_dtype=o.dtype,
        use_initial_state=use_initial_state,
        store_final_state=output_final_state,
        v_split=4,
    )
    num_chunks = tilelang.cdiv(num_tokens, chunk_size)
    kernel(q, k, v, g_cs, b, w, a, attn, initial_state, o, ht, num_chunks)
    return o, (ht if output_final_state else None)
