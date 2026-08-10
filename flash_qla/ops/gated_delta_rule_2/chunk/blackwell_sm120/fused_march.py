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
slice. Warp-specialized (v1.5): 128 consumer threads own the resident
state and the five GEMMs; 128 producer threads fold the NEXT chunk's
operands from raw tensors into double-buffered stages, so folding cost
hides under compute. Barrier pattern mirrors the gdn prepare_h kernel.
"""

import os
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
    gend_dtype,
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

    x_shape = (batch_size, num_tokens, H, DK)
    v_shape = (batch_size, num_tokens, H, DV)
    a_shape = (batch_size, num_tokens, H, chunk_size)
    gend_shape = (batch_size, T.dynamic("gend_chunks"), H, DK)
    h0_shape = (batch_size, H, DK, DV)
    ht_shape = (batch_size, H, DK, DV)

    @T.prim_func
    def tilelang_fused_march_2_kernel(
        ekb: T.Tensor(x_shape, dtype=qkva_dtype),
        kte: T.Tensor(x_shape, dtype=qkva_dtype),
        eq: T.Tensor(x_shape, dtype=qkva_dtype),
        v: T.Tensor(v_shape, dtype=qkva_dtype),
        w: T.Tensor(v_shape, dtype=qkva_dtype),
        gend: T.Tensor(gend_shape, dtype=gend_dtype),
        a: T.Tensor(a_shape, dtype=qkva_dtype),
        attn: T.Tensor(a_shape, dtype=qkva_dtype),
        h0: T.Tensor(h0_shape, dtype=ht_dtype),
        o: T.Tensor(v_shape, dtype=o_dtype),
        ht: T.Tensor(ht_shape, dtype=ht_dtype),
        num_chunks: T.int32,
    ):
        with T.Kernel(batch_size * H * v_split, threads=256) as (bbhv,):
            bb = bbhv // (H * v_split)
            bh = (bbhv % (H * v_split)) // v_split
            vo = (bbhv % v_split) * DVS
            NS = 2

            ekb_shared = T.alloc_shared((NS, block_S, DK), dtype=qkva_dtype)
            kte_shared = T.alloc_shared((NS, block_S, DK), dtype=qkva_dtype)
            eq_shared = T.alloc_shared((NS, block_S, DK), dtype=qkva_dtype)
            mv_shared = T.alloc_shared((NS, block_S, DVS), dtype=qkva_dtype)
            a_shared = T.alloc_shared((NS, block_S, block_S), dtype=qkva_dtype)
            attn_shared = T.alloc_shared((NS, block_S, block_S), dtype=qkva_dtype)
            glast_shared = T.alloc_shared((NS, DK), dtype=accum_dtype, scope="shared")
            s_shared = T.alloc_shared((DK, DVS), dtype=qkva_dtype)
            ymu_shared = T.alloc_shared((block_S, DVS), dtype=qkva_dtype)
            r_shared = T.alloc_shared((block_S, DVS), dtype=qkva_dtype)

            s_fragment = T.alloc_fragment((DK, DVS), dtype=accum_dtype)
            u_fragment = T.alloc_fragment((block_S, DVS), dtype=accum_dtype)
            r_fragment = T.alloc_fragment((block_S, DVS), dtype=accum_dtype)
            o_fragment = T.alloc_fragment((block_S, DVS), dtype=accum_dtype)

            data_is_ready = T.alloc_barrier(arrive_count=[128] * NS)
            data_is_free = T.alloc_barrier(arrive_count=[128] * NS)
            bar_s = T.alloc_barrier(arrive_count=128)
            bar_ymu = T.alloc_barrier(arrive_count=128)
            bar_r = T.alloc_barrier(arrive_count=128)

            tx = T.get_thread_binding()

            if tx < 128:
                # Consumer: resident state + the five GEMMs per chunk.
                if use_initial_state:
                    T.copy(h0[bb, bh, 0:DK, vo:vo + DVS], s_fragment)
                else:
                    T.clear(s_fragment)

                for i_s in T.serial(num_chunks):
                    st = i_s % NS
                    T.barrier_wait(data_is_ready[st], (i_s // NS + 0) % 2)

                    left = i_s * block_S
                    T.copy(s_fragment, s_shared)
                    T.barrier_arrive(bar_s)
                    T.barrier_wait(bar_s, i_s % 2)

                    # U = ekb @ S ; ymu = mv - U
                    T.gemm(
                        ekb_shared[st, :, :], s_shared, u_fragment,
                        clear_accum=True,
                    )
                    for j_s, j_v in T.Parallel(block_S, DVS):
                        u_fragment[j_s, j_v] = (
                            mv_shared[st, j_s, j_v].astype(accum_dtype)
                            - u_fragment[j_s, j_v]
                        )
                    T.copy(u_fragment, ymu_shared)
                    T.barrier_arrive(bar_ymu)
                    T.barrier_wait(bar_ymu, i_s % 2)

                    # R = A @ ymu
                    T.gemm(
                        a_shared[st, :, :], ymu_shared, r_fragment,
                        clear_accum=True,
                    )
                    T.copy(r_fragment, r_shared)
                    T.barrier_arrive(bar_r)
                    T.barrier_wait(bar_r, i_s % 2)

                    # o = scale * (attn @ R + eq @ S)
                    T.clear(o_fragment)
                    T.gemm(
                        attn_shared[st, :, :], r_shared, o_fragment,
                        clear_accum=False,
                    )
                    T.gemm(
                        eq_shared[st, :, :], s_shared, o_fragment,
                        clear_accum=False,
                    )
                    for j_s, j_v in T.Parallel(block_S, DVS):
                        if left + j_s < num_tokens:
                            o[bb, left + j_s, bh, vo + j_v] = (
                                o_fragment[j_s, j_v] * scale
                            ).astype(o_dtype)

                    # S = e^{G_end} * S + kte^T @ R
                    for j_k, j_v in T.Parallel(DK, DVS):
                        s_fragment[j_k, j_v] *= glast_shared[st, j_k]
                    T.gemm(
                        kte_shared[st, :, :], r_shared, s_fragment,
                        transpose_A=True, clear_accum=False,
                    )

                    T.barrier_arrive(data_is_free[st])

                if store_final_state:
                    T.copy(s_fragment, ht[bb, bh, 0:DK, vo:vo + DVS])

            else:
                # Producer: copy the next chunk's PRE-FOLDED operands
                # into the stage (folding runs fully parallel upstream;
                # serializing it into this march was measured 3x worse).
                for i_s in T.serial(num_chunks):
                    st = i_s % NS
                    T.barrier_wait(data_is_free[st], (i_s // NS + 1) % 2)

                    left = i_s * block_S
                    for j_k in T.Parallel(DK):
                        glast_shared[st, j_k] = gend[bb, i_s, bh, j_k]
                    for j_s, j_k in T.Parallel(block_S, DK):
                        if left + j_s < num_tokens:
                            ekb_shared[st, j_s, j_k] = ekb[bb, left + j_s, bh, j_k]
                            kte_shared[st, j_s, j_k] = kte[bb, left + j_s, bh, j_k]
                            eq_shared[st, j_s, j_k] = eq[bb, left + j_s, bh, j_k]
                        else:
                            ekb_shared[st, j_s, j_k] = 0
                            kte_shared[st, j_s, j_k] = 0
                            eq_shared[st, j_s, j_k] = 0
                    for j_s, j_v in T.Parallel(block_S, DVS):
                        if left + j_s < num_tokens:
                            mv_shared[st, j_s, j_v] = (
                                w[bb, left + j_s, bh, vo + j_v].astype(accum_dtype)
                                * v[bb, left + j_s, bh, vo + j_v].astype(accum_dtype)
                            ).astype(qkva_dtype)
                        else:
                            mv_shared[st, j_s, j_v] = 0
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        if left + j_s < num_tokens:
                            a_shared[st, j_s, j_t] = a[bb, left + j_s, bh, j_t]
                            attn_shared[st, j_s, j_t] = attn[bb, left + j_s, bh, j_t]
                        else:
                            a_shared[st, j_s, j_t] = 0
                            attn_shared[st, j_s, j_t] = 0

                    T.barrier_arrive(data_is_ready[st])

    return tilelang_fused_march_2_kernel


def fused_march_2(
    ekb: torch.Tensor,
    kte: torch.Tensor,
    eq: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    gend: torch.Tensor,
    a: torch.Tensor,
    attn: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
):
    """March + output in one pass on PRE-FOLDED operands (from
    prepare_inputs_2b). ekb/kte/eq: [B, T, H, K]; v/w: [B, T, H, V]
    (w*v folds in the producer); gend:
    [B, N, H, K] fp32 (e^{G_end}); a/attn: [B, T, H, chunk] from
    kkt_solve. Returns (o [B, T, H, V], ht)."""
    ekb, kte, eq, v, w, a, attn = (
        x.contiguous() for x in (ekb, kte, eq, v, w, a, attn)
    )
    gend = gend.contiguous()
    batch_size, num_tokens, H, K = ekb.shape
    V = v.shape[3]
    chunk_size = a.shape[-1]
    assert K == 128 and V == 128
    assert gend.dtype == torch.float32
    if scale is None:
        scale = K**-0.5

    use_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = torch.empty(
            (batch_size, H, K, V), dtype=torch.float32, device=ekb.device
        )
    else:
        initial_state = initial_state.float().contiguous()
    o = torch.empty(
        (batch_size, num_tokens, H, V), dtype=ekb.dtype, device=ekb.device
    )
    ht = torch.empty(
        (batch_size, H, K, V), dtype=torch.float32, device=ekb.device
    )
    kernel = tilelang_fused_march_2(
        H,
        1,
        K,
        V,
        chunk_size,
        float(scale),
        accum_dtype="float32",
        qkva_dtype=ekb.dtype,
        gend_dtype=gend.dtype,
        ht_dtype=torch.float32,
        o_dtype=o.dtype,
        use_initial_state=use_initial_state,
        store_final_state=output_final_state,
        v_split=int(os.environ.get("FLASHQLA_MARCH_VSPLIT", "4")),
    )
    num_chunks = tilelang.cdiv(num_tokens, chunk_size)
    kernel(ekb, kte, eq, v, w, gend, a, attn, initial_state, o, ht, num_chunks)
    return o, (ht if output_final_state else None)
