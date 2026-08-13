# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Segment-parallel march for SM120: the sequence-split intended to
remove the march's serial-chain bottleneck.

PARKED, NOT WIRED: numerically sound, but ~10x slower per chunk step
than the single-chain fused_march on SM120 — the per-step cost, not
the chain length, dominates and the cause is unresolved (hoisting the
range check out of the pipelined loop did not recover it). Kept for a
future attempt; compare its generated CUDA against fused_march's.

The chunk recurrence is AFFINE in the state, so a sequence split needs no
bespoke correction machinery — the same kernel runs in three modes:

  zero: every segment marches in parallel from state 0 (segment 0 from
        h0), writing outputs and its zero-state final Q_p.
  homo: writes disabled (mv = 0), initial state = identity COLUMNS —
        the final state of a column-slice IS that slice of the segment
        transition matrix P_p. Outputs skipped entirely.
  corr: initial state = the stitched true segment-start X_p, mv = 0,
        outputs ACCUMULATED into o (the inherited-state contribution).

A tiny torch stitch between passes propagates true starts:
X_p = P_{p-1} @ X_{p-1} + Q_{p-1}; the final state comes from the stitch.
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
def tilelang_seg_march_2(
    H,
    DK,
    DV,
    chunk_size,
    scale,
    accum_dtype,
    qkva_dtype,
    gend_dtype,
    st_dtype,
    o_dtype,
    mode,          # "zero" | "homo" | "corr"
    use_h0,        # zero mode: segment 0 starts from src_state slot 0
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
    st_shape = (batch_size, T.dynamic("num_slots"), H, DK, DV)

    @T.prim_func
    def tilelang_seg_march_2_kernel(
        ekb: T.Tensor(x_shape, dtype=qkva_dtype),
        kte: T.Tensor(x_shape, dtype=qkva_dtype),
        eq: T.Tensor(x_shape, dtype=qkva_dtype),
        v: T.Tensor(v_shape, dtype=qkva_dtype),
        w: T.Tensor(v_shape, dtype=qkva_dtype),
        gend: T.Tensor(gend_shape, dtype=gend_dtype),
        a: T.Tensor(a_shape, dtype=qkva_dtype),
        attn: T.Tensor(a_shape, dtype=qkva_dtype),
        src_state: T.Tensor(st_shape, dtype=st_dtype),
        o: T.Tensor(v_shape, dtype=o_dtype),
        seg_out: T.Tensor(st_shape, dtype=st_dtype),
        num_chunks: T.int32,
        seg_chunks: T.int32,
        num_segments: T.int32,
        seg_base: T.int32,
    ):
        with T.Kernel(
            batch_size * H * v_split * num_segments, threads=256
        ) as (bidx,):
            per_b = H * v_split * num_segments
            bb = bidx // per_b
            rem = bidx % per_b
            bh = rem // (v_split * num_segments)
            rem2 = rem % (v_split * num_segments)
            vo = (rem2 // num_segments) * DVS
            seg = seg_base + rem2 % num_segments
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

            my_chunks = T.alloc_var("int32")
            my_chunks = num_chunks - seg * seg_chunks
            if my_chunks > seg_chunks:
                my_chunks = seg_chunks

            if tx < 128:
                if mode == "zero":
                    if use_h0:
                        if seg == 0:
                            T.copy(
                                src_state[bb, 0, bh, 0:DK, vo:vo + DVS],
                                s_fragment,
                            )
                        else:
                            T.clear(s_fragment)
                    else:
                        T.clear(s_fragment)
                elif mode == "homo":
                    for j_k, j_v in T.Parallel(DK, DVS):
                        if j_k == vo + j_v:
                            s_fragment[j_k, j_v] = 1.0
                        else:
                            s_fragment[j_k, j_v] = 0.0
                else:  # corr
                    T.copy(
                        src_state[bb, seg, bh, 0:DK, vo:vo + DVS], s_fragment
                    )

                for i_s in T.serial(my_chunks):
                    st = i_s % NS
                    T.barrier_wait(data_is_ready[st], (i_s // NS + 0) % 2)

                    left = (seg * seg_chunks + i_s) * block_S
                    T.copy(s_fragment, s_shared)
                    T.barrier_arrive(bar_s)
                    T.barrier_wait(bar_s, i_s % 2)

                    T.gemm(
                        ekb_shared[st, :, :], s_shared, u_fragment,
                        clear_accum=True,
                    )
                    if mode == "zero":
                        for j_s, j_v in T.Parallel(block_S, DVS):
                            u_fragment[j_s, j_v] = (
                                mv_shared[st, j_s, j_v].astype(accum_dtype)
                                - u_fragment[j_s, j_v]
                            )
                    else:
                        for j_s, j_v in T.Parallel(block_S, DVS):
                            u_fragment[j_s, j_v] = -u_fragment[j_s, j_v]
                    T.copy(u_fragment, ymu_shared)
                    T.barrier_arrive(bar_ymu)
                    T.barrier_wait(bar_ymu, i_s % 2)

                    T.gemm(
                        a_shared[st, :, :], ymu_shared, r_fragment,
                        clear_accum=True,
                    )
                    T.copy(r_fragment, r_shared)
                    T.barrier_arrive(bar_r)
                    T.barrier_wait(bar_r, i_s % 2)

                    if mode != "homo":
                        T.clear(o_fragment)
                        T.gemm(
                            attn_shared[st, :, :], r_shared, o_fragment,
                            clear_accum=False,
                        )
                        T.gemm(
                            eq_shared[st, :, :], s_shared, o_fragment,
                            clear_accum=False,
                        )
                        if mode == "zero":
                            for j_s, j_v in T.Parallel(block_S, DVS):
                                if left + j_s < num_tokens:
                                    o[bb, left + j_s, bh, vo + j_v] = (
                                        o_fragment[j_s, j_v] * scale
                                    ).astype(o_dtype)
                        else:
                            for j_s, j_v in T.Parallel(block_S, DVS):
                                if left + j_s < num_tokens:
                                    u_fragment[j_s, j_v] = o[
                                        bb, left + j_s, bh, vo + j_v
                                    ].astype(accum_dtype)
                                else:
                                    u_fragment[j_s, j_v] = 0.0
                            for j_s, j_v in T.Parallel(block_S, DVS):
                                if left + j_s < num_tokens:
                                    o[bb, left + j_s, bh, vo + j_v] = (
                                        u_fragment[j_s, j_v]
                                        + o_fragment[j_s, j_v] * scale
                                    ).astype(o_dtype)

                    for j_k, j_v in T.Parallel(DK, DVS):
                        s_fragment[j_k, j_v] *= glast_shared[st, j_k]
                    T.gemm(
                        kte_shared[st, :, :], r_shared, s_fragment,
                        transpose_A=True, clear_accum=False,
                    )

                    T.barrier_arrive(data_is_free[st])

                if mode != "corr":
                    T.copy(
                        s_fragment, seg_out[bb, seg, bh, 0:DK, vo:vo + DVS]
                    )

            else:
                for i_s in T.serial(my_chunks):
                    st = i_s % NS
                    T.barrier_wait(data_is_free[st], (i_s // NS + 1) % 2)

                    left = (seg * seg_chunks + i_s) * block_S
                    for j_k in T.Parallel(DK):
                        glast_shared[st, j_k] = gend[
                            bb, seg * seg_chunks + i_s, bh, j_k
                        ]
                    for j_s, j_k in T.Parallel(block_S, DK):
                        if left + j_s < num_tokens:
                            ekb_shared[st, j_s, j_k] = ekb[bb, left + j_s, bh, j_k]
                            kte_shared[st, j_s, j_k] = kte[bb, left + j_s, bh, j_k]
                        else:
                            ekb_shared[st, j_s, j_k] = 0
                            kte_shared[st, j_s, j_k] = 0
                    if mode != "homo":
                        for j_s, j_k in T.Parallel(block_S, DK):
                            if left + j_s < num_tokens:
                                eq_shared[st, j_s, j_k] = eq[bb, left + j_s, bh, j_k]
                            else:
                                eq_shared[st, j_s, j_k] = 0
                        for j_s, j_t in T.Parallel(block_S, block_S):
                            if left + j_s < num_tokens:
                                attn_shared[st, j_s, j_t] = attn[bb, left + j_s, bh, j_t]
                            else:
                                attn_shared[st, j_s, j_t] = 0
                    if mode == "zero":
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
                        else:
                            a_shared[st, j_s, j_t] = 0

                    T.barrier_arrive(data_is_ready[st])

    return tilelang_seg_march_2_kernel


def _run(mode, ekb, kte, eq, v, w, gend, a, attn, src_state, o, seg_out,
         scale, num_chunks, seg_chunks, num_segments, seg_base, use_h0):
    batch_size, num_tokens, H, K = ekb.shape
    V = v.shape[3]
    kernel = tilelang_seg_march_2(
        H, K, V, a.shape[-1], float(scale),
        accum_dtype="float32",
        qkva_dtype=ekb.dtype,
        gend_dtype=gend.dtype,
        st_dtype=src_state.dtype,
        o_dtype=o.dtype,
        mode=mode,
        use_h0=use_h0,
        v_split=4,
    )
    kernel(ekb, kte, eq, v, w, gend, a, attn, src_state, o, seg_out,
           num_chunks, seg_chunks, num_segments, seg_base)


def seg_march_2(
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
    num_segments: int = 8,
):
    """Three-pass segment-parallel march. Same contract as fused_march_2."""
    ekb, kte, eq, v, w, a, attn = (
        x.contiguous() for x in (ekb, kte, eq, v, w, a, attn)
    )
    gend = gend.contiguous()
    batch_size, num_tokens, H, K = ekb.shape
    V = v.shape[3]
    chunk_size = a.shape[-1]
    if scale is None:
        scale = K**-0.5
    num_chunks = tilelang.cdiv(num_tokens, chunk_size)
    P = max(1, min(num_segments, num_chunks))
    seg_chunks = tilelang.cdiv(num_chunks, P)
    P = tilelang.cdiv(num_chunks, seg_chunks)

    use_h0 = initial_state is not None
    dev = ekb.device
    slots = max(P, 1)
    src0 = torch.zeros(
        (batch_size, slots, H, K, V), dtype=torch.float32, device=dev
    )
    if use_h0:
        src0[:, 0] = initial_state.float()
    o = torch.empty(
        (batch_size, num_tokens, H, V), dtype=ekb.dtype, device=dev
    )
    q_seg = torch.empty_like(src0)
    p_seg = torch.empty_like(src0)

    args = (ekb, kte, eq, v, w, gend, a, attn)
    # zero pass: outputs + zero-state segment finals
    _run("zero", *args, src0, o, q_seg, scale,
         num_chunks, seg_chunks, P, 0, use_h0)
    if P == 1:
        ht = q_seg[:, 0]
        return o, (ht if output_final_state else None)
    # homogeneous pass: segment transition matrices (identity columns in)
    _run("homo", *args, src0, o, p_seg, scale,
         num_chunks, seg_chunks, P, 0, False)
    # stitch: true segment-start states + final state
    s_starts = torch.zeros_like(src0)
    if use_h0:
        s_starts[:, 0] = initial_state.float()
    for p in range(1, P):
        s_starts[:, p] = (
            torch.einsum("bhoi,bhiv->bhov", p_seg[:, p - 1], s_starts[:, p - 1])
            + q_seg[:, p - 1]
        )
    ht = (
        torch.einsum("bhoi,bhiv->bhov", p_seg[:, P - 1], s_starts[:, P - 1])
        + q_seg[:, P - 1]
    )
    # correction pass over segments 1..P-1: add inherited-state terms
    _run("corr", *args, s_starts, o, q_seg, scale,
         num_chunks, seg_chunks, P - 1, 1, False)
    return o, (ht if output_final_state else None)
