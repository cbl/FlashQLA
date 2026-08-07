# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""GDN2 prepare_h for SM120 (chunk 32): march the recurrent state across
chunks. Operates on the decay-folded operands from ``prepare_inputs_2``
(all exponentials already applied, all bounded), so per chunk with
entering state S:

    U  = ekb @ S                      # what the erase gates read back
    R  = A @ (mv - U)                 # WY payloads (A = kkt_solve output)
    S' = g_end * S + k_to_end^T @ R   # per-KEY-ROW vector decay + writes

Deliberately UN-pipelined (no warp specialization): correctness and
parity first; the producer/consumer skeleton of the gdn kernels is a
later optimization once the full forward exists. Fixed-length only for
now (no cu_seqlens / cp).
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
def tilelang_prepare_h_2(
    H,
    DK,
    DV,
    chunk_size,
    accum_dtype,
    qkva_dtype,
    gend_dtype,
    h_dtype,
    ht_dtype,
    use_initial_state,
    store_final_state,
    store_h,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    num_h_chunks = T.dynamic("num_h_chunks")
    block_S = chunk_size

    x_shape = (batch_size, num_tokens, H, DK)
    v_shape = (batch_size, num_tokens, H, DV)
    a_shape = (batch_size, num_tokens, H, chunk_size)
    gend_shape = (batch_size, num_chunks, H, DK)
    h_shape = (batch_size, num_h_chunks, H, DK, DV)
    h0_shape = (batch_size, H, DK, DV)
    ht_shape = (batch_size, H, DK, DV)

    @T.prim_func
    def tilelang_prepare_h_2_kernel(
        ekb: T.Tensor(x_shape, dtype=qkva_dtype),
        kte: T.Tensor(x_shape, dtype=qkva_dtype),
        mv: T.Tensor(v_shape, dtype=qkva_dtype),
        a: T.Tensor(a_shape, dtype=qkva_dtype),
        g_end_exp: T.Tensor(gend_shape, dtype=gend_dtype),
        h0: T.Tensor(h0_shape, dtype=ht_dtype),
        h: T.Tensor(h_shape, dtype=h_dtype),
        ht: T.Tensor(ht_shape, dtype=ht_dtype),
    ):
        with T.Kernel(batch_size * H, threads=256) as (bbh,):
            bb, bh = bbh // H, bbh % H

            ekb_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            kte_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            mv_shared = T.alloc_shared((block_S, DV), dtype=qkva_dtype)
            a_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            gend_shared = T.alloc_shared((DK), dtype=accum_dtype, scope="shared")
            h_shared = T.alloc_shared((DK, DV), dtype=qkva_dtype)
            ymu_shared = T.alloc_shared((block_S, DV), dtype=qkva_dtype)
            r_shared = T.alloc_shared((block_S, DV), dtype=qkva_dtype)

            h_fragment = T.alloc_fragment((DK, DV), dtype=accum_dtype)
            u_fragment = T.alloc_fragment((block_S, DV), dtype=accum_dtype)
            r_fragment = T.alloc_fragment((block_S, DV), dtype=accum_dtype)

            if use_initial_state:
                T.copy(h0[bb, bh, 0:DK, 0:DV], h_fragment)
            else:
                T.clear(h_fragment)

            for i_s in T.serial(num_chunks):
                left = i_s * block_S

                # Load tiles; zero-pad the tail so padded rows contribute
                # nothing (a's rows past the end are garbage: zero them).
                for j_s, j_k in T.Parallel(block_S, DK):
                    if left + j_s < num_tokens:
                        ekb_shared[j_s, j_k] = ekb[bb, left + j_s, bh, j_k]
                        kte_shared[j_s, j_k] = kte[bb, left + j_s, bh, j_k]
                    else:
                        ekb_shared[j_s, j_k] = 0
                        kte_shared[j_s, j_k] = 0
                for j_s, j_v in T.Parallel(block_S, DV):
                    if left + j_s < num_tokens:
                        mv_shared[j_s, j_v] = mv[bb, left + j_s, bh, j_v]
                    else:
                        mv_shared[j_s, j_v] = 0
                for j_s, j_t in T.Parallel(block_S, block_S):
                    if left + j_s < num_tokens:
                        a_shared[j_s, j_t] = a[bb, left + j_s, bh, j_t]
                    else:
                        a_shared[j_s, j_t] = 0
                for j_k in T.Parallel(DK):
                    gend_shared[j_k] = g_end_exp[bb, i_s, bh, j_k]

                # The chunk's ENTERING state, for the fused_fwd stage.
                T.copy(h_fragment, h_shared)
                if store_h:
                    T.copy(h_shared, h[bb, i_s, bh, 0:DK, 0:DV])

                # U = ekb @ S
                T.gemm(ekb_shared, h_shared, u_fragment, clear_accum=True)
                # ymu = mv - U
                for j_s, j_v in T.Parallel(block_S, DV):
                    u_fragment[j_s, j_v] = (
                        mv_shared[j_s, j_v].astype(accum_dtype)
                        - u_fragment[j_s, j_v]
                    )
                T.copy(u_fragment, ymu_shared)

                # R = A @ (mv - U)
                T.gemm(a_shared, ymu_shared, r_fragment, clear_accum=True)
                T.copy(r_fragment, r_shared)

                # S = g_end * S  (per KEY row), then S += k_to_end^T @ R
                for j_k, j_v in T.Parallel(DK, DV):
                    h_fragment[j_k, j_v] *= gend_shared[j_k]
                T.gemm(
                    kte_shared, r_shared, h_fragment,
                    transpose_A=True, clear_accum=False,
                )

            if store_final_state:
                T.copy(h_fragment, ht[bb, bh, 0:DK, 0:DV])

    return tilelang_prepare_h_2_kernel


def prepare_h_2(
    ekb: torch.Tensor,
    kte: torch.Tensor,
    mv: torch.Tensor,
    a: torch.Tensor,
    g_end_exp: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    output_h: bool = True,
):
    """State march over 32-token chunks on decay-folded operands.

    ekb/kte: [B, T, H, K]; mv: [B, T, H, V]; a: [B, T, H, chunk] from
    kkt_solve; g_end_exp: [B, N, H, K] fp32. Returns (h, ht): h =
    per-chunk ENTERING states [B, N, H, K, V] in input dtype (or None),
    ht = final state [B, H, K, V] fp32 (or None).
    """
    ekb, kte, mv, a = (x.contiguous() for x in (ekb, kte, mv, a))
    g_end_exp = g_end_exp.contiguous()
    batch_size, num_tokens, H, K = ekb.shape
    V = mv.shape[-1]
    chunk_size = a.shape[-1]
    assert K == 128 and V == 128
    assert chunk_size == 32
    assert g_end_exp.dtype == torch.float32

    num_chunks = tilelang.cdiv(num_tokens, chunk_size)
    use_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = torch.empty(
            (batch_size, H, K, V), dtype=torch.float32, device=ekb.device
        )
    else:
        initial_state = initial_state.float().contiguous()
    h = torch.empty(
        (batch_size, num_chunks if output_h else 0, H, K, V),
        dtype=ekb.dtype, device=ekb.device,
    )
    ht = torch.empty(
        (batch_size, H, K, V), dtype=torch.float32, device=ekb.device
    )

    kernel = tilelang_prepare_h_2(
        H,
        K,
        V,
        chunk_size,
        accum_dtype="float32",
        qkva_dtype=ekb.dtype,
        gend_dtype=g_end_exp.dtype,
        h_dtype=h.dtype,
        ht_dtype=ht.dtype,
        use_initial_state=use_initial_state,
        store_final_state=output_final_state,
        store_h=output_h,
    )
    kernel(ekb, kte, mv, a, g_end_exp, initial_state, h, ht)

    if not output_final_state:
        ht = None
    if not output_h:
        h = None
    return h, ht
