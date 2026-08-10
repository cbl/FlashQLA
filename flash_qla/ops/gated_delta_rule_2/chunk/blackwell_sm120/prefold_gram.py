# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Fused prefold + gram kernel for SM120 (chunk 32).

One kernel, one tile load: raw q/k/g/b come in once; L2 normalization,
the chunk-local decay cumsum, the folded operands (eq/ekb/kte/gend) AND
the full gram stage (A-inverse + attention matrix, sub-blocked exactly
as in kkt_solve) all happen on the same resident shared-memory tiles.
Replaces the prepare_inputs_2b + kkt_solve pair: the gram stage's tile
reloads and one kernel launch disappear, and normalized q/k never touch
global memory at all.
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
def tilelang_prefold_gram_2(
    H,
    Hg,
    DK,
    chunk_size,
    accum_dtype,
    qkva_dtype,
    g_in_dtype,
    do_l2norm,
    skip_off16=False,
    skip_off8=False,
    skip_diag8=False,
    skip_inv=False,
):
    data_batch_size = T.dynamic("data_batch_size")
    num_tokens = T.dynamic("num_tokens")
    block_S = chunk_size
    SUB = 16

    qk_shape = (data_batch_size, num_tokens, Hg, DK)
    g_in_shape = (data_batch_size, num_tokens, H, DK)
    b_shape = (data_batch_size, num_tokens, H, DK)
    x_shape = (data_batch_size, num_tokens, H, DK)
    a_shape = (data_batch_size, num_tokens, H, chunk_size)
    gend_shape = (data_batch_size, T.dynamic("gend_chunks"), H, DK)

    @T.prim_func
    def tilelang_prefold_gram_2_kernel(
        q: T.Tensor(qk_shape, dtype=qkva_dtype),
        k: T.Tensor(qk_shape, dtype=qkva_dtype),
        g: T.Tensor(g_in_shape, dtype=g_in_dtype),
        b: T.Tensor(b_shape, dtype=qkva_dtype),
        eq: T.Tensor(x_shape, dtype=qkva_dtype),
        ekb: T.Tensor(x_shape, dtype=qkva_dtype),
        kte: T.Tensor(x_shape, dtype=qkva_dtype),
        gend: T.Tensor(gend_shape, dtype=accum_dtype),
        a: T.Tensor(a_shape, dtype=qkva_dtype),
        attn: T.Tensor(a_shape, dtype=qkva_dtype),
        num_chunks: T.int32,
    ):
        with T.Kernel(num_chunks * H, threads=128) as (bch,):
            bc, bh = bch // H, bch % H
            bhg = bh // (H // Hg)

            bb = bc % data_batch_size
            chunk_idx = bc // data_batch_size
            seq_end_idx = num_tokens
            left = chunk_idx * block_S
            right = left + block_S

            norms_shared = T.alloc_shared(
                (2 * block_S), dtype=accum_dtype, scope="shared"
            )
            nrm_local = T.alloc_local((1), dtype=accum_dtype)

            k_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            b_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            g_shared = T.alloc_shared((block_S, DK), dtype=accum_dtype)
            q_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            kl_shared = T.alloc_shared((SUB, DK), dtype=qkva_dtype)
            ql_shared = T.alloc_shared((SUB, DK), dtype=qkva_dtype)
            attnoff_fragment = T.alloc_fragment((SUB, block_S), dtype=accum_dtype)
            attnoff_shared = T.alloc_shared((SUB, block_S), dtype=qkva_dtype)
            attn_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            klq8_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            kr8_shared = T.alloc_shared((SUB, DK), dtype=qkva_dtype)
            off8_fragment = T.alloc_fragment((block_S, SUB), dtype=accum_dtype)
            off8_shared = T.alloc_shared((block_S, SUB), dtype=qkva_dtype)
            diag4a_shared = T.alloc_shared((8, 4, 4), dtype=accum_dtype)
            diag4q_shared = T.alloc_shared((8, 4, 4), dtype=accum_dtype)
            off4a_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            off4q_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            kr_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            arow_fragment = T.alloc_fragment((SUB, block_S), dtype=accum_dtype)
            arow_shared = T.alloc_shared((SUB, block_S), dtype=qkva_dtype)
            a32_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            diag_local = T.alloc_local((1), dtype=accum_dtype)
            diagq_local = T.alloc_local((1), dtype=accum_dtype)
            e_local = T.alloc_local((1), dtype=accum_dtype)
            cum_local = T.alloc_local((1), dtype=accum_dtype)

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


            # ---- prefold: coalesced loads, then smem-only scans ----
            for j_s, j_k in T.Parallel(block_S, DK):
                if left + j_s < seq_end_idx:
                    q_shared[j_s, j_k] = q[bb, left + j_s, bhg, j_k]
                    k_shared[j_s, j_k] = k[bb, left + j_s, bhg, j_k]
                    b_shared[j_s, j_k] = b[bb, left + j_s, bh, j_k]
                    g_shared[j_s, j_k] = g[bb, left + j_s, bh, j_k].astype(
                        accum_dtype
                    )
                else:
                    q_shared[j_s, j_k] = 0
                    k_shared[j_s, j_k] = 0
                    b_shared[j_s, j_k] = 0
                    g_shared[j_s, j_k] = 0.0

            if do_l2norm:
                for rr in T.Parallel(2 * block_S):
                    nrm_local[0] = 0.0
                    for j_k in T.serial(DK):
                        if rr < block_S:
                            nrm_local[0] += (
                                q_shared[rr, j_k].astype(accum_dtype)
                                * q_shared[rr, j_k].astype(accum_dtype)
                            )
                        else:
                            nrm_local[0] += (
                                k_shared[rr - block_S, j_k].astype(accum_dtype)
                                * k_shared[rr - block_S, j_k].astype(accum_dtype)
                            )
                    norms_shared[rr] = T.rsqrt(nrm_local[0] + 1e-6)
                for j_s, j_k in T.Parallel(block_S, DK):
                    q_shared[j_s, j_k] = (
                        q_shared[j_s, j_k].astype(accum_dtype)
                        * norms_shared[j_s]
                    ).astype(qkva_dtype)
                    k_shared[j_s, j_k] = (
                        k_shared[j_s, j_k].astype(accum_dtype)
                        * norms_shared[block_S + j_s]
                    ).astype(qkva_dtype)

            for j_k in T.Parallel(DK):
                cum_local[0] = 0.0
                for j_s in T.serial(block_S):
                    cum_local[0] += g_shared[j_s, j_k]
                    g_shared[j_s, j_k] = cum_local[0]

            # folded-operand outputs for the march
            for j_s, j_k in T.Parallel(block_S, DK):
                if left + j_s < seq_end_idx:
                    eq[bb, left + j_s, bh, j_k] = (
                        T.exp2(g_shared[j_s, j_k] * L2E)
                        * q_shared[j_s, j_k].astype(accum_dtype)
                    ).astype(qkva_dtype)
                    ekb[bb, left + j_s, bh, j_k] = (
                        T.exp2(g_shared[j_s, j_k] * L2E)
                        * b_shared[j_s, j_k].astype(accum_dtype)
                        * k_shared[j_s, j_k].astype(accum_dtype)
                    ).astype(qkva_dtype)
                    kte[bb, left + j_s, bh, j_k] = (
                        T.exp2(
                            (g_shared[block_S - 1, j_k] - g_shared[j_s, j_k])
                            * L2E
                        )
                        * k_shared[j_s, j_k].astype(accum_dtype)
                    ).astype(qkva_dtype)
            for j_k in T.Parallel(DK):
                gend[bb, chunk_idx, bh, j_k] = T.exp2(
                    g_shared[block_S - 1, j_k] * L2E
                )

            # ---- gram stage (identical to kkt_solve) ----
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

            # 4-token tier: hi-4 x lo-4 quadrants of every 8-block, again
            # via stacked M=32 gemms REUSING the (already consumed) klq8
            # and kr buffers. Rebase: each 8-block's row 4.
            for j_s, j_k in T.Parallel(block_S, DK):
                klq8_shared[j_s, j_k] = (
                    b_shared[
                        ((j_s // 4) // 2) * 16 + ((j_s // 4) % 2) * 8 + 4 + (j_s % 4),
                        j_k,
                    ].astype(accum_dtype)
                    * k_shared[
                        ((j_s // 4) // 2) * 16 + ((j_s // 4) % 2) * 8 + 4 + (j_s % 4),
                        j_k,
                    ].astype(accum_dtype)
                    * T.exp2(
                        (
                            g_shared[((j_s // 4) // 2) * 16 + ((j_s // 4) % 2) * 8 + 4 + (j_s % 4), j_k]
                            - g_shared[((j_s // 4) // 2) * 16 + ((j_s // 4) % 2) * 8 + 4, j_k]
                        )
                        * L2E
                    )
                ).astype(qkva_dtype)
            for j_s, j_k in T.Parallel(block_S, DK):
                kr_shared[j_s, j_k] = (
                    k_shared[
                        ((j_s // 4) // 2) * 16 + ((j_s // 4) % 2) * 8 + (j_s % 4),
                        j_k,
                    ].astype(accum_dtype)
                    * T.exp2(
                        (
                            g_shared[((j_s // 4) // 2) * 16 + ((j_s // 4) % 2) * 8 + 4, j_k]
                            - g_shared[((j_s // 4) // 2) * 16 + ((j_s // 4) % 2) * 8 + (j_s % 4), j_k]
                        )
                        * L2E
                    )
                ).astype(qkva_dtype)
            T.gemm(
                klq8_shared, kr_shared, a32_fragment,
                transpose_B=True, clear_accum=True,
            )
            T.copy(a32_fragment, off4a_shared)
            for j_s, j_k in T.Parallel(block_S, DK):
                klq8_shared[j_s, j_k] = (
                    q_shared[
                        ((j_s // 4) // 2) * 16 + ((j_s // 4) % 2) * 8 + 4 + (j_s % 4),
                        j_k,
                    ].astype(accum_dtype)
                    * T.exp2(
                        (
                            g_shared[((j_s // 4) // 2) * 16 + ((j_s // 4) % 2) * 8 + 4 + (j_s % 4), j_k]
                            - g_shared[((j_s // 4) // 2) * 16 + ((j_s // 4) % 2) * 8 + 4, j_k]
                        )
                        * L2E
                    )
                ).astype(qkva_dtype)
            T.gemm(
                klq8_shared, kr_shared, a32_fragment,
                transpose_B=True, clear_accum=True,
            )
            T.copy(a32_fragment, off4q_shared)

            if not skip_diag8:
                # The eight 4x4 diagonals stay elementwise (80 inclusive
                # pairs, exactly one per thread), one exp2 feeding both grams.
                for b4, j_s, j_t in T.Parallel(8, 4, 4):
                    diag_local[0] = 0.0
                    diagq_local[0] = 0.0
                    if j_s >= j_t:
                        for j_ko in T.serial(8):
                            for j_ki in T.unroll(16):
                                e_local[0] = T.exp2(
                                    (
                                        g_shared[(b4 // 4) * 16 + ((b4 % 4) // 2) * 8
                                                 + (b4 % 2) * 4 + j_s,
                                                 j_ko * 16 + j_ki]
                                        - g_shared[(b4 // 4) * 16 + ((b4 % 4) // 2) * 8
                                                   + (b4 % 2) * 4 + j_t,
                                                   j_ko * 16 + j_ki]
                                    )
                                    * L2E
                                ) * k_shared[(b4 // 4) * 16 + ((b4 % 4) // 2) * 8
                                             + (b4 % 2) * 4 + j_t,
                                             j_ko * 16 + j_ki].astype(accum_dtype)
                                diagq_local[0] += (
                                    q_shared[(b4 // 4) * 16 + ((b4 % 4) // 2) * 8
                                             + (b4 % 2) * 4 + j_s,
                                             j_ko * 16 + j_ki].astype(accum_dtype)
                                    * e_local[0]
                                )
                                diag_local[0] += (
                                    b_shared[(b4 // 4) * 16 + ((b4 % 4) // 2) * 8
                                             + (b4 % 2) * 4 + j_s,
                                             j_ko * 16 + j_ki].astype(accum_dtype)
                                    * k_shared[(b4 // 4) * 16 + ((b4 % 4) // 2) * 8
                                               + (b4 % 2) * 4 + j_s,
                                               j_ko * 16 + j_ki].astype(accum_dtype)
                                    * e_local[0]
                                )
                    if j_s > j_t:
                        diag4a_shared[b4, j_s, j_t] = diag_local[0]
                    else:
                        diag4a_shared[b4, j_s, j_t] = 0.0
                    diag4q_shared[b4, j_s, j_t] = diagq_local[0]

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


    return tilelang_prefold_gram_2_kernel


def prefold_gram_2(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    chunk_size: int = 32,
    do_l2norm: bool = False,
    **ablate,
):
    """Fused prefold + gram: (eq, ekb, kte, gend, a, attn) from raw
    q/k [B, T, Hg, K], g (raw log decay) and b [B, T, H, K]."""
    q, k, g, b = (x.contiguous() for x in (q, k, g, b))
    batch_size, num_tokens, Hg, K = k.shape
    H = g.shape[2]
    assert K == 128
    assert chunk_size == 32
    num_chunks = tilelang.cdiv(num_tokens, chunk_size)

    eq = torch.empty(
        (batch_size, num_tokens, H, K), dtype=q.dtype, device=q.device
    )
    ekb = torch.empty_like(eq)
    kte = torch.empty_like(eq)
    gend = torch.empty(
        (batch_size, num_chunks, H, K), dtype=torch.float32, device=q.device
    )
    a = torch.empty(
        (batch_size, num_tokens, H, chunk_size), dtype=k.dtype, device=k.device
    )
    attn = torch.empty_like(a)

    kernel = tilelang_prefold_gram_2(
        H,
        Hg,
        K,
        chunk_size,
        accum_dtype="float32",
        qkva_dtype=k.dtype,
        g_in_dtype=g.dtype,
        do_l2norm=do_l2norm,
        **ablate,
    )
    kernel(q, k, g, b, eq, ekb, kte, gend, a, attn,
           batch_size * num_chunks)
    return eq, ekb, kte, gend, a, attn
