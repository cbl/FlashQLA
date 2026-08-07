# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Unit tests for the GDN2 (gated_delta_rule_2) kernels.

Verification ladder (DESIGN_GDR2.md): the two tests that run before any
gdn2 kernel exists are the chunked-math check (rung 2) and the degeneracy
check against the shipped gdn kernels (rung 3). Per-kernel stage tests
unlock milestone by milestone.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ref_gdr2 import (
    chunk_gdn2_bwd_ref,
    chunk_gdn2_fwd_ref,
    gdn2_sequential,
    gdn_gates_as_gdn2,
)

RTOL = 0.02
HEAD_DIM_K = 128
HEAD_DIM_V = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# (B, T, Hk, Hv) — chunk-boundary-straddling T values on purpose.
CONFIGS = [
    pytest.param(1, 256, 4, 4, id="B1-T256-H4"),
    pytest.param(2, 200, 2, 8, id="B2-T200-H8G4"),
    pytest.param(1, 1000, 4, 16, id="B1-T1000-H16G4"),
]


def _make_inputs_2(batch_size, num_tokens, hk, hv,
                   d_k=HEAD_DIM_K, d_v=HEAD_DIM_V, seed=42,
                   dtype=torch.float64, device=DEVICE):
    """GDN2 inputs with trained-model-like gate statistics: per-channel
    decay via -A*softplus (A drawn per channel for channel diversity),
    sigmoid erase/write gates, l2-normalized q/k."""
    torch.manual_seed(seed)
    q = torch.nn.functional.normalize(
        torch.randn(batch_size, num_tokens, hk, d_k, device=device), dim=-1)
    k = torch.nn.functional.normalize(
        torch.randn(batch_size, num_tokens, hk, d_k, device=device), dim=-1)
    v = torch.randn(batch_size, num_tokens, hv, d_v, device=device)
    amp = torch.empty(hv, d_k, device=device).uniform_(1.0, 16.0)
    g = -amp * torch.nn.functional.softplus(
        torch.randn(batch_size, num_tokens, hv, d_k, device=device) * 0.5 + 1.0)
    b = torch.rand(batch_size, num_tokens, hv, d_k, device=device).sigmoid()
    w = torch.rand(batch_size, num_tokens, hv, d_v, device=device).sigmoid()
    h0 = torch.randn(batch_size, hv, d_k, d_v, device=device)
    return tuple(x.to(dtype) for x in (q, k, v, g, b, w, h0))


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


# ---------------------------------------------------------------------------
# Rung 2: the chunked decomposition IS the recurrence (fp64, no kernels)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("B,T,Hk,Hv", CONFIGS)
@pytest.mark.parametrize("use_h0", [False, True])
def test_chunked_ref_matches_sequential(B, T, Hk, Hv, use_h0):
    q, k, v, g, b, w, h0 = _make_inputs_2(B, T, Hk, Hv)
    h0 = h0 if use_h0 else None
    o_seq, s_seq = gdn2_sequential(q, k, v, g, b, w, initial_state=h0)
    o_chk, s_chk = chunk_gdn2_fwd_ref(q, k, v, g, b, w, initial_state=h0)
    assert _rel(o_chk, o_seq) < 1e-10
    assert _rel(s_chk, s_seq) < 1e-10


def test_bwd_ref_shapes():
    q, k, v, g, b, w, h0 = _make_inputs_2(1, 128, 2, 4)
    o, s = gdn2_sequential(q, k, v, g, b, w, initial_state=h0)
    grads = chunk_gdn2_bwd_ref(
        q, k, v, g, b, w, torch.randn_like(o),
        dht=torch.randn_like(s), initial_state=h0,
    )
    dq, dk, dv, dg, db, dw, dh0 = grads
    assert dq.shape == q.shape and dk.shape == k.shape
    assert dv.shape == v.shape and dg.shape == g.shape
    assert db.shape == b.shape and dw.shape == w.shape
    assert dh0.shape == h0.shape


@pytest.mark.skipif(DEVICE != "cuda", reason="tilelang kernels are CUDA-only")
@pytest.mark.parametrize("B,T,Hk,Hv", CONFIGS)
def test_prepare_inputs_fused_matches_torch(B, T, Hk, Hv):
    try:
        from flash_qla.ops.gated_delta_rule_2.chunk import CHUNK_SIZE_2
        from flash_qla.ops.gated_delta_rule_2.chunk.prepare_inputs import (
            prepare_inputs_2,
        )
        from flash_qla.ops.gated_delta_rule_2.chunk.prepare_inputs_tl import (
            prepare_inputs_2_fused,
        )
    except Exception as e:
        pytest.skip(f"flash_qla unavailable here ({type(e).__name__})")
    if CHUNK_SIZE_2 is None:
        pytest.skip("unsupported arch")

    q, k, v, g, b, w, h0 = _make_inputs_2(B, T, Hk, Hv)
    args = (q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16),
            g.float(), b.to(torch.bfloat16), w.to(torch.bfloat16), CHUNK_SIZE_2)
    ref = prepare_inputs_2(*args)
    fused = prepare_inputs_2_fused(*args)
    for name, a, c in zip(("g_cs", "eq", "ekb", "kte", "mv", "gend"), fused, ref):
        assert _rel(a, c.double()) < RTOL, f"{name} mismatch"


# ---------------------------------------------------------------------------
# Rung 3: degeneracy — per-head gates reduce GDN2 to GDN, checked against
# the shipped gdn kernels (runs before any gdn2 kernel exists)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(DEVICE != "cuda", reason="gdn kernels are CUDA-only")
@pytest.mark.parametrize("B,T,Hk,Hv", CONFIGS)
def test_degeneracy_vs_gdn_kernel(B, T, Hk, Hv):
    from flash_qla import chunk_gated_delta_rule

    q, k, v, _, _, _, h0 = _make_inputs_2(B, T, Hk, Hv)
    torch.manual_seed(7)
    g_head = -torch.nn.functional.softplus(
        torch.randn(B, T, Hv, device=DEVICE) * 0.5 + 1.0) * 8
    beta_head = torch.rand(B, T, Hv, device=DEVICE).sigmoid()

    g2, b2, w2 = gdn_gates_as_gdn2(g_head, beta_head, HEAD_DIM_K, HEAD_DIM_V)
    o_ref, s_ref = gdn2_sequential(
        q, k, v, g2.double(), b2.double(), w2.double(), initial_state=h0)

    o_gdn, s_gdn = chunk_gated_delta_rule(
        q=q.to(torch.bfloat16), k=k.to(torch.bfloat16), v=v.to(torch.bfloat16),
        g=g_head.float(), beta=beta_head.float(),
        initial_state=h0.float(), output_final_state=True,
    )
    assert _rel(o_gdn, o_ref) < RTOL
    assert _rel(s_gdn, s_ref) < RTOL


# ---------------------------------------------------------------------------
# Rung 4: per-kernel stage tests — unlocked milestone by milestone
# ---------------------------------------------------------------------------

def _chunk_cumsum(g: torch.Tensor, chunk_size: int = 64) -> torch.Tensor:
    """Chunk-local inclusive cumsum over T, shape-preserving."""
    from ref_gdr2 import _pad_chunks
    bsz, t = g.shape[:2]
    gc = _pad_chunks(g, chunk_size).cumsum(dim=2)
    return gc.reshape(bsz, -1, *g.shape[2:])[:, :t]


@pytest.mark.parametrize("B,T,Hk,Hv", CONFIGS)
def test_kkt_solve_2_matches_ref(B, T, Hk, Hv):
    try:
        from flash_qla.ops.gated_delta_rule_2.chunk import CHUNK_SIZE_2, kkt_solve
    except Exception as e:  # no CUDA / no tilelang on this machine
        pytest.skip(f"flash_qla unavailable here ({type(e).__name__})")
    if kkt_solve is None:
        pytest.skip("gdn2 kkt_solve: unsupported arch (SM90/SM120 only)")

    from ref_gdr2 import _expand_heads, _pad_chunks, ref_kkt_2, ref_solve

    q, k, v, g, b, w, h0 = _make_inputs_2(B, T, Hk, Hv)
    chunk = CHUNK_SIZE_2
    k_bf = k.to(torch.bfloat16)
    b_bf = b.to(torch.bfloat16)
    g_cs = _chunk_cumsum(g.float(), chunk)                     # fp32, kernel input

    a_kernel, attn_kernel = kkt_solve(k_bf, g_cs, b_bf, q.to(torch.bfloat16),
                                      chunk_size=chunk)

    # fp64 reference from the SAME quantized inputs.
    kc = _pad_chunks(_expand_heads(k_bf.double(), Hv), chunk)
    bc = _pad_chunks(b_bf.double(), chunk)
    gc = _pad_chunks(g.float().double(), chunk).cumsum(dim=2)
    t_ref = ref_solve(ref_kkt_2(kc, gc, bc))                   # [B, N, Hv, C, C]
    t_ref = t_ref.permute(0, 1, 3, 2, 4).reshape(B, -1, Hv, chunk)[:, :T]

    assert _rel(a_kernel[:, :T], t_ref) < RTOL

    # the merged attention output: inclusive-tril decayed q-gram
    from ref_gdr2 import _causal_decay
    qc = _pad_chunks(_expand_heads(q.to(torch.bfloat16).double(), Hv), chunk)
    diff = _causal_decay(gc)
    attn_ref = torch.einsum("bnihk,bnijhk,bnjhk->bnhij", qc, diff, kc)
    attn_ref = attn_ref.permute(0, 1, 3, 2, 4).reshape(B, -1, Hv, chunk)[:, :T]
    assert _rel(attn_kernel[:, :T], attn_ref) < RTOL


@pytest.mark.parametrize("B,T,Hk,Hv", CONFIGS)
@pytest.mark.parametrize("use_h0", [False, True])
def test_prepare_h_2_matches_ref(B, T, Hk, Hv, use_h0):
    try:
        from flash_qla.ops.gated_delta_rule_2.chunk import (
            CHUNK_SIZE_2, kkt_solve, prepare_h_2, prepare_inputs_2,
        )
    except Exception as e:
        pytest.skip(f"flash_qla unavailable here ({type(e).__name__})")
    if prepare_h_2 is None:
        pytest.skip("gdn2 prepare_h_2: not built for this arch yet")

    from ref_gdr2 import chunk_gdn2_fwd_ref

    q, k, v, g, b, w, h0 = _make_inputs_2(B, T, Hk, Hv)
    chunk = CHUNK_SIZE_2
    k_bf, v_bf = k.to(torch.bfloat16), v.to(torch.bfloat16)
    b_bf, w_bf = b.to(torch.bfloat16), w.to(torch.bfloat16)
    g32 = g.float()
    h0_arg = h0.float() if use_h0 else None

    g_cs, _, ekb, kte, mv, gend = prepare_inputs_2(
        q.to(torch.bfloat16), k_bf, v_bf, g32, b_bf, w_bf, chunk)
    a, _ = kkt_solve(k_bf, g_cs, b_bf, q.to(torch.bfloat16), chunk_size=chunk)
    h_kernel, ht_kernel, r_kernel = prepare_h_2(ekb, kte, mv, a, gend,
                                                initial_state=h0_arg)

    # fp64 reference from the SAME quantized inputs.
    _, s_ref, hs_ref = chunk_gdn2_fwd_ref(
        q, k_bf.double(), v_bf.double(), g32.double(),
        b_bf.double(), w_bf.double(),
        initial_state=h0.double() if use_h0 else None,
        chunk_size=chunk, output_h=True,
    )
    assert _rel(ht_kernel, s_ref) < RTOL
    assert _rel(h_kernel, hs_ref.to(h_kernel.dtype).double()) < RTOL

    # r feeds fused_fwd and has no other test: emulate R = A @ (mv - ekb@S)
    # in fp32 from the kernel's own (already validated) inputs and states.
    from ref_gdr2 import _pad_chunks
    a_c = _pad_chunks(a.float(), chunk)          # [B, N, C, H, C]
    ekb_c = _pad_chunks(ekb.float(), chunk)
    mv_c = _pad_chunks(mv.float(), chunk)
    r_c = _pad_chunks(r_kernel.float(), chunk)
    for i in range(min(2, h_kernel.shape[1])):
        u = torch.einsum("bchk,bhkv->bchv", ekb_c[:, i], h_kernel[:, i].float())
        r_ref = torch.einsum("bihj,bjhv->bihv", a_c[:, i], mv_c[:, i] - u)
        assert _rel(r_c[:, i], r_ref.double()) < RTOL, f"r mismatch chunk {i}"


@pytest.mark.parametrize("B,T,Hk,Hv", CONFIGS)
@pytest.mark.parametrize("use_h0", [False, True])
def test_chunk_gdn2_fwd_matches_ref(B, T, Hk, Hv, use_h0):
    try:
        from flash_qla import chunk_gdn2
        from flash_qla.ops.gated_delta_rule_2.chunk import CHUNK_SIZE_2, prepare_h_2
    except Exception as e:
        pytest.skip(f"flash_qla unavailable here ({type(e).__name__})")
    if prepare_h_2 is None:
        pytest.skip("gdn2 forward: not built for this arch yet")

    from ref_gdr2 import chunk_gdn2_fwd_ref

    q, k, v, g, b, w, h0 = _make_inputs_2(B, T, Hk, Hv)
    q_bf, k_bf = q.to(torch.bfloat16), k.to(torch.bfloat16)
    v_bf = v.to(torch.bfloat16)
    b_bf, w_bf = b.to(torch.bfloat16), w.to(torch.bfloat16)
    g32 = g.float()

    o_k, ht_k = chunk_gdn2(
        q_bf, k_bf, v_bf, g32, b_bf, w_bf,
        initial_state=h0.float() if use_h0 else None,
        output_final_state=True,
    )
    o_ref, s_ref = chunk_gdn2_fwd_ref(
        q_bf.double(), k_bf.double(), v_bf.double(), g32.double(),
        b_bf.double(), w_bf.double(),
        initial_state=h0.double() if use_h0 else None,
        chunk_size=CHUNK_SIZE_2,
    )
    assert _rel(o_k, o_ref) < RTOL
    assert _rel(ht_k, s_ref) < RTOL


@pytest.mark.parametrize("B,T,Hk,Hv", CONFIGS)
def test_chunk_gdn2_fwd_vs_fla(B, T, Hk, Hv):
    try:
        from flash_qla import chunk_gdn2
        from flash_qla.ops.gated_delta_rule_2.chunk import prepare_h_2
    except Exception as e:
        pytest.skip(f"flash_qla unavailable here ({type(e).__name__})")
    if prepare_h_2 is None:
        pytest.skip("gdn2 forward: not built for this arch yet")
    try:
        from fla.ops.gdn2 import chunk_gdn2 as fla_gdn2
    except Exception as e:
        pytest.skip(f"fla gdn2 unavailable ({type(e).__name__})")

    q, k, v, g, b, w, h0 = _make_inputs_2(B, T, Hk, Hv)
    q_bf, k_bf = q.to(torch.bfloat16), k.to(torch.bfloat16)
    v_bf = v.to(torch.bfloat16)
    b_bf, w_bf = b.to(torch.bfloat16), w.to(torch.bfloat16)
    g32 = g.float()

    o_ours, ht_ours = chunk_gdn2(
        q_bf, k_bf, v_bf, g32, b_bf, w_bf,
        initial_state=h0.float(), output_final_state=True,
    )
    grp = Hv // Hk
    try:
        o_fla, ht_fla = fla_gdn2(
            q=q_bf.repeat_interleave(grp, dim=2),
            k=k_bf.repeat_interleave(grp, dim=2),
            v=v_bf, g=g32, b=b_bf, w=w_bf,
            initial_state=h0.float(), output_final_state=True,
        )
    except Exception as e:
        pytest.skip(f"fla gdn2 call failed here ({type(e).__name__}: {e})")
    # Two bf16 kernels with different chunk algebra: allow 2x the ref rtol.
    assert _rel(o_ours, o_fla) < 2 * RTOL
    assert _rel(ht_ours, ht_fla) < 2 * RTOL


@pytest.mark.skip(reason="M4: fused_bwd_2 not yet implemented")
def test_chunk_gdn2_bwd_vs_ref():
    pass
