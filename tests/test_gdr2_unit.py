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

@pytest.mark.skip(reason="M1: kkt_solve_2 not yet implemented")
def test_kkt_solve_2_matches_ref():
    pass


@pytest.mark.skip(reason="M2: prepare_h_2 not yet implemented")
def test_prepare_h_2_matches_ref():
    pass


@pytest.mark.skip(reason="M3: fused_fwd_2 not yet implemented")
def test_fused_fwd_2_matches_ref():
    pass


@pytest.mark.skip(reason="M3: end-to-end fwd parity vs fla chunk_gdn2")
def test_chunk_gdn2_fwd_vs_fla():
    pass


@pytest.mark.skip(reason="M4: fused_bwd_2 not yet implemented")
def test_chunk_gdn2_bwd_vs_ref():
    pass
