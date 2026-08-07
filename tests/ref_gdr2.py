# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Reference implementations for GDN2 (gated_delta_rule_2).

Recurrence (per head; see DESIGN_GDR2.md):

    S_t = (I - k_t (b_t*k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t*v_t)^T
    o_t = scale * S_t^T q_t

    q, k, g, b : [B, T, H, K]   (g, b per KEY channel; g in log space)
    v, w       : [B, T, H, V]

Two references:

- ``gdn2_sequential``: O(T) loop, fp64, differentiable — ground truth.
- ``chunk_gdn2_fwd_ref``: the chunked decomposition the kernels implement,
  with every exponent in bounded pairwise-difference form (G_i - G_j <= 0
  on the used triangle). Matching ``gdn2_sequential`` validates the MATH
  before any tilelang exists; each kernel then matches its stage here.

GVA: q/k may have Hk < Hv heads; references repeat to Hv.
"""

import torch

CHUNK_SIZE = 64


def _expand_heads(x: torch.Tensor, hv: int) -> torch.Tensor:
    hk = x.shape[2]
    return x if hk == hv else x.repeat_interleave(hv // hk, dim=2)


def gdn2_sequential(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float64,
):
    """Exact recurrence. Differentiable; use fp64 inputs with
    ``requires_grad`` for backward references."""
    hv = v.shape[2]
    q, k = _expand_heads(q, hv), _expand_heads(k, hv)
    q, k, v, g, b, w = (x.to(dtype) for x in (q, k, v, g, b, w))
    bsz, t, h, d_k = k.shape
    d_v = v.shape[-1]
    if scale is None:
        scale = d_k**-0.5
    state = (
        initial_state.to(dtype) if initial_state is not None
        else q.new_zeros(bsz, h, d_k, d_v)
    )
    outs = []
    for i in range(t):
        state = g[:, i].exp()[..., None] * state
        erase = torch.einsum("bhk,bhkv->bhv", b[:, i] * k[:, i], state)
        state = state - torch.einsum("bhk,bhv->bhkv", k[:, i], erase)
        state = state + torch.einsum("bhk,bhv->bhkv", k[:, i], w[:, i] * v[:, i])
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, i] * scale, state))
    return torch.stack(outs, dim=1), state


def _pad_chunks(x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """[B, T, ...] -> [B, N, C, ...], zero-padded."""
    bsz, t = x.shape[:2]
    pad = (-t) % chunk_size
    if pad:
        x = torch.cat([x, x.new_zeros(bsz, pad, *x.shape[2:])], dim=1)
    return x.view(bsz, -1, chunk_size, *x.shape[2:])


def _causal_decay(g_cs: torch.Tensor) -> torch.Tensor:
    """Pairwise decay exp(G_i - G_j) [B, N, Ci, Cj, H, K], masked to the
    causal triangle IN THE EXPONENT: the anticausal exponents are positive
    and overflow to inf for fast-decaying channels, and inf * 0 = NaN if
    masked after exponentiation."""
    c = g_cs.shape[2]
    expo = g_cs[:, :, :, None] - g_cs[:, :, None, :]
    mask = torch.ones(c, c, dtype=torch.bool, device=g_cs.device).tril()
    return expo.masked_fill(
        ~mask[None, None, :, :, None, None], float("-inf")
    ).exp()


def ref_kkt_2(k: torch.Tensor, g_cs: torch.Tensor, b: torch.Tensor):
    """Stage 1: the strictly-lower chunk Gram matrix, decay folded in.

    A_ij = sum_c (b*k)_ic exp(G_ic - G_jc) k_jc  for i > j, else 0.
    k, b: [B, N, C, H, K]; g_cs: intra-chunk INCLUSIVE cumsum of g.
    Returns [B, N, H, C, C].
    """
    c = k.shape[2]
    decay = _causal_decay(g_cs)
    a = torch.einsum("bnihk,bnijhk,bnjhk->bnhij", b * k, decay, k)
    tri = torch.ones(c, c, dtype=torch.bool, device=k.device).tril(-1)
    return a * tri


def ref_solve(a: torch.Tensor) -> torch.Tensor:
    """Stage 2: T = (I + A)^{-1} for strictly-lower A [.., C, C]."""
    c = a.shape[-1]
    eye = torch.eye(c, dtype=a.dtype, device=a.device)
    return torch.linalg.solve_triangular(a + eye, eye.expand_as(a), upper=False)


def chunk_gdn2_fwd_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    chunk_size: int = CHUNK_SIZE,
    dtype: torch.dtype = torch.float64,
    output_h: bool = False,
):
    """Chunked forward, the kernel decomposition in plain torch.

    Per chunk with entering state S0 (G = intra-chunk inclusive cumsum g):
      A    = strict-lower [(b*k) e^{G_i-G_j} k]          (ref_kkt_2)
      T    = (I + A)^{-1}                                 (ref_solve)
      R    = T @ (w*v - (e^G * b * k) @ S0)               (the WY payloads)
      o    = scale * [ tril(q e^{G_i-G_j} k^T) @ R  +  (e^G * q) @ S0 ]
      S'   = e^{G_C} * S0  +  (e^{G_C - G} * k)^T @ R
    """
    hv = v.shape[2]
    q, k = _expand_heads(q, hv), _expand_heads(k, hv)
    q, k, v, g, b, w = (x.to(dtype) for x in (q, k, v, g, b, w))
    bsz, t, h, d_k = k.shape
    d_v = v.shape[-1]
    if scale is None:
        scale = d_k**-0.5

    qc, kc, vc, gc, bc, wc = (_pad_chunks(x, chunk_size) for x in (q, k, v, g, b, w))
    n, c = qc.shape[1], chunk_size
    g_cs = gc.cumsum(dim=2)                                   # [B, N, C, H, K]

    # Pairwise per-channel decay, masked in the exponent (see _causal_decay).
    diff = _causal_decay(g_cs)                                # [B,N,Ci,Cj,H,K]
    tril_inc = torch.ones(c, c, dtype=torch.bool, device=k.device).tril()
    a = torch.einsum("bnihk,bnijhk,bnjhk->bnhij", bc * kc, diff, kc)
    a = a * tril_inc.tril(-1)                                 # strict lower
    tmat = ref_solve(a)

    attn = torch.einsum("bnihk,bnijhk,bnjhk->bnhij", qc, diff, kc) * tril_inc

    state = (
        initial_state.to(dtype) if initial_state is not None
        else q.new_zeros(bsz, h, d_k, d_v)
    )
    g_end = g_cs[:, :, -1]                                    # [B, N, H, K]
    ekb = (g_cs.exp() * bc * kc)                              # e^G * b * k
    k_to_end = (g_end[:, :, None] - g_cs).exp() * kc          # e^{G_C - G} * k
    eq = g_cs.exp() * qc                                      # e^G * q
    mv = wc * vc

    outs, hs = [], []
    for i in range(n):
        if output_h:
            hs.append(state)
        p = torch.einsum("bchk,bhkv->bchv", ekb[:, i], state)
        r = torch.einsum("bhij,bjhv->bihv", tmat[:, i], mv[:, i] - p)
        o_intra = torch.einsum("bhij,bjhv->bihv", attn[:, i], r)
        o_inter = torch.einsum("bchk,bhkv->bchv", eq[:, i], state)
        outs.append(scale * (o_intra + o_inter))
        state = g_end[:, i].exp()[..., None] * state + torch.einsum(
            "bchk,bchv->bhkv", k_to_end[:, i], r
        )
    o = torch.cat(outs, dim=1)[:, :t]
    if output_h:
        return o, state, torch.stack(hs, dim=1)
    return o, state


def chunk_gdn2_bwd_ref(
    q, k, v, g, b, w,
    do: torch.Tensor,
    dht: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
):
    """Backward reference via autograd through the fp64 sequential loop.
    Returns (dq, dk, dv, dg, db, dw, dh0); dh0 is None without h0."""
    leaves = [x.detach().to(torch.float64).requires_grad_(True)
              for x in (q, k, v, g, b, w)]
    h0 = None
    if initial_state is not None:
        h0 = initial_state.detach().to(torch.float64).requires_grad_(True)
    o, ht = gdn2_sequential(*leaves, scale=scale, initial_state=h0)
    loss = (o * do.to(torch.float64)).sum()
    if dht is not None:
        loss = loss + (ht * dht.to(torch.float64)).sum()
    grads = torch.autograd.grad(loss, leaves + ([h0] if h0 is not None else []))
    dq, dk, dv, dg, db, dw = grads[:6]
    dh0 = grads[6] if h0 is not None else None
    # GVA needs no manual fold: the leaves are the UNrepeated q/k, and
    # autograd through the reference's repeat_interleave already sums the
    # grouped heads back down.
    return dq, dk, dv, dg, db, dw, dh0


def gdn_gates_as_gdn2(g_head: torch.Tensor, beta_head: torch.Tensor,
                      d_k: int, d_v: int):
    """Degeneracy map: gdn (scalar per-head g, beta) == gdn2 with
    g/b broadcast over key channels and w = beta broadcast over value
    channels (scalar gates commute out of the diagonal). Exercised by
    the degeneracy test against the shipped gdn kernels."""
    g2 = g_head[..., None].expand(*g_head.shape, d_k)
    b2 = beta_head[..., None].expand(*beta_head.shape, d_k)
    w2 = beta_head[..., None].expand(*beta_head.shape, d_v)
    return g2, b2, w2
