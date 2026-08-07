# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Decay-folded operands for the GDN2 chunk kernels.

GDN2's per-key-channel decay cannot stay a scalar side-channel like gdn's
(see DESIGN_GDR2.md): folding it into the operand tensors AHEAD of the
kernels keeps every exponent bounded (all exponents here are <= 0) and
keeps the kernels' shared-memory footprint at gdn levels — promoting the
gates to [chunk, 128] tiles inside the kernels instead would not fit
SM120. Torch implementation; a fused elementwise kernel is a later
optimization (fla's Triton gdn2 materializes comparable intermediates).

Given per-chunk inclusive cumsum G of the log decay g:

    g_cs      = G                                  [B, T, H, K]  fp32
    ekb       = exp(G) * b * k                     [B, T, H, K]  k.dtype
    k_to_end  = exp(G_end - G) * k                 [B, T, H, K]  k.dtype
    mv        = w * v                              [B, T, H, V]  v.dtype
    g_end_exp = exp(G_end) per chunk               [B, N, H, K]  fp32

k is expanded from Hg to H heads (the gates are per value head, so the
folded tensors cannot stay group-shared).
"""

import torch


def prepare_inputs_2(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    chunk_size: int,
):
    bsz, t, hg, d_k = k.shape
    h = g.shape[2]
    if hg != h:
        k = k.repeat_interleave(h // hg, dim=2)

    pad = (-t) % chunk_size
    gp = g.float()
    if pad:
        # Zero-padding the log decay makes padded rows carry decay 1, so
        # G_end matches the last real row and padded contributions vanish
        # with the zero-padded k/v in the kernels.
        gp = torch.cat([gp, gp.new_zeros(bsz, pad, h, d_k)], dim=1)
    gc = gp.view(bsz, -1, chunk_size, h, d_k).cumsum(dim=2)
    g_end = gc[:, :, -1]                                    # [B, N, H, K]
    g_cs = gc.reshape(bsz, -1, h, d_k)[:, :t].contiguous()
    to_end = (
        g_end[:, :, None] - gc
    ).reshape(bsz, -1, h, d_k)[:, :t]

    ekb = (g_cs.exp() * b.float() * k.float()).to(k.dtype)
    k_to_end = (to_end.exp() * k.float()).to(k.dtype)
    mv = (w.float() * v.float()).to(v.dtype)
    return g_cs, ekb, k_to_end, mv, g_end.exp().contiguous()
