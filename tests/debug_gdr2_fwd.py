# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Bisect the gdn2 forward against torch references built from the
kernels' own validated intermediates.

    python tests/debug_gdr2_fwd.py

Prints relative errors for: kkt's attention output, the full output,
the intra term alone (attn @ R), the inter term alone (eq @ S), and a
sanity check that the torch-side term decomposition reproduces the full
reference.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ref_gdr2 import _causal_decay, _pad_chunks, chunk_gdn2_fwd_ref  # noqa: E402
from test_gdr2_unit import _make_inputs_2, _rel  # noqa: E402

from flash_qla.ops.gated_delta_rule_2.chunk import (  # noqa: E402
    CHUNK_SIZE_2,
    kkt_solve,
    prepare_h_2,
    prepare_inputs_2,
)
from flash_qla.ops.gated_delta_rule_2.chunk.blackwell_sm120.fused_fwd import (  # noqa: E402
    fused_fwd_2,
)


def main():
    B, T, Hk, Hv = 1, 256, 4, 4
    chunk = CHUNK_SIZE_2
    q, k, v, g, b, w, h0 = _make_inputs_2(B, T, Hk, Hv)
    q_bf, k_bf = q.to(torch.bfloat16), k.to(torch.bfloat16)
    v_bf = v.to(torch.bfloat16)
    b_bf, w_bf = b.to(torch.bfloat16), w.to(torch.bfloat16)
    g32 = g.float()
    scale = 128 ** -0.5

    g_cs, eq, ekb, kte, mv, gend = prepare_inputs_2(
        q_bf, k_bf, v_bf, g32, b_bf, w_bf, chunk)
    a, attn_k = kkt_solve(k_bf, g_cs, b_bf, q_bf, chunk_size=chunk)
    h, ht, r = prepare_h_2(ekb, kte, mv, a, gend, initial_state=h0.float())

    # torch references from the SAME r/h the kernel reads
    qc = _pad_chunks(q_bf.float(), chunk)
    kc = _pad_chunks(k_bf.float(), chunk)
    gc = _pad_chunks(g32, chunk).cumsum(dim=2)
    rc = _pad_chunks(r.float(), chunk)
    diff = _causal_decay(gc)
    attn = torch.einsum("bnihk,bnijhk,bnjhk->bnhij", qc, diff, kc)
    attn_rows = attn.permute(0, 1, 3, 2, 4).reshape(B, -1, Hv, chunk)[:, :T]
    print(f"attn : rel={_rel(attn_k[:, :T], attn_rows.double()):.3e}")

    o_intra = scale * torch.einsum("bnhij,bnjhv->bnihv", attn, rc)
    o_inter = scale * torch.einsum(
        "bnchk,bnhkv->bnchv", gc.exp() * qc, h.float()
    )
    o_intra = o_intra.reshape(B, -1, Hv, 128)[:, :T]
    o_inter = o_inter.reshape(B, -1, Hv, 128)[:, :T]

    for label, intra, inter, ref in (
        ("full ", True, True, o_intra + o_inter),
        ("intra", True, False, o_intra),
        ("inter", False, True, o_inter),
    ):
        o_k = fused_fwd_2(eq, attn_k, r, h, scale,
                          include_intra=intra, include_inter=inter)
        print(f"{label}: rel={_rel(o_k, ref.double()):.3e}")

    o_ref, _ = chunk_gdn2_fwd_ref(
        q_bf.double(), k_bf.double(), v_bf.double(), g32.double(),
        b_bf.double(), w_bf.double(), initial_state=h0.double(),
        chunk_size=chunk,
    )
    print(f"sanity: torch-terms vs full ref rel="
          f"{_rel((o_intra + o_inter).double(), o_ref):.3e}")


if __name__ == "__main__":
    main()
