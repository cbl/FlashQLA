# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Benchmark GDN2: flash_qla.chunk_gdn2 vs fla.ops.gdn2.chunk_gdn2.

    python benchmark/bench_gated_delta_rule_2.py [--bwd] [--hk 8 --hv 16]

Until the flash_qla kernels land (M3/M4 in DESIGN_GDR2.md) this reports
fla numbers alone — the baseline the fwd-bench gate compares against.
"""

import argparse
import time

import torch
import torch.nn.functional as F


def make_inputs(batch, seq, hk, hv, d_k, d_v, device="cuda"):
    torch.manual_seed(0)
    q = F.normalize(torch.randn(batch, seq, hk, d_k, device=device,
                                dtype=torch.bfloat16), dim=-1)
    k = F.normalize(torch.randn(batch, seq, hk, d_k, device=device,
                                dtype=torch.bfloat16), dim=-1)
    v = torch.randn(batch, seq, hv, d_v, device=device, dtype=torch.bfloat16)
    g = (-F.softplus(torch.randn(batch, seq, hv, d_k, device=device) * 0.5 + 1.0)
         ).to(torch.bfloat16)
    b = torch.rand(batch, seq, hv, d_k, device=device).sigmoid().to(torch.bfloat16)
    w = torch.rand(batch, seq, hv, d_v, device=device).sigmoid().to(torch.bfloat16)
    return dict(q=q, k=k, v=v, g=g, b=b, w=w)


def timeit(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def bench_impl(kernel, inputs, expand_qk, warmup, iters, bwd):
    args = dict(inputs)
    if expand_qk:  # fla's chunk_gdn2 wants equal head counts
        groups = args["v"].shape[2] // args["q"].shape[2]
        if groups > 1:
            args["q"] = args["q"].repeat_interleave(groups, dim=2)
            args["k"] = args["k"].repeat_interleave(groups, dim=2)

    def fwd():
        with torch.no_grad():
            kernel(**args, output_final_state=True,
                   use_qk_l2norm_in_kernel=True)

    fwd_ms = timeit(fwd, warmup, iters)
    if not bwd:
        return fwd_ms, None
    grad_args = {n: t.detach().clone().requires_grad_(True)
                 for n, t in args.items()}

    def step():
        o, s = kernel(**grad_args, output_final_state=True,
                      use_qk_l2norm_in_kernel=True)
        (o.float().square().mean() + 0.1 * s.float().square().mean()).backward()
        for t in grad_args.values():
            t.grad = None

    return fwd_ms, timeit(step, warmup, iters)


def _chunk_cumsum(g, chunk_size=64):
    bsz, t = g.shape[:2]
    pad = (-t) % chunk_size
    if pad:
        g = torch.cat([g, g.new_zeros(bsz, pad, *g.shape[2:])], dim=1)
    g = g.view(bsz, -1, chunk_size, *g.shape[2:]).cumsum(dim=2)
    return g.reshape(bsz, -1, *g.shape[3:])[:, :t]


def bench_kkt_stage(args):
    """Time gdn2 kkt_solve_2 against the gdn kkt_solve at the same
    shapes — a lower bound, since gdn2's A-build does strictly more
    work. An early perf signal available before the full forward (M3)."""
    from flash_qla.ops.gated_delta_rule.chunk import kkt_solve as kkt_gdn
    from flash_qla.ops.gated_delta_rule_2.chunk import kkt_solve as kkt_gdn2
    assert kkt_gdn2 is not None, "gdn2 kkt_solve targets SM90 only"

    print(f"kkt stage bench  B={args.batch} hk={args.hk} hv={args.hv} "
          f"d={args.dim} bf16")
    print(f"{'seq':>8} {'gdn kkt':>10} {'gdn2 kkt':>10} {'ratio':>6}")
    for seq in args.seqlens:
        inputs = make_inputs(args.batch, seq, args.hk, args.hv,
                             args.dim, args.dim)
        k, g, b = inputs["k"], inputs["g"], inputs["b"]
        beta_head = b.float().mean(-1)                 # gdn's scalar beta
        g_cs = _chunk_cumsum(g.float())
        t_gdn = timeit(lambda: kkt_gdn(k=k, b=beta_head),
                       args.warmup, args.iters)
        t_gdn2 = timeit(lambda: kkt_gdn2(k, g_cs, b),
                        args.warmup, args.iters)
        print(f"{seq:>8} {t_gdn:9.3f}ms {t_gdn2:9.3f}ms "
              f"{t_gdn2 / t_gdn:5.2f}x")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seqlens", type=int, nargs="+",
                   default=[2048, 4096, 8192, 16384, 32768])
    p.add_argument("--hk", type=int, default=8)
    p.add_argument("--hv", type=int, default=16)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--bwd", action="store_true")
    p.add_argument("--stage", choices=["full", "kkt"], default="full",
                   help="'kkt' times the gdn2 A-build kernel alone vs gdn's")
    args = p.parse_args()

    if args.stage == "kkt":
        bench_kkt_stage(args)
        return

    from fla.ops.gdn2 import chunk_gdn2 as fla_gdn2
    try:
        from flash_qla import chunk_gdn2 as qla_gdn2
        qla_gdn2(**make_inputs(1, 128, args.hk, args.hv, args.dim, args.dim))
        have_qla = True
    except Exception as e:
        have_qla = False
        qla_reason = f"{type(e).__name__}: {e}"

    mode = "fwd+bwd" if args.bwd else "fwd"
    print(f"GDN2 bench  B={args.batch} hk={args.hk} hv={args.hv} "
          f"d={args.dim} bf16  ({mode})")
    if not have_qla:
        print(f"  flash_qla.chunk_gdn2 unavailable ({qla_reason}); "
              f"fla baseline only")
    print(f"{'seq':>8} {'fla fwd':>10} {'qla fwd':>10} {'x':>6} "
          f"{'fla f+b':>10} {'qla f+b':>10} {'x':>6}")
    for seq in args.seqlens:
        inputs = make_inputs(args.batch, seq, args.hk, args.hv,
                             args.dim, args.dim)
        ff, fb = bench_impl(fla_gdn2, inputs, True, args.warmup,
                            args.iters, args.bwd)
        row = f"{seq:>8} {ff:9.3f}ms"
        if have_qla:
            qf, qb = bench_impl(qla_gdn2, inputs, False, args.warmup,
                                args.iters, args.bwd)
            row += f" {qf:9.3f}ms {ff / qf:5.2f}x"
            if fb is not None:
                row += f" {fb:9.3f}ms {qb:9.3f}ms {fb / qb:5.2f}x"
        elif fb is not None:
            row += f" {'':>10} {'':>6} {fb:9.3f}ms"
        print(row)


if __name__ == "__main__":
    main()
