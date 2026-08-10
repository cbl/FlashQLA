# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Profile GDN2 kernels under torch.profiler.

    python profile/profile_gdr2.py [--impl fla|qla] [--seq 8192] [--bwd]

Prints the CUDA-time kernel table and optionally dumps a chrome trace.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
from bench_gated_delta_rule_2 import bench_impl, make_inputs  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--impl", choices=["fla", "qla"], default="fla")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=8192)
    p.add_argument("--hk", type=int, default=8)
    p.add_argument("--hv", type=int, default=16)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--bwd", action="store_true")
    p.add_argument("--table", action="store_true",
                   help="also print the full per-kernel table")
    p.add_argument("--trace", default=None, metavar="OUT.json")
    args = p.parse_args()

    if args.impl == "fla":
        from fla.ops.gdn2 import chunk_gdn2 as kernel
        expand_qk = True
    else:
        from flash_qla import chunk_gdn2 as kernel
        expand_qk = False

    inputs = make_inputs(args.batch, args.seq, args.hk, args.hv,
                         args.dim, args.dim)
    bench_impl(kernel, inputs, expand_qk, warmup=3, iters=2, bwd=args.bwd)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        bench_impl(kernel, inputs, expand_qk, warmup=0,
                   iters=args.iters, bwd=args.bwd)
    events = prof.key_averages()
    buckets = {"kkt": 0.0, "march": 0.0, "fused": 0.0, "gcs": 0.0,
               "other": 0.0}
    for ev in events:
        t = getattr(ev, "self_device_time_total",
                    getattr(ev, "self_cuda_time_total", 0.0))
        name = ev.key.lower()
        if "kkt" in name:
            buckets["kkt"] += t
        elif "prepare_h" in name or "fused_march" in name:
            buckets["march"] += t
        elif "fused_fwd" in name:
            buckets["fused"] += t
        elif "gcs" in name:
            buckets["gcs"] += t
        else:
            buckets["other"] += t
    total = sum(buckets.values()) or 1.0
    print("-- compact (self GPU time; 'other' = torch precompute/copies) --")
    for name, t in buckets.items():
        print(f"{name:6} {t / 1e3:8.2f}ms {100 * t / total:5.1f}%")
    if args.table:
        print(prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=25))
    if args.trace:
        prof.export_chrome_trace(args.trace)
        print(f"trace written to {args.trace}")


if __name__ == "__main__":
    main()
