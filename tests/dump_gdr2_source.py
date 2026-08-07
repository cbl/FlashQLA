# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Dump the generated CUDA source of the gdn2 kkt_solve kernel and report
every atomic operation in it (with context lines).

    python tests/dump_gdr2_source.py [--full]

Atomic operations in this kernel are tilelang layout-fallback codegen,
not intended behavior; zero hits is the goal.
"""

import argparse

import torch
import tilelang


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true", help="print entire source")
    args = p.parse_args()

    arch = tilelang.contrib.nvcc.get_target_compute_version()
    if arch == "9.0":
        from flash_qla.ops.gated_delta_rule_2.chunk.hopper.kkt_solve import (
            tilelang_kkt_solve_2,
        )
        chunk = 64
    elif arch == "12.0":
        from flash_qla.ops.gated_delta_rule_2.chunk.blackwell_sm120.kkt_solve import (
            tilelang_kkt_solve_2,
        )
        chunk = 32
    else:
        raise SystemExit(f"unsupported arch {arch}")

    kernel = tilelang_kkt_solve_2(
        4, 2, 128, chunk,
        accum_dtype="float32",
        qkva_dtype=torch.bfloat16,
        g_dtype=torch.float32,
        seqlen_dtype="int32",
        is_varlen=False,
    )
    src = kernel.get_kernel_source()
    lines = src.splitlines()
    if args.full:
        print(src)
    hits = [i for i, l in enumerate(lines) if "atomic" in l.lower()]
    if not hits:
        print(f"OK: no atomic operations in generated source ({len(lines)} lines)")
        return
    print(f"{len(hits)} atomic line(s) in {len(lines)} lines of source:")
    for i in hits:
        for j in range(max(0, i - 2), min(len(lines), i + 3)):
            mark = ">>" if j == i else "  "
            print(f"{mark} {j:5d} {lines[j]}")
        print()


if __name__ == "__main__":
    main()
