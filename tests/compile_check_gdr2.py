# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""GPU-less compile check for the gdn2 kernels (explicit sm_120 target).

    python tests/compile_check_gdr2.py [sm_120|sm_90a]

Compiles every gdn2 tilelang kernel factory at representative shapes on
a machine with no GPU (pip-shipped nvcc suffices). Catches the whole
compile-error class — TVM lowering asserts, layout/atomic fallbacks,
GEMM warp-partition checks — before anything touches real hardware.
"""

import importlib.util
import os
import sys
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCH = sys.argv[1] if len(sys.argv) > 1 else "sm_120"
TARGET = {"kind": "cuda", "arch": ARCH}

import tilelang  # noqa: E402

_orig_jit = tilelang.jit


def _patched_jit(*args, **kwargs):
    kwargs.setdefault("target", TARGET)
    return _orig_jit(*args, **kwargs)


tilelang.jit = _patched_jit

# The kernel modules import flash_qla.utils, whose package __init__ needs a
# GPU; stub just what they use so the files load standalone.
fake_utils = types.ModuleType("flash_qla.utils")
fake_utils.prepare_chunk_indices = lambda *a, **k: None
fake_pkg = types.ModuleType("flash_qla")
fake_pkg.utils = fake_utils
sys.modules.setdefault("flash_qla", fake_pkg)
sys.modules.setdefault("flash_qla.utils", fake_utils)


def load(relpath, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BASE = "flash_qla/ops/gated_delta_rule_2/chunk/"
BF16, F32 = torch.bfloat16, torch.float32

CASES = [
    ("seg_march_zero", BASE + "blackwell_sm120/seg_march.py",
     "tilelang_seg_march_2",
     dict(H=4, DK=128, DV=128, chunk_size=32, scale=0.088388,
          accum_dtype="float32", qkva_dtype=BF16, gend_dtype=F32,
          st_dtype=F32, o_dtype=BF16, mode="zero", use_h0=True,
          v_split=4)),
    ("seg_march_homo", BASE + "blackwell_sm120/seg_march.py",
     "tilelang_seg_march_2",
     dict(H=4, DK=128, DV=128, chunk_size=32, scale=0.088388,
          accum_dtype="float32", qkva_dtype=BF16, gend_dtype=F32,
          st_dtype=F32, o_dtype=BF16, mode="homo", use_h0=False,
          v_split=4)),
    ("seg_march_corr", BASE + "blackwell_sm120/seg_march.py",
     "tilelang_seg_march_2",
     dict(H=4, DK=128, DV=128, chunk_size=32, scale=0.088388,
          accum_dtype="float32", qkva_dtype=BF16, gend_dtype=F32,
          st_dtype=F32, o_dtype=BF16, mode="corr", use_h0=False,
          v_split=4)),
    ("prefold_gram", BASE + "blackwell_sm120/prefold_gram.py",
     "tilelang_prefold_gram_2",
     dict(H=4, Hg=2, DK=128, chunk_size=32, accum_dtype="float32",
          qkva_dtype=BF16, g_in_dtype=F32, do_l2norm=True)),
    ("prefold_2b", BASE + "prepare_inputs_tl.py",
     "tilelang_prepare_inputs_2b",
     dict(H=4, Hg=2, DK=128, chunk_size=32, accum_dtype="float32",
          qkva_dtype=BF16, g_in_dtype=F32, do_l2norm=True)),
    ("gcs", BASE + "prepare_inputs_tl.py",
     "tilelang_gcs_2",
     dict(H=4, DK=128, chunk_size=32, accum_dtype="float32",
          g_in_dtype=F32)),
    ("fused_march", BASE + "blackwell_sm120/fused_march.py",
     "tilelang_fused_march_2",
     dict(H=4, Hg=1, DK=128, DV=128, chunk_size=32, scale=0.088388,
          accum_dtype="float32", qkva_dtype=BF16, gend_dtype=F32,
          ht_dtype=F32, o_dtype=BF16, use_initial_state=True,
          store_final_state=True, v_split=4)),
    ("prepare_inputs_tl", BASE + "prepare_inputs_tl.py",
     "tilelang_prepare_inputs_2",
     dict(H=4, Hg=2, DK=128, DV=128, chunk_size=32, accum_dtype="float32",
          qkva_dtype=BF16, g_in_dtype=F32)),
    ("kkt_solve", BASE + "blackwell_sm120/kkt_solve.py",
     "tilelang_kkt_solve_2",
     dict(H=4, Hg=2, DK=128, chunk_size=32, accum_dtype="float32",
          qkva_dtype=BF16, g_dtype=F32, seqlen_dtype="int32",
          is_varlen=False)),
    ("prepare_h", BASE + "blackwell_sm120/prepare_h.py",
     "tilelang_prepare_h_2",
     dict(H=4, DK=128, DV=128, chunk_size=32, accum_dtype="float32",
          qkva_dtype=BF16, gend_dtype=F32, h_dtype=BF16, ht_dtype=F32,
          use_initial_state=True, store_final_state=True, store_h=True,
          v_split=4)),
    ("fused_fwd", BASE + "blackwell_sm120/fused_fwd.py",
     "tilelang_fused_fwd_2",
     dict(H=4, DK=128, DV=128, chunk_size=32, scale=0.088388,
          accum_dtype="float32", qkva_dtype=BF16, h_dtype=BF16,
          o_dtype=BF16, include_intra=True, include_inter=True)),
]


def main():
    failed = 0
    for name, path, factory_name, kwargs in CASES:
        try:
            mod = load(path, f"cc_{name}")
            kernel = getattr(mod, factory_name)(**kwargs)
            src = kernel.get_kernel_source()
            atomics = sum("atomic" in l.lower() for l in src.splitlines())
            extra = f", {atomics} atomic lines" if atomics else ""
            print(f"{name:18} OK  ({len(src.splitlines())} lines{extra})")
            if atomics:
                failed += 1  # atomics are always a layout bug in these kernels
        except Exception as e:
            failed += 1
            msg = str(e).strip().splitlines()
            tail = msg[-1][:200] if msg else type(e).__name__
            print(f"{name:18} FAIL  {type(e).__name__}: {tail}")
    print(f"\n{ARCH}: {len(CASES) - failed}/{len(CASES)} kernels compile clean")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
