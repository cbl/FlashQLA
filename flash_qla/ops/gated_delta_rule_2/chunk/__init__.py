# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""GDN2 (GatedDeltaNet-2) chunked kernels. See DESIGN_GDR2.md.

Semantics follow fla's ``naive_recurrent_gdn2`` / ``chunk_gdn2``:

    S_t = (I - k_t (b_t*k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t*v_t)^T
    o_t = scale * S_t^T q_t

Kernel status (milestones in DESIGN_GDR2.md):
    M1 kkt_solve_2 / M2 prepare_h_2 / M3 fused_fwd_2 / M4 fused_bwd_2
"""

import glob
import os
import site

import torch
import tilelang


def _ensure_cuda_cccl() -> None:
    """tilelang-generated code can include <cuda/atomic> (libcu++/CCCL).
    Minimal CUDA installs keep those headers off nvcc's default include
    path; locate them and prepend CPATH so the in-process nvcc invocation
    finds them. No-op when already resolvable or genuinely absent."""
    patterns = [
        "/usr/local/cuda/include/cuda/atomic",
        "/usr/local/cuda-*/include/cuda/atomic",
    ]
    try:
        for sp in site.getsitepackages():
            patterns += [
                os.path.join(sp, "nvidia", "*", "include", "cuda", "atomic"),
                os.path.join(sp, "nvidia", "*", "*", "include", "cuda", "atomic"),
            ]
    except Exception:
        pass
    for pattern in patterns:
        for hit in glob.glob(pattern):
            inc = os.path.dirname(os.path.dirname(hit))
            cpath = os.environ.get("CPATH", "")
            if inc not in cpath.split(":"):
                os.environ["CPATH"] = f"{inc}:{cpath}" if cpath else inc
            return


_ensure_cuda_cccl()

from .prepare_inputs import prepare_inputs_2

_ARCH = tilelang.contrib.nvcc.get_target_compute_version()
if _ARCH == "9.0":
    from .hopper import kkt_solve
    prepare_h_2 = None  # hopper port pending (sm120-first development)
    CHUNK_SIZE_2 = 64
elif _ARCH == "12.0":
    from .blackwell_sm120 import kkt_solve
    from .blackwell_sm120.prepare_h import prepare_h_2
    CHUNK_SIZE_2 = 32
else:
    # Unsupported archs see the informative raise in chunk_gdn2 below
    # rather than an import failure.
    kkt_solve = None
    prepare_h_2 = None
    CHUNK_SIZE_2 = None


def chunk_gdn2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, HV, V]`.
            GVA (Grouped Value Attention) is applied if `HV > H`.
        g (torch.Tensor):
            per-KEY-CHANNEL forget gates of shape `[B, T, HV, K]`,
            in log space.
        b (torch.Tensor):
            per-KEY-CHANNEL erase gates of shape `[B, T, HV, K]`.
        w (torch.Tensor):
            per-VALUE-CHANNEL write gates of shape `[B, T, HV, V]`.
        scale (Optional[float]):
            Defaults to `1 / sqrt(K)`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, HV, K, V]`.
        output_final_state (Optional[bool]):
            Whether to return the final state `[N, HV, K, V]`.
        use_qk_l2norm_in_kernel (bool):
            Whether to apply L2norm to q/k internally.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths for variable-length training.

    Returns:
        o (torch.Tensor): `[B, T, HV, V]`.
        final_state (Optional[torch.Tensor]): `[N, HV, K, V]`.
    """
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype in (torch.bfloat16, torch.float16), (
        "FlashQLA only supports bfloat16 and float16."
    )
    assert v.shape[2] % k.shape[2] == 0, (
        "num_qk_heads must be divisible to num_v_heads."
    )
    assert g.shape == (*v.shape[:3], k.shape[-1]), (
        "g must be per key channel: [B, T, HV, K]."
    )
    assert b.shape == g.shape, "b must be per key channel: [B, T, HV, K]."
    assert w.shape == v.shape, "w must be per value channel: [B, T, HV, V]."
    raise NotImplementedError(
        "GDN2 kernels are under construction (M1-M4, see DESIGN_GDR2.md; "
        "SM90/Hopper first); reference implementations live in "
        "tests/ref_gdr2.py."
    )
