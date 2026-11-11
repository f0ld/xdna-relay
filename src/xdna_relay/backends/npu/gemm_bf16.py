"""NPU backend for gemm_bf16 via IRON's GEMM operator.

Zero-pads every dimension to a multiple of 64, runs IRON GEMM, then slices the result.
"""

from __future__ import annotations

import functools

import numpy as np
import torch
from iron.common import Precision
from iron.operators.gemm import GEMM
from ml_dtypes import bfloat16

import aie.iron as iron

import xdna_relay.ops  # noqa: F401  (ensures the gemm_bf16 contract is in DEFAULT)
from xdna_relay.ops.registry import DEFAULT


PAD = 64  # every padded dim is a multiple of this (GEMM's default tile_k/tile_n)


def padded_shape(m: int, k: int, n: int) -> tuple[int, int, int]:
    """Each of m, k, n rounded UP to a multiple of PAD.

    >>> padded_shape(100, 300, 70)
    (128, 320, 128)
    """
    return (
        (m + PAD - 1) // PAD * PAD,
        (k + PAD - 1) // PAD * PAD,
        (n + PAD - 1) // PAD * PAD,
    )


@functools.cache
def gemm_op(m_pad: int, k_pad: int, n_pad: int) -> GEMM:
    """One IRON operator per padded shape, cached across calls."""
    return GEMM(M=m_pad, K=k_pad, N=n_pad, precision=Precision.HIGHEST)


def to_numpy_bf16(t: torch.Tensor) -> np.ndarray:
    """torch bf16 → numpy ml_dtypes.bfloat16, same shape, bit-exact."""
    return t.contiguous().view(torch.int16).numpy().view(bfloat16)


def to_torch_bf16(a: np.ndarray) -> torch.Tensor:
    """numpy ml_dtypes.bfloat16 → torch bf16, same shape, bit-exact, owns its memory."""
    return torch.from_numpy(np.ascontiguousarray(a).view(np.int16).copy()).view(torch.bfloat16)


def gemm_bf16_npu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = A @ B on the NPU. a: (M, K) bf16, b: (K, N) bf16 (any strides).

    Shapes and dtypes are already checked by Registry.run; do not re-check them.
    Raises ValueError if M, K or N is 0 (checked before any NPU call).
    """
    m, k = a.shape
    n = b.shape[1]

    # Validate dimensions before any NPU call.
    if m == 0 or k == 0 or n == 0:
        raise ValueError(f"gemm_bf16_npu: dimensions must be > 0, got ({m}, {k}, {n})")

    # Pad up to multiples of 64.
    mp, kp, np_ = padded_shape(m, k, n)

    # Zero-padded host tensors; slice assignment handles any strides.
    ap = torch.zeros(mp, kp, dtype=torch.bfloat16)
    ap[:m, :k] = a
    bp = torch.zeros(kp, np_, dtype=torch.bfloat16)
    bp[:k, :n] = b

    # Move to NPU device tensors.
    A = iron.tensor(to_numpy_bf16(ap), dtype=bfloat16, device="npu")
    B = iron.tensor(to_numpy_bf16(bp), dtype=bfloat16, device="npu")
    C = iron.zeros((mp, np_), dtype=bfloat16, device="npu")

    # Run GEMM (compiles on first shape, cached thereafter).
    gemm_op(mp, kp, np_)(A, B, C)

    # Slice off padding and return contiguous.
    return to_torch_bf16(C.numpy().reshape(mp, np_))[:m, :n].contiguous()


# Register the NPU backend into the ops DEFAULT registry.
DEFAULT.register("gemm_bf16", "npu", gemm_bf16_npu)
