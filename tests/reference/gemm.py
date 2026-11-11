"""Reference implementations for GEMM ops. Architect-owned (protected).

Independent oracle: computes in float64 so backend outputs (bf16/f32) are compared
against the exact product, not against another low-precision implementation.
"""

import torch


def gemm_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = A @ B in float64. a: (M, K), b: (K, N). Returns (M, N) float64."""
    return a.to(torch.float64) @ b.to(torch.float64)
