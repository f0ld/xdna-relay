"""Reference implementations for elementwise ops. Architect-owned (protected)."""

import numpy as np


def vector_scalar_mul_ref(a: np.ndarray, factor: int) -> np.ndarray:
    """c = a * factor in a's dtype. Inputs in tests never overflow, so wrap vs.
    saturate does not matter (see docs/knowledge/pitfalls.md)."""
    return (a.astype(np.int64) * int(factor)).astype(a.dtype)
