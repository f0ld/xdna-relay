"""gemm_bf16 op contract and PyTorch reference backend.

C = A @ B, accumulated in float32 for reproducibility.
"""

import torch

from .contract import OpContract, TensorSpec, Tolerance
from .registry import DEFAULT


GEMM_BF16 = OpContract(
    name="gemm_bf16",
    inputs=(
        TensorSpec("A", ("M", "K"), "bf16", "row_major"),
        TensorSpec("B", ("K", "N"), "bf16", "row_major"),
    ),
    outputs=(TensorSpec("C", ("M", "N"), "bf16", "row_major"),),
    params=(),
    tolerance=Tolerance(atol=1e-2, rtol=2e-2),
)


def gemm_bf16_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = A @ B. Upcast both to float32, matmul, cast to bfloat16."""
    return (a.to(torch.float32) @ b.to(torch.float32)).to(torch.bfloat16)


# Register into the module-level DEFAULT registry.
DEFAULT.add_contract(GEMM_BF16)
DEFAULT.register("gemm_bf16", "ref", gemm_bf16_ref)
