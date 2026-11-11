"""Op contract: data types, tensor specs, tolerance, and shape binding.

Pure Python + PyTorch on the CPU. No numpy, no NPU code.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, fields


# Symbolic dtype names → torch.dtype objects.
DTYPES: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "f16": torch.float16,
    "f32": torch.float32,
    "i8": torch.int8,
    "i16": torch.int16,
    "i32": torch.int32,
    "u8": torch.uint8,
}


@dataclass(frozen=True)
class TensorSpec:
    """Shape-independent description of a tensor (input or output)."""
    name: str
    shape: tuple[str | int, ...]
    dtype: str
    layout: str

    def __post_init__(self) -> None:
        # name must be non-empty
        if not self.name:
            raise ValueError("TensorSpec.name must be non-empty")
        # layout must be non-empty
        if not self.layout:
            raise ValueError("TensorSpec.layout must be non-empty")
        # dtype must be a known key
        if self.dtype not in DTYPES:
            raise ValueError(f"Unknown dtype {self.dtype!r}")
        # each dimension entry: must be a non-empty str (symbol) or int > 0 (literal);
        # bool is a subclass of int so check it first.
        for d in self.shape:
            if isinstance(d, bool) or not isinstance(d, (str, int)):
                raise ValueError(f"TensorSpec shape entry {d!r} is not a valid dimension")
            if isinstance(d, int) and d <= 0:
                raise ValueError(f"TensorSpec literal dimension must be > 0, got {d}")
            if isinstance(d, str) and not d:
                raise ValueError("TensorSpec symbol must be non-empty")


@dataclass(frozen=True)
class Tolerance:
    """Relative + absolute tolerance for backend output checks."""
    atol: float
    rtol: float

    def __post_init__(self) -> None:
        if self.atol < 0 or self.rtol < 0:
            raise ValueError("Tolerance.atol and Tolerance.rtol must be >= 0")

    def allclose(self, actual: torch.Tensor, expected: torch.Tensor) -> bool:
        """Return False if shapes differ (no exception). Otherwise compare in float64."""
        if tuple(actual.shape) != tuple(expected.shape):
            return False
        return torch.allclose(
            actual.to(torch.float64),
            expected.to(torch.float64),
            rtol=self.rtol,
            atol=self.atol,
        )


@dataclass(frozen=True)
class OpContract:
    """A backend-agnostic declaration of an operation."""
    name: str
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    params: tuple[str, ...]
    tolerance: Tolerance

    def __post_init__(self) -> None:
        # No two tensors (inputs + outputs together) share a name.
        seen: set[str] = set()
        for spec in self.inputs + self.outputs:
            if spec.name in seen:
                raise ValueError(f"Duplicate tensor name {spec.name!r} in contract {self.name!r}")
            seen.add(spec.name)
        # Every output shape symbol must appear in at least one input shape.
        input_symbols: set[str] = set()
        for spec in self.inputs:
            for d in spec.shape:
                if isinstance(d, str):
                    input_symbols.add(d)
        for spec in self.outputs:
            for d in spec.shape:
                if isinstance(d, str) and d not in input_symbols:
                    raise ValueError(
                        f"Output tensor {spec.name!r} uses symbol {d!r} "
                        f"that does not appear in any input"
                    )

    def bind_shapes(self, *inputs: torch.Tensor) -> dict[str, int]:
        """Bind symbolic shape entries from concrete input tensors.

        Raises ValueError if:
        - wrong number of inputs;
        - any input is not a torch.Tensor;
        - dtype, ndim, or literal dim doesn't match;
        - a symbol is bound to different sizes by different inputs.
        """
        if len(inputs) != len(self.inputs):
            raise ValueError(
                f"Expected {len(self.inputs)} inputs, got {len(inputs)}"
            )
        bindings: dict[str, int] = {}
        for spec, inp in zip(self.inputs, inputs):
            if not isinstance(inp, torch.Tensor):
                raise ValueError(f"Input for {spec.name!r} is not a torch.Tensor")
            expected_dtype = DTYPES[spec.dtype]
            if inp.dtype != expected_dtype:
                raise ValueError(
                    f"Input {spec.name!r}: expected dtype {expected_dtype}, "
                    f"got {inp.dtype}"
                )
            shape_tuple = tuple(inp.shape)
            if len(shape_tuple) != len(spec.shape):
                raise ValueError(
                    f"Input {spec.name!r}: expected {len(spec.shape)} dims, "
                    f"got {len(shape_tuple)}"
                )
            for sym_or_lit, actual_dim in zip(spec.shape, shape_tuple):
                if isinstance(sym_or_lit, int):
                    if sym_or_lit != actual_dim:
                        raise ValueError(
                            f"Input {spec.name!r}: literal dim {sym_or_lit} "
                            f"mismatch (got {actual_dim})"
                        )
                else:
                    # sym_or_lit is a str symbol
                    if sym_or_lit in bindings:
                        if bindings[sym_or_lit] != actual_dim:
                            raise ValueError(
                                f"Symbol {sym_or_lit!r} bound to {bindings[sym_or_lit]} "
                                f"and {actual_dim}"
                            )
                    else:
                        bindings[sym_or_lit] = actual_dim
        return bindings

    def output_shapes(self, bindings: dict[str, int]) -> tuple[tuple[int, ...], ...]:
        """Return the concrete output shapes after substituting symbols."""
        result: list[tuple[int, ...]] = []
        for spec in self.outputs:
            concrete = tuple(bindings[s] if isinstance(s, str) else s for s in spec.shape)
            result.append(concrete)
        return tuple(result)
