"""Backend registry: maps (op, backend) to implementations with contract validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .contract import OpContract, DTYPES


@dataclass(frozen=True)
class Backend:
    """A registered implementation for an op."""
    name: str
    impl: Callable[..., Any]
    prepare: Callable[..., Any] | None = field(default=None)


class Registry:
    """Backend registry — instances share nothing (no class-level state)."""

    def __init__(self) -> None:
        self._contracts: dict[str, OpContract] = {}
        # (op, backend_name) → Backend
        self._backends: dict[tuple[str, str], Backend] = {}

    def add_contract(self, contract: OpContract) -> None:
        """Register an op contract. KeyError if name already added."""
        if contract.name in self._contracts:
            raise ValueError(f"Contract name {contract.name!r} already registered")
        self._contracts[contract.name] = contract

    def contract(self, op: str) -> OpContract:
        """Look up a contract by op name. KeyError if unknown."""
        c = self._contracts.get(op)
        if c is None:
            raise KeyError(op)
        return c

    def ops(self) -> tuple[str, ...]:
        """Sorted tuple of registered op names."""
        return tuple(sorted(self._contracts))

    def register(self, op: str, backend: str, impl: Callable[..., Any],
                 prepare: Callable[..., Any] | None = None) -> None:
        """Register a backend implementation for an op."""
        if op not in self._contracts:
            raise KeyError(op)
        if (op, backend) in self._backends:
            raise ValueError(
                f"Backend {backend!r} already registered for op {op!r}"
            )
        self._backends[(op, backend)] = Backend(name=backend, impl=impl, prepare=prepare)

    def backend(self, op: str, backend: str) -> Backend:
        """Look up a backend by (op, name). KeyError if unknown."""
        b = self._backends.get((op, backend))
        if b is None:
            raise KeyError(f"({op!r}, {backend!r})")
        return b

    def backends(self, op: str) -> tuple[str, ...]:
        """Sorted tuple of backend names for an op. () if none; KeyError if op unknown."""
        if op not in self._contracts:
            raise KeyError(op)
        return tuple(sorted(name for o, name in self._backends if o == op))

    def run(self, op: str, backend: str, *inputs, **params):
        """Run an op through the registry with full validation.

        Steps:
        1. Look up contract and backend (KeyError if unknown).
        2. ValueError if set(params) != set(contract.params).
        3. Bind shapes from inputs (raises ValueError on mismatch).
        4. Call impl(*inputs, **params).
        5. Validate output count, dtype, and shape.
        6. Return single output as Tensor, or tuple of outputs.
        """
        contract = self.contract(op)
        be = self.backend(op, backend)

        # Step 2: validate params
        if set(params) != set(contract.params):
            raise ValueError(
                f"Params mismatch for {op!r}: expected {set(contract.params)}, "
                f"got {set(params)}"
            )

        # Step 3: bind shapes
        bindings = contract.bind_shapes(*inputs)

        # Step 4: call impl
        out = be.impl(*inputs, **params)
        outs = out if isinstance(out, tuple) else (out,)

        # Step 5: validate outputs
        if len(outs) != len(contract.outputs):
            raise RuntimeError(
                f"Output count mismatch for {op!r}: expected {len(contract.outputs)}, "
                f"got {len(outs)}"
            )
        expected_shapes = contract.output_shapes(bindings)
        for i, (o, spec) in enumerate(zip(outs, contract.outputs)):
            if not isinstance(o, torch.Tensor):
                raise RuntimeError(
                    f"Output {i} for {op!r} is not a torch.Tensor"
                )
            if o.dtype != DTYPES[spec.dtype]:
                raise RuntimeError(
                    f"Output {i} dtype for {op!r}: expected {DTYPES[spec.dtype]}, "
                    f"got {o.dtype}"
                )
            if tuple(o.shape) != expected_shapes[i]:
                raise RuntimeError(
                    f"Output {i} shape for {op!r}: expected {expected_shapes[i]}, "
                    f"got {tuple(o.shape)}"
                )

        # Step 6: return
        return outs[0] if len(outs) == 1 else outs


DEFAULT = Registry()   # module-level registry that op modules register into
