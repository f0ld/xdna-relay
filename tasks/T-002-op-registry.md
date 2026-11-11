# T-002 — Op contract + backend registry skeleton, `ref` backend for `gemm_bf16`
Status: DONE
Milestone: M1
Depends on: T-001 (DONE)

## Goal
Create the backend-agnostic op layer from ADR 0001 / `docs/ARCHITECTURE.md` §1: a frozen
`OpContract` that declares an op's tensors (symbolic shapes, dtype, layout), params and
tolerance; a `Registry` that maps (op, backend) to an implementation and checks inputs and
outputs against the contract; and the first op, `gemm_bf16`, with its PyTorch `ref` backend.
Pure Python + PyTorch on the CPU. No NPU code in this task.

## Files you may change
- `src/xdna_relay/ops/__init__.py` (create)
- `src/xdna_relay/ops/contract.py` (create)
- `src/xdna_relay/ops/registry.py` (create)
- `src/xdna_relay/ops/gemm_bf16.py` (create)
- `docs/knowledge/pitfalls.md` (append only, if you hit something new)
- `PLAN.md` (only the T-002 status line), `handoff/T-002-iN.md`

Do **not** create `src/xdna_relay/backends/`. NPU backends come in T-004.

## Files to read first
- `tests/kernels/test_t002_op_registry.py`: the tests you must pass (do not edit)
- `tests/reference/gemm.py`: the float64 oracle (do not edit)
- `docs/ARCHITECTURE.md` §1 "Op contracts" (the target shape; this spec is the exact interface)

## Interface / contract
Only `torch` and the Python standard library (`dataclasses`, `typing`). No numpy in `ops/`.
All error checks raise exactly the exception type named below.

### `src/xdna_relay/ops/contract.py`

```python
DTYPES: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32,
    "i8": torch.int8, "i16": torch.int16, "i32": torch.int32, "u8": torch.uint8,
}

@dataclass(frozen=True)
class TensorSpec:
    name: str                      # non-empty
    shape: tuple[str | int, ...]   # each entry: non-empty str (symbol) or int > 0 (literal)
    dtype: str                     # a key of DTYPES
    layout: str                    # non-empty, e.g. "row_major"; not checked at runtime
    # __post_init__: raise ValueError if any rule above is broken.
    # A bool is NOT a valid dim (bool is a subclass of int: check `isinstance(d, bool)` first).

@dataclass(frozen=True)
class Tolerance:
    atol: float                    # >= 0
    rtol: float                    # >= 0
    # __post_init__: ValueError if atol < 0 or rtol < 0.
    def allclose(self, actual: torch.Tensor, expected: torch.Tensor) -> bool:
        """False if shapes differ (no exception). Otherwise
        torch.allclose(actual.to(torch.float64), expected.to(torch.float64),
                       rtol=self.rtol, atol=self.atol)."""

@dataclass(frozen=True)
class OpContract:
    name: str
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    params: tuple[str, ...]        # names of required keyword params, e.g. ("eps",)
    tolerance: Tolerance           # how close any backend must be to the oracle
    # __post_init__: ValueError if
    #   - two TensorSpecs (inputs + outputs together) share a name, or
    #   - an output shape uses a symbol that appears in no input shape.

    def bind_shapes(self, *inputs: torch.Tensor) -> dict[str, int]:
        """Check inputs against self.inputs and return {symbol: size}.
        ValueError if: len(inputs) != len(self.inputs); an input is not a torch.Tensor;
        its dtype != DTYPES[spec.dtype]; its ndim != len(spec.shape); a literal dim
        differs; a symbol already bound gets a different size."""

    def output_shapes(self, bindings: dict[str, int]) -> tuple[tuple[int, ...], ...]:
        """Concrete shape of every output, in order (symbols replaced from bindings)."""
```

### `src/xdna_relay/ops/registry.py`

```python
@dataclass(frozen=True)
class Backend:
    name: str                                   # "ref", "npu", ...
    impl: Callable[..., Any]                    # impl(*inputs, **params) -> Tensor | tuple[Tensor, ...]
    prepare: Callable[..., Any] | None = None   # stored only; run() never calls it

class Registry:
    def __init__(self) -> None: ...             # empty; instances share nothing
    def add_contract(self, contract: OpContract) -> None   # ValueError if name already added
    def contract(self, op: str) -> OpContract              # KeyError if unknown
    def ops(self) -> tuple[str, ...]                       # sorted op names
    def register(self, op: str, backend: str, impl, prepare=None) -> None
        # KeyError if op unknown; ValueError if (op, backend) already registered
    def backend(self, op: str, backend: str) -> Backend    # KeyError if op or backend unknown
    def backends(self, op: str) -> tuple[str, ...]         # sorted; () if none; KeyError if op unknown
    def run(self, op: str, backend: str, *inputs, **params):
        """1. look up contract and backend (KeyError as above)
           2. ValueError if set(params) != set(contract.params)
           3. bindings = contract.bind_shapes(*inputs)      (raises ValueError)
           4. out = impl(*inputs, **params); outs = out if isinstance(out, tuple) else (out,)
           5. RuntimeError if len(outs) != len(contract.outputs), or any output is not a
              torch.Tensor, has dtype != DTYPES[spec.dtype], or
              tuple(shape) != contract.output_shapes(bindings)[i]
           6. return outs[0] if there is exactly one output, else the tuple"""

DEFAULT = Registry()   # module-level registry that op modules register into
```

### `src/xdna_relay/ops/gemm_bf16.py`

```python
GEMM_BF16 = OpContract(
    name="gemm_bf16",
    inputs=(TensorSpec("A", ("M", "K"), "bf16", "row_major"),
            TensorSpec("B", ("K", "N"), "bf16", "row_major")),
    outputs=(TensorSpec("C", ("M", "N"), "bf16", "row_major"),),
    params=(),
    tolerance=Tolerance(atol=1e-2, rtol=2e-2),
)

def gemm_bf16_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = A @ B. Upcast both to float32, matmul, cast the result to bfloat16."""

# at module level, after the definitions:
DEFAULT.add_contract(GEMM_BF16)
DEFAULT.register("gemm_bf16", "ref", gemm_bf16_ref)
```

### `src/xdna_relay/ops/__init__.py`
One docstring line plus `from . import gemm_bf16  # noqa: F401` so that `import xdna_relay.ops`
fills `DEFAULT`. Python runs a module body only once per process, so importing it twice does
not register twice. Do not add guards or try/except around the registration.

## Steps
1. `git switch -c task/T-002-op-registry main`; set T-002 to `IN PROGRESS` in `PLAN.md`.
2. Run `scripts/test.sh T-002`. Expect `FAIL 0/35` (`xdna_relay.ops` missing). This is the baseline.
3. Write `contract.py` exactly as in the interface (`DTYPES`, `TensorSpec`, `Tolerance`, `OpContract`).
   Run `scripts/test.sh T-002 -k "dtypes or tensorspec or tolerance or contract_rejects or bind_shapes"`: must print `PASS 18/18`.
4. Write `registry.py` (`Backend`, `Registry`, `DEFAULT`). Run
   `scripts/test.sh T-002 -k "not gemm"`: must print `PASS 28/28`.
5. Write `gemm_bf16.py` and `ops/__init__.py`.
6. Run `scripts/test.sh T-002`. Must print `PASS 35/35`. Then do the end-of-iteration steps in
   AGENTS.md §5 with `@DONE`.

## Done-command
scripts/test.sh T-002   # must print PASS 35/35

## Tolerances / perf target
`gemm_bf16` contract tolerance: `atol=1e-2, rtol=2e-2` vs the float64 oracle. This is provisional;
T-004 (NPU backend) may change it with measured evidence. The `ref` backend itself must be
within one bf16 ulp (relative error <= 2^-8) of the oracle. That holds only if it accumulates
in float32 (step "upcast"). No perf target.

## Known pitfalls
- `isinstance(True, int)` is `True`. Reject bools in `TensorSpec.shape` before the int check.
- Frozen dataclass: validate in `__post_init__`; never assign to fields there.
- `torch.allclose` raises on shape mismatch. `Tolerance.allclose` must compare shapes first
  and return `False`.
- Compare dtypes with `t.dtype != DTYPES[spec.dtype]` (torch dtype objects), not strings.
- Shapes: compare `tuple(t.shape)` with a tuple of ints. `torch.Size` equals a tuple, but build
  `output_shapes` as plain tuples.
- Do not compute `gemm_bf16_ref` as `a @ b` in bf16. Whether the CPU bf16 matmul accumulates in
  float32 is `UNVERIFIED`. Upcasting to float32 first makes the one-ulp test independent of it.
- Register into `DEFAULT` only in `gemm_bf16.py`. Tests create fresh `Registry()` objects for
  everything else, so do not make `Registry` a singleton or use class-level dicts
  (`test_registries_are_independent` will fail).
