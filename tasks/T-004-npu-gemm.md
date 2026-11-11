# T-004 — `npu` backend for `gemm_bf16` via IRON `GEMM`
Status: DONE
Milestone: M1
Depends on: T-002 (DONE)

## Goal
Register a second backend, `"npu"`, for the existing `gemm_bf16` contract. It runs IRON's
`GEMM` operator on the NPU. Arbitrary `(M, K, N)` are supported by zero-padding every
dimension up to a multiple of 64 and slicing the result. This is the first op that runs
behind our contract on hardware. Correctness only; speed is M2 (T-006..T-008).

## Files you may change
- `src/xdna_relay/backends/__init__.py` (create)
- `src/xdna_relay/backends/npu/__init__.py` (create)
- `src/xdna_relay/backends/npu/gemm_bf16.py` (create)
- `docs/knowledge/pitfalls.md` (append only, if you hit something new)
- `PLAN.md` (only the T-004 status line), `handoff/T-004-iN.md`

Do **not** touch `src/xdna_relay/ops/`. The contract and tolerance stay as they are.

## Files to read first
- `tests/kernels/test_t004_npu_gemm_bf16.py`: the tests you must pass (do not edit)
- `src/xdna_relay/ops/gemm_bf16.py`, `src/xdna_relay/ops/registry.py`: what you register into
- `docs/knowledge/pitfalls.md`: sections "Testing" and "Op layer"

## Verified API (installed wheel; all symbols below were run on this machine)
- `from iron.operators.gemm import GEMM` (`.venv/lib64/python3.14/site-packages/iron/operators/gemm.py`)
- `from iron.common import Precision` (enum: `HIGHEST`, `HIGH`, `MEDIUM`). Use **`Precision.HIGHEST`** only.
- `GEMM(M=..., K=..., N=..., precision=Precision.HIGHEST)` builds an operator. Leave every other
  argument at its default (tiles, columns and placement resolve automatically).
  Its buffers, in call order: `A (M, K)`, `B (K, N)`, `C (M, N)`, all `ml_dtypes.bfloat16`,
  row-major.
- `op(A_dev, B_dev, C_dev)` compiles on first use (a few seconds; disk-cached), then runs.
  Arguments must be device tensors.
- Device tensors (same module as T-001): `import aie.iron as iron`;
  `iron.tensor(np_array, dtype=bfloat16, device="npu")`;
  `iron.zeros((rows, cols), dtype=bfloat16, device="npu")`; read back with `.numpy()`.
- `from ml_dtypes import bfloat16`: numpy's bf16 dtype. torch and numpy cannot convert
  bf16 directly; reinterpret the bits through int16 (see interface).

## Interface / contract

### `src/xdna_relay/backends/__init__.py`
Docstring only: `"""Backend implementations of the op contracts (one subpackage per device)."""`

### `src/xdna_relay/backends/npu/__init__.py`
Docstring plus `from . import gemm_bf16  # noqa: F401`. Importing `xdna_relay.backends.npu`
registers every NPU backend.

### `src/xdna_relay/backends/npu/gemm_bf16.py`

```python
PAD = 64   # every padded dim is a multiple of this (GEMM's default tile_k/tile_n; covers tile_m * rows)

def padded_shape(m: int, k: int, n: int) -> tuple[int, int, int]:
    """Each of m, k, n rounded UP to a multiple of PAD. padded_shape(100, 300, 70) == (128, 320, 128)."""

@functools.cache
def gemm_op(m_pad: int, k_pad: int, n_pad: int) -> GEMM:
    """One IRON operator per padded shape: GEMM(M=m_pad, K=k_pad, N=n_pad, precision=Precision.HIGHEST).
    functools.cache makes repeated calls return the same object (no re-compile)."""

def to_numpy_bf16(t: torch.Tensor) -> np.ndarray:
    """torch bf16 -> numpy ml_dtypes.bfloat16, same shape, bit-exact:
    t.contiguous().view(torch.int16).numpy().view(bfloat16)"""

def to_torch_bf16(a: np.ndarray) -> torch.Tensor:
    """numpy ml_dtypes.bfloat16 -> torch bf16, same shape, bit-exact, owns its memory:
    torch.from_numpy(np.ascontiguousarray(a).view(np.int16).copy()).view(torch.bfloat16)"""

def gemm_bf16_npu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = A @ B on the NPU. a: (M, K) bf16, b: (K, N) bf16 (any strides). Returns (M, N) bf16,
    contiguous. ValueError if M, K or N is 0 (checked before any NPU call).
    Shapes and dtypes are already checked by Registry.run; do not re-check them."""

# at module level, after the definitions:
import xdna_relay.ops  # noqa: F401   (ensures the gemm_bf16 contract is in DEFAULT; put with the other imports)
DEFAULT.register("gemm_bf16", "npu", gemm_bf16_npu)
```

`gemm_bf16_npu` body, in this order:
1. `m, k = a.shape`; `n = b.shape[1]`; ValueError if any is 0.
2. `mp, kp, np_ = padded_shape(m, k, n)`. Do not name a variable `np`: it shadows numpy.
3. `ap = torch.zeros(mp, kp, dtype=torch.bfloat16)`; `ap[:m, :k] = a`. Same for `bp` (kp, np_).
   Zero padding is exact: padded K columns/rows contribute 0, and padded M rows/N columns are
   sliced away.
4. `A = iron.tensor(to_numpy_bf16(ap), dtype=bfloat16, device="npu")`, the same for `B`;
   `C = iron.zeros((mp, np_), dtype=bfloat16, device="npu")`.
5. `gemm_op(mp, kp, np_)(A, B, C)`.
6. `return to_torch_bf16(C.numpy().reshape(mp, np_))[:m, :n].contiguous()`.

Allocate new device tensors on every call (no buffer reuse yet; M2 handles that).

## Steps
1. `git switch -c task/T-004-npu-gemm main`; set T-004 to `IN PROGRESS` in `PLAN.md`.
2. `scripts/env-check.sh --no-flm`. If not `ENV PASS`, stop → `@ESCALATE`.
3. `scripts/test.sh T-004`. Expect `FAIL 1/15`. The one passing test is
   `test_ops_import_stays_npu_free`, which must still pass at the end. This is the baseline.
4. Create the three files.
5. `scripts/test.sh T-004 -k "padded or conversion or registered or gemm_op or empty or stays"`
   → must print `PASS 6/6` (no kernel runs yet; constructing `GEMM` does not compile).
6. `scripts/test.sh T-004` → must print `PASS 15/15`. The first run compiles several shapes
   (about 5 s each). Then also run `scripts/test.sh T-002`, which must still print `PASS 35/35`.
7. End-of-iteration steps in AGENTS.md §5 with `@DONE`.

## Done-command
scripts/test.sh T-004   # must print PASS 15/15

## Tolerances / perf target
The tests check three things, all against the float64 oracle (`tests/reference/gemm.py`):
- the `gemm_bf16` contract tolerance (`atol=1e-2, rtol=2e-2`);
- IRON's own `HIGHEST` bound: `|C - oracle| <= 2 * 2^-8 * (|A| @ |B|)` elementwise;
- vs the `ref` backend: at most 1 bf16 ulp + f32 accumulation-order error, on < 1% of outputs.
Architect measurement (2026-11-11): `HIGHEST` passes all three with margin; `MEDIUM` fails
all three. No perf target.

## Known pitfalls
- **Use `Precision.HIGHEST`.** `MEDIUM` rounds inputs to bfp16 and fails the tests (measured);
  `HIGH` rounds partial sums per K tile.
- **The `-k` filter matches the file name too.** `test_t004_npu_gemm_bf16.py` contains `npu`,
  `gemm`, `bf16`, so `-k npu` selects everything. Use the exact expression from step 5.
- **Do not import `iron`/`aie` from `src/xdna_relay/ops/`.** `test_ops_import_stays_npu_free` checks
  that `import xdna_relay.ops` loads no `aie`, `iron` or `pyxrt` module. Only `backends/npu/`
  imports them.
- **Ad-hoc runs:** `. scripts/env.sh` sets `PYTHONPATH=/opt/xilinx/xrt/python` (pyxrt).
  `PYTHONPATH=src python ...` overwrites it and fails with "CPUOnlyTensor is in use: pyxrt is not
  importable". Use `PYTHONPATH=src:$PYTHONPATH`, or just `scripts/test.sh`.
- `torch.Tensor.numpy()` fails on bf16. Always go through `to_numpy_bf16` / `to_torch_bf16`.
- Inputs can be non-contiguous views (the tests pass a transposed view). `ap[:m, :k] = a`
  handles any strides; do not call `.numpy()` on `a` directly.
- Do not modify `a` or `b` in place (tested).
- If a run hangs or prints `ERT_CMD_STATE_TIMEOUT`: stop, retry once, then `@ESCALATE` with the
  command and `journalctl -k -n 20`. The largest test shape (512×4096×2048) compiled and ran in
  about 2 s on this machine.
