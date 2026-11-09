# T-001 — Port upstream vector_scalar_mul and run it on the NPU
Status: TODO
Milestone: M0
Depends on: — (bootstrap done: `.venv` installed, `scripts/env-check.sh` → ENV PASS)

## Goal
Prove the toolchain end to end: an `@iron.jit` design compiles with Peano/aiecc from our
`.venv`, runs on the NPU, and matches a numpy reference. Port the upstream mlir-aie example
`vector_scalar_mul` into our package behind a plain numpy function.

## Files you may change
- `src/xdna_relay/examples/vector_scalar_mul.py` (create)
- `docs/knowledge/pitfalls.md` (append only, if you hit something new)
- `PLAN.md` (only the T-001 status line), `handoff/T-001-iN.md`

## Files to read first
- `docs/knowledge/toolchain.md` — section "Smoke example: vector_scalar_mul" (has the full design code)
- `tests/kernels/test_t001_vector_scalar_mul.py` — the tests you must pass (do not edit)
- `docs/knowledge/pitfalls.md`

## Interface / contract
In `src/xdna_relay/examples/vector_scalar_mul.py`:

```python
def vector_scalar_mul(a: np.ndarray, factor: int) -> np.ndarray:
    """Return a * factor computed on the NPU, same shape and dtype as `a`.

    a: 1-D np.int16 or np.int32 array. a.nbytes must be a multiple of 128 and >= 1024.
    factor: Python int (sent to the NPU as one int32).
    Raises ValueError for any other dtype, ndim != 1, or bad size (checked BEFORE touching the NPU).
    """
```

Internally keep the upstream `@iron.jit` design as a private module-level function, e.g.
`_vsm_design(A: In, C: Out, F: In, *, in1_size: CompileTime[int], int_bit_width: CompileTime[int] = 16)`.
Use only these symbols (all verified in the installed wheel):
`aie.iron` (`jit`, `tensor`, `zeros`, `full`), `aie.iron.CompileTime`, `aie.iron.In`,
`aie.iron.Out`, `aie.iron.kernels.scale`, `aie.iron.algorithms.transform`.

## Steps
1. `git switch -c task/T-001-vector-scalar-mul main`; set T-001 to `IN PROGRESS` in `PLAN.md`.
2. Run `scripts/env-check.sh --no-flm`. If not `ENV PASS`, stop → `@ESCALATE`.
3. Run `scripts/test.sh T-001`. Expect `FAIL 1/9` (module missing). This is the baseline.
4. Create `src/xdna_relay/examples/vector_scalar_mul.py`:
   a. Copy the design function from toolchain.md; drop `trace_size` / `use_chess` / `vectorized`
      params (use `vectorized=True`, `use_chess=False`, `trace_size=0` as fixed values).
   b. Write `vector_scalar_mul(a, factor)`: validate input (dtype in {int16, int32},
      `a.ndim == 1`, `a.nbytes % 128 == 0`, `a.nbytes >= 1024`) and raise `ValueError` otherwise.
   c. Build NPU tensors: `iron.tensor(a, dtype=a.dtype, device="npu")`,
      `iron.full((1,), int(factor), dtype=np.int32, device="npu")`,
      `iron.zeros(a.size, dtype=a.dtype, device="npu")`.
   d. Call the design with **argument order (A, C, F)** and
      `in1_size=a.nbytes, int_bit_width=8 * a.itemsize`.
   e. Return `c_t.numpy().copy()`.
5. Run `scripts/test.sh T-001`. The first NPU run compiles (may take a minute); later runs use
   the cache in `~/.npu/cache`.
6. Must print `PASS 9/9`. Then do the end-of-iteration steps in AGENTS.md §5 with `@DONE`.

## Done-command
scripts/test.sh T-001   # must print PASS 9/9

## Tolerances / perf target
Exact equality (integers). No perf target.

## Known pitfalls
- `in1_size` is in **bytes**, not elements.
- Argument order of the design is `(A, C, F)` — output before the factor.
- `pyxrt` import errors mean XRT is not sourced: always use `scripts/test.sh`.
- If the run hangs or prints `ERT_CMD_STATE_TIMEOUT`: stop, retry once, then `@ESCALATE` with
  the command and `journalctl -k -n 20` output. Do not kill other processes.
- If `env-check` warns `npu-in-use`, mention it in the handoff (contio/FLM may hold the NPU).
- Compile errors mentioning a missing symbol: do not guess another name; grep
  `.venv/lib64/python3.14/site-packages/mlir_aie/python/aie/iron/` and escalate if absent.
