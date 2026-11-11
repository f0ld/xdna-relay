# Pitfalls

Filled during reviews. One entry per pitfall: **symptom** → cause → fix. Implementer may append;
Architect curates.

## Environment

- **`ModuleNotFoundError: No module named 'pyxrt'`** → XRT not sourced. Run through
  `scripts/test.sh` / `scripts/bench.sh`, or `. scripts/env.sh` first.
- **`ERT_CMD_STATE_TIMEOUT` on long kernels** → driver `tdr_timeout_ms` default (~4 s budget).
  Operator TODO in environment.md. Do not retry in a loop; escalate.
- **NPU busy / open fails** → contio's FastFlowLM backend or `flm serve` holds the NPU.
  `scripts/env-check.sh` shows `npu-in-use`. Ask the Operator; never kill processes.

## Kernels

- **Integer overflow semantics of `kernels.scale`** (`to_vector<T>(0)` from an acc32/acc64):
  wrap vs saturate depends on the AIE saturation mode — `UNVERIFIED`. Tests keep products in
  range.
- **`in1_size` in vector_scalar_mul is bytes, not elements.**

## Testing

- **`scripts/test.sh T-XXX -k "<expr>"` selects nothing or too much** → pytest `-k` matches the
  full node id, including the file name (`test_t002_op_registry.py` contains `registr`, `op`, …).
  Filter by words that only appear in test function names.

## Op layer (`src/xdna_relay/ops/`)

- **New op = new module + one import line**: define the `OpContract` and the `ref` impl in
  `ops/<op>.py`, register into `registry.DEFAULT` at module level, and add
  `from . import <op>` to `ops/__init__.py`. Tests use fresh `Registry()` objects for anything
  else.
- **`run()` errors**: bad inputs/params → `ValueError`; a backend returning the wrong
  count/dtype/shape → `RuntimeError`; unknown op/backend → `KeyError`.
