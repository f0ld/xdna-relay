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

- **`PYTHONPATH=src python ...` after `. scripts/env.sh` → "CPUOnlyTensor is in use: pyxrt is not
  importable"** → env.sh puts pyxrt on `PYTHONPATH` (`/opt/xilinx/xrt/python`); overriding it drops
  it. Use `PYTHONPATH=src:$PYTHONPATH`, or `scripts/test.sh`.

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
- **IRON `GEMM` at `Precision.MEDIUM` misses the `gemm_bf16` contract** (abs err up to 0.13 at
  |C|≈5, K≤4096; measured 2026-11-11). bfp16 needs its own contract/tolerance, not `gemm_bf16`.
- **torch bf16 ↔ numpy**: `Tensor.numpy()` rejects bf16. Reinterpret bits:
  `t.contiguous().view(torch.int16).numpy().view(ml_dtypes.bfloat16)` and back with
  `torch.from_numpy(a.view(np.int16).copy()).view(torch.bfloat16)`.

## Formats (GGUF, Q4NX)

- **GGUF dims are reversed**: gguf-py's `ReaderTensor.shape` is `[in, out]` (numpy uint64);
  `.data` of a quantized tensor is `uint8 (out, row_bytes)`; `gguf.quants.dequantize` returns
  `(out, in)` float32 (= PyTorch `Linear.weight` layout). Verified 2026-11-11, gguf 0.19.0.
- **"Q4_K_M" GGUF files are a mix**: Qwen3.5-9B-Q4_K_M by params = Q4_K 66%, Q6_K 21%,
  Q5_K 14%, Q8_0 0.1%. Only Q4_0/Q4_K go to Q4NX (ADR 0005); the rest take the bf16 fallback.
- **gguf-py cannot quantize Q4_K/Q5_K/Q6_K** (`NotImplementedError`). Tests use random raw blocks
  with finite f16 scales (`tests/reference/gguf_files.py`).
