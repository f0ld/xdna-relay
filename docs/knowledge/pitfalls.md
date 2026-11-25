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
- **Two Q4NX block orders in IRON**: `DequantBFP` reads column blocks `(N/64, K/512, k-half,
  n-half)`. `flm/layer/reference.py::pack_q4nx` writes `(N/64, K/256, n-half)` for the fused
  layer kernel. Our blob (`formats/q4nx.py`) follows `DequantBFP`.
- **bf16 rounding bound**: bf16 has 8 significant bits; RNE moves a value by ≤ 2^-8 of itself
  (not 2^-9: the Architect got this wrong once in a test draft).

## IRON device loop needs a newer amdxdna (2026-11-24)
`Session.draw` / `Model.stream` with a full-ELF decode step run IRON's device-side sampling
loop, which aliases a parameter scratchpad as a user-pointer BO
(`mlir_aie/.../parameter_scratchpad.py: ParameterScratchpad.alias`). The in-kernel amdxdna
on kernel 7.3.4 refuses it: `DRM_IOCTL_AMDXDNA_CREATE_BO IOCTL failed (err=-14): Bad address`
(docstring: needs amdxdna >= 2.27, xdna-driver 72e64a6). So `python -m
iron.applications.llama.model ...` fails here. Workaround: drive steps from the host with
`model.session.logits(tokens)` (`bench/iron_llama_probe.py`) — works with full ELF.

## KV handoff IRON -> llama.cpp: K channel order (2026-11-24)
IRON's Llama uses Hugging Face weights and RoPE in the two-halves layout (pairs j, j+D/2;
`iron/operators/rope.py` default `two_halves=True`). GGUF conversion permutes q/k weights to
the interleaved layout (pairs 2j, 2j+1). Cached K rows therefore differ in channel order:
`kv/llama_state.halves_to_interleaved` before import. Without it K is 140% off and decode
produces garbage; with it K matches llama.cpp's own cache to 0.4–1.4% (bf16). V needs no
permutation. IRON KV is readable via `model.prompt_chunk.read(layer.cache.keys)` → `(rows, kv_heads, head_dim)`.

## Long IRON prompt chunks hit the driver command timeout (2026-11-24)
Llama 3.2 3B, one 1493-row prompt chunk (`IronLmExecutor.eval`) → `RuntimeError: Llama: the run
ended in ert_cmd_state.ERT_CMD_STATE_TIMEOUT`; kernel log `amdxdna ... aie2_set_cmd_timeout`.
Default tdr timeout (~2 s) < chunk time. 1B at 1494 rows and 3B at 346 rows are fine.
Fix (applied 2026-11-24): `options amdxdna tdr_timeout_ms=30000` in /etc/modprobe.d/amdxdna-tdr.conf + module reload; 3B at 1494 rows then runs. The NPU recovers on its own.

## In-process IRON starves other threads (GIL) (2026-11-24)
IRON's host loop keeps the GIL while the NPU runs; an iGPU decode thread in the same process
drops from 29 to ~6 tok/s. Run the NPU executor in its own process
(`backends/npu/worker.py: NpuWorker`).

## IRON autotune: standalone gains do not transfer to the graph (2026-11-24)
`bench/iron_tune.py` (IRON's `autotune`, timed at rows=2048) chose MHA `B_q=32, pv_cores=2`
for Llama 3B (−4.75 ms/call standalone); in the real prompt graph at 1494 rows the chunk went
1.61 s → 4.74 s. Cause UNVERIFIED (rows/kv_len bound differs from the timed point?). Always
A/B a tuned profile with a real eval before keeping it. IRON's shipped llama `npu2.json`
already has 3B entries.
`iron.common.declare.Profile.save` raises `TypeError: ... TensorAccessPattern is not JSON
serializable` for profiles holding Copy entries (also GEMM's b_dtype/epilogue objects): the
tuner's result is lost unless those entries are filtered out first.

## KV handoff: K layout depends on the architecture (2026-11-25)
llama.cpp caches **Llama's** K in interleaved RoPE pairs (GGUF conversion permutes q/k),
but **Qwen's** in Hugging Face's two halves (NEOX RoPE, no permutation). Permuting Qwen3's K
gives K rel.err 1.41 and garbage tokens. `IronLmExecutor.llama_cpp_k_interleaved` (from
`config.json` `model_type`) decides it; handoff and worker follow it.
