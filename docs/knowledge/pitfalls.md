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

## IRON NPU prefill re-reads weights per 256-row unit (2026-11-25)
`flm.ProjectionGEMM` streams all of B from DDR for every 256-row unit a call reaches, so
prompt chunks are weight-bandwidth bound (~22 GB/s) and time steps every 256 rows.
`m_chunk > 1` is refused under a per-call row bound. Pre-packed bfp16 B (`bfp16_b=True`,
`models/packed.py`) halves the bytes. A packed B can't be an `iron.param` (numpy has no
block-float dtype): hold it as an `iron.State(..., v8bfp16ebs8)` and write its bytes via the
compiled graph's storage (`CompiledGraph.write` slices by block count, not bytes). mlir-aie's
`Device.dma_bd_fits` raises `TypeError: Converting np.generic to a dtype` for large bfp
transfers: patched in `models/packed.py`. `load_versions(release=...)` frees host weights:
pack before or skip the release.

## NPU TDRs can pass silently (2026-11-25)
Three amdxdna TDR timeouts (08:16, 09:26, 09:28) during IRON per-op timing runs left no error
in the scripts. Benches now call `bench/tdr_guard.guard()`, which prints any
`aie2_tdr_detect` kernel lines at exit ("results suspect"). Check that line before using a
number.

## Online placement calibration: prior strength vs bias (2026-11-25)
`sched/online.LinFit` pulls toward the preset (ridge, `tau`) so data at a single prompt length
cannot make the fit degenerate. With `tau=2, forget=0.95` (~20 effective observations) the
intercept stayed 0.07 s off a true 0.5 s after hundreds of observations: prefill intercepts are
0.1–0.5 s, so that bias decides placements. Now `tau=1, forget=0.97` (T-038 test).

## Background jobs ignore SIGINT (2026-11-25)
A process started with `&` from a non-interactive shell (scripts, `( ... ) &`) inherits SIGINT
as ignored, and Python then never raises KeyboardInterrupt: `kill -INT` does nothing and a
script that waits for the server hangs forever (cost a 2 h stall in the T-040 check). The
server now reinstalls `signal.default_int_handler`; in scripts prefer `kill -TERM`. Also:
`pkill -f '<pattern>'` from a shell whose own command line contains the pattern kills that shell.

## llama.cpp batch shapes on the 890M (T-041, 2026-11-25)
- Qwen3-8B Q4_K_M batch cost has a cliff after 8 tokens per `llama_decode`: ~95 ms at 8,
  ~165–190 ms from 9 (likely the Vulkan matrix-vector path's column limit; UNVERIFIED).
  Never interpolate batch costs across it: measure them (engine start-up sweep).
- With per-sequence KV (`kv_unified=False`), a batch whose sequences get different token
  counts runs as one pass per distinct count ([2,1,1,1] ≈ 2× [1,1,1,1]). A unified buffer
  runs it in one pass but makes every sequence attend over all sequences' cells.
- `llama_decode` with no logits requested returns before the GPU finishes; the work shows up
  in the next call's time. Ask for one logit when timing matters.

## NpuWorker: a dead worker hung the parent (2026-11-25)
The parent kept the child's end of the `Pipe` open, so `recv()` never saw EOF when the worker
died (e.g. a script without an `if __name__ == "__main__":` guard: spawn re-runs it). The
parent now closes it after `start()` and raises "NPU worker exited".
