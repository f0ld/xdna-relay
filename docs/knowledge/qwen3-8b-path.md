# Resume guide: the Qwen3-8B NPU/iGPU path (paused 2026-11-25, ADR 0008)

Everything below is committed on `main` and covered by tests; to resume, read this, run the
listed suites, then pick from "Open".

## What it is
Qwen3-8B Q4_K_M decodes on the iGPU (llama.cpp Vulkan through `native/llama_shim`); prompts
can prefill on the NPU (IRON graph, packed bfp16 weights) and hand their KV to llama.cpp;
a placement policy chooses NPU / iGPU / split per request and calibrates itself online;
decode is batched across requests and speculative with a Qwen3-0.6B draft.

## Run it
```
. scripts/env.sh
python -m xdna_relay.server --gguf ~/.local/share/contio-spike/gguf/Qwen3-8B-Q4_K_M.gguf \
  --draft ~/.cache/xdna-relay/gguf/Qwen3-0.6B-Q8_0.gguf --hf ~/.cache/xdna-relay/hf/Qwen3-8B \
  --npu --packed --preset qwen3-8b-packed --slots 2 --max-tokens 4096 --port 8090
```
(`--no-think` for thinking off by default; calibration persists under
`~/.cache/xdna-relay/calibration/<setup key>.json`; stop with SIGTERM, not SIGINT from a script.)
Operator setup: `tdr_timeout_ms=30000` (amdxdna), NPU `xrt-smi configure --pmode performance`.

## Where things are
| Piece | Code | Tests | Results |
|---|---|---|---|
| IRON graphs (Llama, Qwen3, packed bfp16, two prompt versions) | `backends/npu/models/{qwen3,packed}.py`, `iron_lm.py`, `worker.py` | T-031, T-033, T-022 | 2026-11-25-qwen3-iron.md, -packed-gemm.md, -logits-check.md |
| KV handoff NPU → llama.cpp (per-arch K layout, multi-stream) | `kv/llama_state.py`, `kv/handoff.py` | T-016, T-031, T-034 | |
| Placement + online calibration + persistence | `sched/placement.py`, `online.py`, `calibration_store.py` | T-032, T-038, T-040 | 2026-11-25-placement.md, -calibration.md |
| Batched decode, mixed steps | `backends/igpu/llama_multi.py`, `server/engine.py` | T-035 | 2026-11-25-batched-decode.md |
| Speculative (single stream T-030; batched T-041) | `sched/speculative.py` (`AdaptiveK`, `BatchK`) | T-030, T-041 | 2026-11-24-relay-spec.md, 2026-11-25-batched-spec.md |
| Server (OpenAI API, SSE, timings) | `server/` | T-034 | 2026-11-25-server.md |

Key numbers: NPU prefill 8B packed 1493 tok in 2.6 s; two requests: A keeps 10 vs 2–3 tok/s
while B prefills on the NPU; batched spec throughput +78/+30/+9% at 1/2/4 requests.

## Open (in priority order when resuming)
1. Fairness bound in BatchK (a low-acceptance request loses ~7% at 4 requests).
2. Calibration: battery runs (separate key), long-run drift, Llama 3B via calibration only.
3. Intermittent half-speed plain decode at 4 requests (bench rows now record GPU clock and
   other GPU clients: `bench/gpu_watch.py`).
4. Placement decode weight at near-ties (~279 tokens: iGPU vs split).

## Changes made on the 35B track that also affect the 8B (keep in mind when resuming)
- T-044: a prompt prefilling while nothing decodes takes the whole n_batch (512) per step
  instead of `--prefill-chunk` (`Engine._chunk`); with sequences decoding the chunk is
  unchanged. 8B suites T-035/T-041 pass with it; the 8B's solo-prefill calibration (TTFT fits)
  re-learns online, a stored calibration from before may overestimate iGPU TTFT at first.
- T-044: the server splits `<think>` text into `reasoning_content` (`--reasoning-format none`
  restores the raw text); Qwen3-8B's template opens no `<think>` in the prompt, its replies
  start with `<think>` and are split the same way.
