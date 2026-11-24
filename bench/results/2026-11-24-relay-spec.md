# T-023/T-024/T-030 — our speculative loop on the iGPU (xdna_relay.sched.speculative)

Platform profile `performance`, on AC. Target `Qwen3-8B-Q4_K_M.gguf`, draft
`Qwen3-0.6B-Q8_0.gguf`, both via `LlamaCppExecutor` (native/llama_shim over our patched
llama.cpp 6b0f75b, Vulkan). Greedy, 256 tokens, Qwen3 chat with thinking off.
Tool: `bench/relay_spec.py` (same 4 prompts as `bench/spec_probe.py`).

| config | code | explain | rewrite | story | mean |
|---|---|---|---|---|---|
| plain (ours) | 15.72 | 15.16 | 15.41 | 15.65 | 15.49 |
| llama-server draft n=3 (T-025) | 27.50 | 19.48 | 28.36 | 17.67 | 23.25 |
| ours, fixed k=3 | 29.69 | 21.46 | 30.14 | 19.48 | 25.19 |
| ours, fixed k=5 (earlier run) | 31.16 | 18.71 | 29.80 | 15.72 | 23.85 |
| ours, adaptive (run 1) | 27.75 | 21.02 | 31.31 | 20.18 | 25.07 |
| ours, adaptive (run 2) | 26.81 | 21.07 | 30.84 | 20.26 | 24.75 |

- Our loop is ~8% faster than llama-server's at the same k (less per-round overhead: no
  draft re-processing of the verify batch, no checkpoints for dense models).
- Adaptive ≈ best fixed k on average, without tuning, and avoids the collapse of a wrong fixed
  k (k=8 on prose: −45%). It under-picks k on code (2.1–2.6 where 5 is best): open item.
- History: first adaptive version charged the draft's catch-up to a per-token cost (mean k
  1.2 on explain); second added per-position acceptance (25.35 mean); a regression run then
  collapsed to k≈0.65 on explain because catch-up after plain rounds inflated the first-call
  cost — fixed by learning that cost only from rounds with catch-up <= 2 tokens.
- Divergences from plain greedy: near-ties only (gap 0.03–0.17 logits), see test_t030.
