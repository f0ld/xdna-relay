# T-041 — speculative decoding in the batched engine

On AC, platform profile `balanced`, iGPU 890M (Vulkan, our llama.cpp 6b0f75b). Target
Qwen3-8B Q4_K_M, draft Qwen3-0.6B Q8_0, `Engine(..., draft_gguf=...)` (batched; before T-041
a draft model forced unbatched slots). Greedy, 128 tokens, prompts code / explain / rewrite /
story (spec_probe), prefilled first (prefix cache). Tool: `bench/batched_spec.py`.
`k=0` = the same engine never drafting (= the plain batched engine: same per-sequence KV).

| concurrent | throughput all active, k=0 | adaptive | slowest request k=0 | adaptive | mean k | acceptance |
|---|---|---|---|---|---|---|
| 1 | 15.4 | **27.4** (+78%) | 15.4 | 27.4 | 2.43 | 0.72 |
| 2 | 29.5 | **38.3** (+30%) | 14.7 | 17.5 | 1.00 | 0.68 |
| 4 | 54.1 | **58.8** (+9%) | 13.5 | 12.5 (−7%) | 0.93 | 0.69 |

Throughput "all active" = tokens of all requests while every one is decoding / that time
(the server's rate at that concurrency). At 4, the story request (low acceptance) shares
longer rounds with requests that draft well: it loses ~7% while the batch gains 9%. A
fairness bound (no sequence slower than plain by more than x%) would be a policy knob; not
added.

## What it took (each measured; pitfalls.md has the facts)

1. **Verify cost has a cliff at 8 tokens per batch** (Qwen3-8B Q4_K_M): 1–4 tokens 65–80 ms,
   8 tokens ~95 ms, 9+ tokens ~165–190 ms (presumably llama.cpp Vulkan's matrix-vector path
   covering up to 8 columns: UNVERIFIED). Linear interpolation between a few online
   measurements drafted across it (4 requests, k=3: 37 vs 52 tok/s). Now the engine sweeps
   both models' batch costs at start (~5 s: 1–12, 16, 24, 32, slots×(k_max+1), chunk sizes).
2. **Per-sequence KV buffers split uneven batches**: llama.cpp runs [2,1,1,1] as two passes
   over the weights (140 ms vs 72 ms for [1,1,1,1]; [3,3,1,1] 147 ms). A unified KV buffer
   runs them in one pass (81 / 97 ms) but slows every decode step (each sequence attends over
   all sequences' cells: 4 requests plain 46 vs 53 tok/s). Kept per-sequence KV for the
   target; `BatchK(split_ragged=True)` costs one pass per distinct token count, so the
   policy prefers even shapes (all decoders draft the same k, or none). The draft context
   (0.6B, attention cheap, uneven every round) uses a unified buffer.
3. **llama_decode without outputs returns early**: a draft catch-up with no logits took
   1.5 ms and its ~12 ms of GPU work landed in the next verify. Catch-up parts now request a
   logit (synchronous; costs attributed correctly).
4. **Outcome check**: the component model still misses costs (Python bookkeeping, rounds
   after requests finish). `BatchK` keeps measured tokens/s per batch size with and without
   drafting and does not draft where drafting measured slower; every 16th round takes the
   other arm.

Draft call overhead is the remaining limit: a 0.6B step costs 11–16 ms (mostly fixed), a
fifth of an 8B step, so with several sequences k stays at 1–2.

## Open

- Plain batched rounds at 4 requests intermittently took ~137 ms instead of ~75 ms (3 of ~9
  runs, whole runs; clocks normal in a sampled rerun). Unexplained.
- T-036 measured mixed prefill+decode steps on per-sequence KV, where a riding chunk is a
  second pass; worth re-measuring with that in mind.
- NPU prefill on this prompt flips a 0.02-logit near-tie at token 5 even without drafting
  (T-031's "identical tokens" held for its prompts only); T-041's NPU test uses the near-tie
  rule.
