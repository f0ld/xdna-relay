# T-027 — prefix cache, TTFT on Qwen3-8B (iGPU, plain decode)

Platform profile `performance`, on AC. `bench/prefix_cache_bench.py`, Qwen3-8B-Q4_K_M,
system prompt + ~2.4k-token context, 48 generated tokens per request.

| scenario | cold TTFT | cached TTFT (snapshot after generation) |
|---|---|---|
| A multi-turn, turn 1 (first) | 5.9–6.4 s | 6.3 s (nothing to reuse) |
| A multi-turn, turns 2–4 | 6.3–6.5 s | 0.22–0.34 s (58–72 new tokens evaluated) |
| B shared system prompt, other conversation in between | 6.2–6.3 s | 0.09–0.19 s (restore ~20 ms + 1–19 new tokens) |

- Snapshot size ~147 KB/token (Qwen3-8B, f16 KV): ~350 MB at 2.4k tokens; take ~47 ms,
  restore ~20 ms. Taking it after generation keeps it off TTFT and includes the answer.
- Multi-turn reuse stops ~4 tokens before the previous prompt's end: Qwen3's history drops
  the empty `<think>` block, so the re-rendered prompt diverges there (template, not cache).
- Memory: 7 entries = 2.5 GB; default budget 4 GB (LRU by bytes). Prefix-redundant
  entries are dropped on insert.
