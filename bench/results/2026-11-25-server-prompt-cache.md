# T-021: llama-server prompt cache baseline vs our prefix cache (2026-11-25)

`bench/server_prompt_cache.py`: the T-027 scenarios (same prompts, Qwen3-8B-Q4_K_M,
~2.4k-token system prompt + context, 48 generated tokens) against llama-server 6b0f75b
(Vulkan, `-fa on -c 8192`). TTFT = wall clock to the first streamed token. Platform
`balanced`, on AC. Our numbers: bench/results/2026-11-24-prefix-cache.md (profile
`performance`; cold 5.9–6.5 s there vs 6.1–7.2 s here).

| scenario | cold (`--cache-ram 0`, no `cache_prompt`) | llama-server default (1 slot, `cache_prompt`, `--cache-ram 8192`) | `-np 4` | ours (T-027) |
|---|---|---|---|---|
| A multi-turn, turns 2–4 | 6.28–7.19 s | 0.24–0.32 s (58–72 tok evaluated) | 0.24–0.33 s | 0.22–0.34 s |
| B shared system prompt, other conversation between | 6.84–6.87 s | 0.09–0.23 s (4–19 tok evaluated, 2451–2461 restored) | 0.10–0.24 s | 0.09–0.19 s |

- Parity: llama-server's slot reuse + host-RAM prompt cache (llama.cpp PR 16391) gives the
  same TTFT as our prefix cache on the iGPU alone; it also restores the shared system prompt
  after an unrelated conversation evicted the slot.
- Both reuse the same prefixes (same `cache_n`; multi-turn reuse stops where Qwen3's template
  drops the empty think block in the history).
- So our prefix cache earns its place only through what llama-server does not do: it lives
  in the scheduler, next to NPU prefill and KV handoff (a cached prefix shortens the NPU's or
  the iGPU's prefill; `PlacementPolicy.choose(n, n_npu=...)`), and its snapshots are plain
  llama.cpp state blobs the handoff already speaks.
