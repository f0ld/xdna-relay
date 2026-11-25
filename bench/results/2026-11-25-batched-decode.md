# T-035: batched decode across requests (2026-11-25) — IN PROGRESS

One llama.cpp context, several sequences (`backends/igpu/llama_multi.py`, shim
`rl_load_multi` / `rl_decode_multi` / `*_seq`), one stepper thread in the server engine.
Platform `balanced`, on AC.

## Decode throughput, Qwen3-8B Q4_K_M
| sequences | llama-batched-bench total | ours (SharedLlamaContext) total | separate contexts (old engine) |
|---|---|---|---|
| 1 | 15.3 | 15.6 | 15.8 |
| 2 | 30.2 | 30.3 | 16.0 (8.0 each) |
| 4 | 57.5 | 56.6 | — |
Through the HTTP server (`bench/server_concurrency.py`, 4 slots, client-side):
| clients | batched per client / total | unbatched per client / total |
|---|---|---|
| 1 | 15.5 / 15.6 | 15.6 / 15.8 |
| 2 | 14.8 / 30.1 | 8.0 / 16.0 |
| 4 | 13.7 / 56.3 | 4.0 / 15.9 |

## Determinism
Batch size (2 vs 4 sequences: other kernels) and KV cell placement flip near-ties only:
every divergence measured sat at a top-2 gap of 0.002–0.07 logits (Qwen3-0.6B). One KV
buffer per sequence (`kv_unified=false`) is as fast as unified here and makes same-path
reruns repeatable on 8B (0/4 vs 1/4 differing); default.

## Open (stopped for suspend, 11:3x)
- Per-sequence KV makes sequence state multi-stream (one stream per buffer). `llama_state`
  now parses/builds that (T-035 round-trip test passes), but restoring NPU-handed-off KV into
  sequence 1 of a 2-slot server fails ("state restore failed",
  `test_t034_server.py::test_npu_handoff_into_second_slot`). Suspect: the handoff template
  is cached per model name from slot 0's snapshot (stream 0, seq_id 0) and llama.cpp's
  `state_read_meta` checks it against the destination. Next step: template per sequence, or
  write the destination's stream/seq_id.
- Then: the two-request NPU scenario through the batched server (bench/server_load.py).
