# T-035: batched decode across requests (2026-11-25)

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

## NPU handoff into a per-sequence-KV context (fixed after resume)
Per-sequence KV makes a sequence's state multi-stream (one stream per buffer; only the
sequence's own holds cells; llama.cpp restores the cells into the destination's stream and
requires the stream count to match). `llama_state` handles it; the failure was the handoff's
template cache, keyed by model name only, reusing a 1-stream template (from a 1-slot engine
in the same process) for a 2-stream context. Now keyed by (model, KV streams).
`test_npu_handoff_into_second_slot` passes (T-034 10/10).

## Two requests through the batched server (`bench/server_load.py`), Qwen3-8B
2 slots, NPU packed, preset `qwen3-8b-packed`. Platform `balanced`, **on battery**, NPU
pmode `performance`; no NPU TDR. A streams 200 tokens; B (~1.7k-token prompt) arrives 1 s in.
| placement | A tok/s while B waits | A overall | B TTFT | B placed | A done | B done |
|---|---|---|---|---|---|---|
| `igpu` | 0.9 | 11.5–11.6 | 4.42–4.50 s | iGPU | 17.5 s | 6.7–6.8 s |
| `auto` | **10.0** | **13.6** | 4.40–4.49 s | NPU | **14.9 s** | 6.7–6.8 s |
Unbatched server, for comparison (2026-11-25-server.md, on AC): iGPU B TTFT 5.6–6.0 s, A
1.8–2.7 during; NPU B TTFT 4.7–4.9 s, A 10.2.
- In the shared context B's iGPU prefill runs in 512-token chunks under the context lock: B's
  TTFT drops to the NPU's, but A gets ~1 step per chunk (0.9 tok/s). With the NPU, A keeps
  10 tok/s and finishes 2.6 s (15%) sooner at the same TTFT for B.
- The placement presets were fitted on the unbatched engine; on the iGPU side they now
  overestimate B's TTFT and underestimate A's loss. Same decision here; recalibrate for the
  batched engine (follow-up).
