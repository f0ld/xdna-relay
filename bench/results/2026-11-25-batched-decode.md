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

## Mixed batches: iGPU prefill chunks ride along with decode tokens (T-036)
The stepper now does all work on the shared context: each `llama_decode` carries every
decoding sequence's next token plus the next `--prefill-chunk` tokens of the oldest request
still prefilling (shim `rl_decode_batch` / `rl_argmax_at`; request threads only restore from
the prefix cache / NPU handoff). iGPU-only server, Qwen3-8B, 2 slots, B ~1.7k tokens,
platform `balanced`, **on battery**:
| prefill chunk | A tok/s while B prefills | B TTFT | A done | B done |
|---|---|---|---|---|
| 512 (= before: one 512-token eval per lock hold) | 1.0–1.3 | 4.64–4.82 s | 17.7–18.0 s | 6.9–7.1 s |
| **256 (default)** | 1.6 | 4.86–4.99 s | 17.9 s | 7.1–7.2 s |
| 128 | 2.4–2.5 | 5.77–5.99 s | 18.3–18.4 s | 8.0–8.2 s |
| 64 | 3.6–3.7 | 7.55–7.61 s | 19.2–19.3 s | 9.8–9.9 s |
Interleaving only trades: smaller chunks share the iGPU more fairly, but each step's fixed
cost and the smaller prefill matmuls add work, so everything ends later. 256 keeps B's TTFT
and both totals within a few percent of 512. The way to keep A decoding is the NPU (A 10
tok/s during B's prefill, above).

## Placement recalibrated for the batched engine (T-037)
`bench/server_load.py --b-words 30,200,900`, server batched (chunk 256), `--placement igpu`
vs `npu`, Qwen3-8B packed, on battery, A alone ~15.5 tok/s; no NPU TDR.
| B tokens | iGPU: B TTFT / A during | NPU: B TTFT / A during |
|---|---|---|
| 85 | 0.37–0.43 s / 4.7–5.4 | 0.94–0.95 s / 9.5–10.7 |
| 401 | 1.41 s / 2.1 | 1.55–1.57 s / 9.6–9.7 |
| 1721 | 5.43–5.44 s / 1.5 | 4.55–4.69 s / 10.1–10.2 |
Fit (preset `qwen3-8b-packed-batched`; the server picks a preset's `-batched` variant when
batched): iGPU 1.240x alone + 0.157 s, keep 0.067; NPU 1.488x + 0.082 s, keep 0.643 (all
TTFT fits within 4%). Compared with the unbatched engine, an iGPU prefill waits less
(0.16 vs 0.53 s) but stalls A harder (keeps 7% vs 20%).

Decisions: 85 tokens iGPU; ~180+ NPU; just past a 256-row block **split** (NPU whole blocks,
the iGPU the tail inside a mixed step). Split validated at 282 tokens (`--placement split`):
| placement | B TTFT | A during | A done | TTFT + A lost (measured / estimated) |
|---|---|---|---|---|
| igpu | 1.09–1.11 s | 1.8–2.7 | 14.1–14.4 s | 2.04 / 1.83 |
| npu | 1.48–1.53 s | 9.5–9.8 | 14.4 s | 2.07 / 2.14 |
| **split** (256 NPU + 26 iGPU) | 1.21–1.37 s | 8.7–10.2 | **13.7–14.0 s** | **1.81 / 1.52** |
| auto (chose split) | 1.21–1.32 s | 9.1 | 14.0 s | |
First measurement where split wins: in the batched engine the iGPU tail is cheap (it rides in
a mixed step). The policy ranks the options as measured; it is optimistic about split by ~0.3.
