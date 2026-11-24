# T-028: IRON prompt-chunk profile, Llama 3.2 3B (2026-11-24)

Machine: Ryzen AI 9 HX 370, platform profile `balanced`, on AC, amdxdna with
`tdr_timeout_ms=30000`. Per-operator times from `bench/iron_op_profile.py`: each distinct
operator design timed alone with IRON's harness (`iron.tune.measure.measure`, dispatch
cancelled), at the per-call values of a 1494-token prompt, times its count in the graph.
No iGPU load (standalone figures; under concurrent iGPU decode the NPU eval is ~25% slower).

## Where the time goes (one 2048-row chunk, 1494 valid rows)
| operator | high (IRON's default for 3B, bf16 macs) | medium (bfp16 macs) |
|---|---|---|
| FFN gate+up GEMM 3072→8192 (×56) | 1581 ms | 445 ms |
| FFN down GEMM 8192→3072 (×28) | 789 ms | 222 ms |
| q/o GEMM 3072→3072 (×56) | 595 ms | 168 ms |
| k/v GEMM 3072→1024 (×56) | 201 ms | 56 ms |
| MHA 24/8 heads d=128, 2048×2048 (×28) | 584 ms | 584 ms (not a matmul precision op) |
| everything else (norms, SiLU, mul/add, RoPE, copies, LM head) | ~140 ms | ~140 ms |
| **sum** | **3.89 s (385 tok/s)** | **~1.6 s (~930 tok/s)** |

GEMMs are 81% of the chunk at `high`. `medium` makes them 3.55× faster
(2048×3072×8192: 3.7 → 13 TFLOPS).

**Accuracy of `medium` for prefill** (`bench/handoff_probe.py`, 346-token prompt, KV handed
to llama.cpp BF16): first-token KL 1.6e-3 (high: 3.3e-4; a Q4_K_M model computing its own KV:
2.2e-2), 64/64 greedy tokens identical. K/V rel. error per layer 0.3–3% / 1.5–8%.

## 512-row chunks (`--chunk 512`, medium): worse
GEMMs cost the same in total (M=512: 2.7 ms vs 7.9 ms at 2048 for the FFN GEMM), but MHA
at seq_len 512 is untuned and slow: 16.8 ms/layer for the first chunk, 58.9 ms/layer for
the third (vs 20.8 ms for the whole 2048-row chunk). 1494 tokens: ~3.2 s. Short prompts gain
(346 tokens: ~0.84 s), long ones lose.

## Host-side handoff fixed
Before: worker eval → export f32 → interleave + f16 cast (350 ms) → shm → parent copy
(50 ms) → `from_kv` f16→f32→f16 + `build` joins (620 ms) → import. Now the worker writes the
finished llama.cpp blob into shared memory (`llama_state.layout`), bf16→f32 by bit shift,
4 threads; the parent restores from the shm view: pack 150 ms, IPC 2 ms, import 33 ms.

## Two requests, 3B (bench/hybrid_bench.py, Q4_K_M on iGPU, B prompt 1494 tokens)
A alone 35 tok/s.
| mode | A tok/s during B prefill | A overall | B TTFT | makespan |
|---|---|---|---|---|
| B prefill on iGPU | 6.2–8.2 | 23.5–23.6 | 2.09–2.20 s | 8.56–8.57 s |
| NPU high, old handoff (earlier today) | 28.4 | 29.1–29.3 | 5.74–5.85 s | 7.88–7.90 s |
| NPU medium, old handoff | 23.3–23.5 | 25.4–25.5 | 3.36–3.44 s | 7.90–7.93 s |
| **NPU medium, blob in shm** | **20.1** | **25.6** | **2.39 s** | **7.88–7.89 s** |

Worker breakdown (last row): NPU eval 2.12 s, KV read 5 ms, pack 150 ms, import 33 ms.

## Autotuning the 3B prompt shapes (`bench/iron_tune.py`, IRON's tuner, medium): no gain
The shipped `npu2.json` already holds 3B prompt entries (seven shapes; not only 1B's).
The tuner (at rows=2048, standalone) reported gains for MHA (−4.75 ms/call), Add, SiLU, Mul,
LM-head GEMV and the KV Copy; GEMMs kept their defaults. In the real graph (1494-token eval,
no iGPU load, NPU pmode `default`), swapping the tuned entries into the shipped profile:
| swapped in | eval 1493 tokens |
|---|---|
| none (shipped) | **1.61 s (~930 tok/s)** |
| MHA (B_q 32, pv_cores 2, 8 pipelines) | 4.74 s |
| Add, SiLU, Mul | 1.61 s |
| GEMV, RMSNorm | 1.61 s |
| Gather, RoPE, Sample (new entries) | 1.61 s |
Tuned profile discarded. Also `Profile.save` cannot write entries whose dimensions are
objects (Copy's TensorAccessPattern, GEMM's dtype/epilogue): see pitfalls.

## Where 3B NPU prefill stands
1493 tokens in 1.61 s standalone (iGPU Q4_K_M alone ~2.0 s), ~2.1 s while the iGPU decodes.
Remaining per-chunk: GEMMs ~0.89 s at bfp16 (13 TFLOPS on the big ones), MHA 0.58 s, rest
~0.14 s. NPU power mode is `default`; `performance` needs root (untested).
