# T-033: prompt-chunk GEMMs on pre-packed bfp16 weights (2026-11-25)

Machine: Ryzen AI 9 HX 370, platform `balanced`, on AC, NPU pmode `performance`.

## Why: NPU prefill is weight-bandwidth bound
`flm.ProjectionGEMM` re-reads its whole B (the weight) from DDR for every 256-row unit of the
prompt (`_units_valid`, `b_passes_repeat`). Per 256-row block: 3B 0.266 s for ~5.6 GB bf16
weights, 8B 0.620 s for ~14 GB: ~22 GB/s. That is also why NPU prefill and iGPU decode
contend so hard. `m_chunk` (one B fetch for several row blocks) is refused under a per-call
row bound ("m_chunk must be 1 under a per-call row bound"), which the prompt graph needs.
Q4 weights do not help prefill here: IRON dequantizes Q4NX into bfp16 in DDR first
(`DequantBFP`), and the GEMM re-reads that per unit anyway.

## Standalone GEMM, 8B shapes, 2048 rows (iron.tune.measure)
| K x N | bf16 col-major (IRON's F.linear) | bfp16 packed (`bfp16_b=True`) |
|---|---|---|
| 4096 x 12288 (gate/up) | 20.95 ms | 10.74 ms |
| 12288 x 4096 (down) | 20.96 ms | 16.25 ms |
| 4096 x 4096 (q/o) | 7.04 ms | 3.77 ms |
| 4096 x 1024 (k/v) | 1.77 ms | 0.94 ms |

## In the graph (`models/packed.py`, `IronLmExecutor(packed=True)`)
Packed B held as an `iron.State` of bfp16ebs8 blocks, filled after load with
`ProjectionGEMM.pack_b` (same profile as the graph's GEMM). Needed a patch of mlir-aie's
`Device.dma_bd_fits` (np.dtype() on the block-float marker raises for the 8B's large B).

Accuracy (handoff into llama.cpp, 345 tokens): 0.6B (Q8_0 ref) KL 3.75e-3, 64/64 identical
(unpacked 4.5e-3); 8B (Q4_K_M ref) KL 2.79e-2, 50/64 (unpacked the same).

Warm 8B prefill (`bench/npu_prefill_curve.py`):
| tokens | 64 | 256 | 512 | 1024 | 1493 | 2000 |
|---|---|---|---|---|---|---|
| bf16 | 0.599 | 0.697 | 1.263 | 2.663 | 3.939 | 4.810 |
| packed | 0.863 | 0.950 | 1.255 | 1.907 | 2.623 | 3.486 |
Packed: 0.510 s + 0.362 s per 256 rows (bf16: 0.024 + 0.620). Long prompts 1.33-1.5x
faster; below ~400 tokens slower (a +0.26 s fixed cost per call, cause not yet found).

## Two requests, Qwen3-8B packed (A decodes; B prefills)
| B tokens | mode | A during B | B TTFT | makespan |
|---|---|---|---|---|
| 67 | iGPU | 9.4–10.3 | 0.64–0.78 s | 10.08 s |
| 67 | NPU packed | 12.1–12.5 | 1.32–1.36 s | 10.08 s |
| 382 | iGPU | 5.5–6.7 | 1.99–2.10 s | 11.56–12.09 s |
| 382 | NPU packed | 9.7–10.5 | 1.90–1.95 s | 10.97–11.43 s |
| 1702 | iGPU | 2.8 | 5.64 s | 14.47 s |
| 1702 | NPU packed | **10.4** | **4.63 s** | **11.42–11.44 s** |
| 1702 | NPU bf16 (earlier) | 8.1–8.6 | 5.59–5.66 s | 12.35–12.72 s |
Packed: B's TTFT −18% vs bf16 NPU and −18% vs iGPU; A keeps 69% of its speed (bf16 NPU
53%, iGPU prefill 20%); makespan −21% vs iGPU. Placement preset `qwen3-8b-packed`.

## Short-prompt fixed cost: found and fixed (two prompt versions)
Per-op at 64 rows (8B, GEMMs only): bf16 377 ms, packed 620 ms. The packed GEMM resolves to
a **resident** B (`_b_resident`: its 8 k-iterations fit the memtile slots, which bf16's do
not): B is read from DDR once per call and replayed to every 256-row unit, but under a row
bound the cores still drain the replays of all 8 units ("past them the cores drain each
column-block's B replay uncomputed"), a fixed cost. `k_tile=256` (16 k-iterations) makes it
stream per unit like bf16: 64 rows 299 ms, 1494 rows 1770 ms (resident 1361 ms).
`pack_b` output is identical for k_tile 256 and 512, so both GEMMs read one packed copy.
`Qwen3Packed.load_versions` compiles two prompt-chunk versions into one arena (shared
weights, packed states, KV cache): `short` (profile `qwen3-streamed`) for calls of up to 640
rows, `long` (resident) above; `IronLmExecutor.eval` picks per call.

Warm 8B prefill (s):
| tokens | 64 | 128 | 256 | 384 | 512 | 768 | 1024 | 1493 | 2000 |
|---|---|---|---|---|---|---|---|---|---|
| bf16 | 0.599 | 0.623 | 0.697 | 1.166 | 1.263 | 1.813 | 2.663 | 3.939 | 4.810 |
| packed, resident only | 0.863 | 0.889 | 0.950 | 1.194 | 1.255 | 1.569 | 1.907 | 2.623 | 3.486 |
| **packed, two versions** | **0.525** | **0.545** | **0.607** | **0.990** | **1.046** | 1.572 | 1.908 | 2.615 | 3.476 |
Accuracy: 0.6B short (345 tok) as before; long (1445 tok, `long` version) KL 3.59e-3, greedy
identical up to 20 (unpacked: KL 5.25e-3, also 20).

Two requests (8B, A alone 16 tok/s), B TTFT / A during B's prefill:
| B tokens | iGPU | NPU packed, two versions |
|---|---|---|
| 67 | 0.60–0.72 s / 10.0–11.1 | 0.96 s / 11.4–11.5 |
| 382 | 1.70–1.94 s / 5.9–7.2 | **1.57–1.63 s / 10.2–10.5** |
| 1702 | 5.44–5.82 s / 2.8–3.6 | **4.50–4.78 s / 10.4–10.7** (makespan 11.3 vs 14.3 s) |

Placement: preset `qwen3-8b-packed` got the short version's curve (0.100 s + 0.459 s/256
rows up to 640 rows); the decode-loss term now excludes the fixed waits (A decodes at full
speed while an iGPU prefill queues behind its steps), which had sent a 67-token prompt to the
NPU. Decisions now: iGPU at 64 tokens, NPU from ~200 tokens (8B), from ~1000 (3B).
