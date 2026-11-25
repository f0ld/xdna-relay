# T-032: prefill placement policy (2026-11-25)

Machine: Ryzen AI 9 HX 370, platform profile `balanced`, on AC, NPU pmode `performance`.
Policy: `src/xdna_relay/sched/placement.py`; tests `scripts/test.sh T-032` (19/19).
Scenario (`bench/hybrid_bench.py`): request A decodes on the iGPU; B arrives and prefills on
the iGPU (second context), on the NPU (worker process + KV handoff), split (NPU whole
256-row blocks, iGPU the rest) or wherever `auto` (the policy) says.

## Warm NPU prefill vs length (`bench/npu_prefill_curve.py`, nothing else running)
| tokens | 64 | 128 | 256 | 384 | 512 | 768 | 1024 | 1493 | 2000 |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B s | 0.599 | 0.623 | 0.697 | 1.166 | 1.263 | 1.813 | 2.663 | 3.939 | 4.810 |
| Llama 3B (medium) s | 0.307 | 0.318 | 0.351 | 0.547 | 0.580 | 0.822 | 1.076 | 1.612 | 2.205 |
Steps every 256 rows: `fixed + block x ceil(n/256)` fits within 12% (8B 0.024 + 0.620 s,
3B 0.042 + 0.266 s). (The earlier "345 tokens in 2.15 s" was the probe's cold first call.)

## B's TTFT while A decodes (s), measured / policy estimate
| model | B tokens | iGPU | NPU | split |
|---|---|---|---|---|
| Qwen3-8B | 67 | 0.57–0.72 / 0.74 | 1.06 / 1.04 | — |
| Qwen3-8B | 382 | 1.81–1.84 / 1.72 | 1.74–1.80 / 1.80 | 2.03–2.12 / 1.96 |
| Qwen3-8B | 1702 | 5.52–6.05 / 5.83 | 5.59–5.66 / 5.61 | 5.72–5.94 / 5.90 |
| Llama 3B | 64 | 0.26–0.38 / 0.35 | 0.60–0.62 / 0.60 | — |
| Llama 3B | 350 | 0.68–0.77 / 0.69 | 0.94 / 0.95 | 0.88–0.89 / 0.99 |
| Llama 3B | 1494 | 1.85–2.20 / 2.06 | 2.30–2.39 / 2.34 | — |
Contended TTFT = slow x alone + wait (least squares over the three lengths): 8B iGPU 1.259x
+ 0.527 s, NPU 1.229x + 0.251 s; 3B iGPU 0.889x + 0.272 s, NPU 1.309x + 0.199 s. The iGPU
prefill queues behind A's decode steps (fixed wait); the NPU's wait includes the handoff.
A's speed during B's prefill (long B): iGPU keeps ~20%, NPU ~53–57%.

Split never wins once the iGPU tail pays its wait: the policy keeps the option but does not
pick it at these presets.

## `auto` (decode_weight 1: TTFT + A's lost decode-seconds), Qwen3-8B
| B tokens | chose | B TTFT | A during B | makespan |
|---|---|---|---|---|
| 67 | iGPU | 0.64 s | 10.9–11.0 | 9.86–10.00 s |
| 382 | NPU | 1.63–1.75 s | 8.6–9.2 | 10.47–10.63 s |
| 1702 | NPU | 5.48–5.54 s | 8.4–8.5 | 12.36–12.37 s |
In each case the measured best (or tied-best) option. Decisions alternate near 256-row
boundaries where the two costs are within a few percent.
