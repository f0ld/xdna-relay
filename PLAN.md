# PLAN

Status values: `TODO`, `IN PROGRESS`, `REVIEW`, `DONE`, `BLOCKED` (see PROTOCOL.md §6).

## Current

- Since 2026-11-24 the Architect implements directly (Implementer paused, ADR 0006). Tasks
  keep a done-command and tests; review files are written only for larger milestones.
- Active task: next — open: persist calibration across restarts; Llama 3B / other pairs via calibration; long-run drift (thermal, AC/battery)
- Power: the machine runs platform profile `low-power` (fan noise); NPU may run FLM
  `--pmode performance`. Every bench row records the profile.
- Operator TODOs: see `docs/knowledge/environment.md` → "Operator TODOs" (driver timeout).

## Done (foundation)

- T-001 — upstream `vector_scalar_mul` on the NPU — DONE
- T-002 — op contract + backend registry, `ref` backend for `gemm_bf16` — DONE
- T-003 — GGUF reader + reference dequant — DONE
- T-004 — `npu` backend for `gemm_bf16` via IRON `GEMM` — DONE
- Baseline probe 2026-11-24: FLM vs llama.cpp Vulkan, alone and concurrent (ADR 0006).

## MP — Profile (what limits us)

- T-019 — Contention matrix: {NPU prefill, NPU decode, idle} × {iGPU prefill, iGPU decode},
  tok/s of each side; DRAM bandwidth per device alone (replaces UNVERIFIED numbers in
  xdna2-architecture.md; absorbs old T-007). `bench/contention.py` — Status: DONE
- T-020 — Bench harness: CSV schema, `scripts/bench.sh`, results in `bench/results/` (old T-006) — Status: TODO
- T-021 — Baselines of existing tools: llama.cpp same-device speculative decoding (draft on
  iGPU) and llama-server prompt cache, for the candidate model pairs; acceptance rates — Status: DONE (spec decode: 2026-11-24-spec-igpu.md; prompt cache: llama-server's slot + RAM cache matches our T-027 TTFT, 2026-11-25-server-prompt-cache.md)

## MN — NPU executor on IRON full-model graphs

- T-022 — Run IRON `applications.llama` (Llama 3.2 1B, npu2 profile) from our venv; tok/s and
  logits vs HF reference — Status: DONE (`scripts/test.sh T-022` 1/1; KL vs HF float32: ~1e-3 at high, 2–10e-3 at medium, as close as llama.cpp Q8_0/Q4_K_M; bench/results/2026-11-25-logits-check.md)
- T-023 — `sched/executor.py`: executor interface (prefill, decode, verify(k tokens) → k
  logits rows, rollback(pos), kv export/import) + `NpuExecutor` over IRON `Session` +
  `RefExecutor` (PyTorch) — Status: IN PROGRESS (interface `sched/executor.py` done; NPU/Ref executors and KV export/import open)
- T-024 — `LlamaCppExecutor` (iGPU) behind the same interface — Status: DONE (`scripts/test.sh T-024` 6/6)
- T-028 — Per-op profile of the IRON Llama graph (prefill and decode separately) — Status: IN PROGRESS

Own pieces, each an A/B against the IRON graph it replaces (Operator, 2026-11-24):
- T-018 — Q4NX-Gemma4 pack + GGUF Q4_0/Q4_K → Q4NX repack *(spec, tests exist)* — Status: TODO
- T-005 — `q4_gemm` npu backend via `flm.Dequant` + `flm.ProjectionGEMM` (prefill) — Status: TODO
- T-029 — (deprioritised, ADR 0007) Q4 GEMV for decode (IRON Llama decode is bf16 GEMV, ~2.5 GB/token at 1B vs ~0.7 at Q4) — Status: TODO
- T-008 — Tune q4_gemm / q4_gemv — Status: TODO
- Small ops (norms, RoPE, eltwise) only if T-028 shows they matter.

## MS — Speculative decoding across devices

- T-025 — Draft on NPU, verify on iGPU — Status: DONE (negative, ADR 0007: ~0–10% at best; FLM unusable as draft)
- T-040 — Persist online calibration per setup (`sched/calibration_store.py`: key = model, NPU, packed, engine mode, prior, power source; atomic save every 30 s and on close; restore on start; `--calibration-file`, `--no-persist`) — Status: DONE (`scripts/test.sh T-040` 6/6; hardware: a restarted server starts converged, no exploration, estimates within ±16% from its first request; bench/results/2026-11-25-calibration.md)
- T-039 — Calibration fixes from hardware runs 3–8 (loss per device/size from request windows, bounded exploration, per-bin corrections, split's own loss) — Status: DONE (`scripts/test.sh T-038` 14/14; from a wrong prior every size converges to the measured-best placement within ~20 requests, estimates ±13%)
- T-038 — Online calibration of the placement cost model (`sched/online.py`: per-part ridge/forgetting line fits seeded by the preset; decode keep from stepper step times; server default, `--no-calibrate`, `GET /x_relay/calibration`) — Status: DONE (`scripts/test.sh T-038` 8/8; hardware: from a wrong prior (Llama 3B preset on Qwen3-8B) estimates within ±11% for ≥279 tokens after ~12 requests, decisions match the measured best except a near-tie at 398 tokens; bench/results/2026-11-25-calibration.md)
- T-037 — Placement presets recalibrated for the batched engine (`qwen3-8b-packed-batched`, auto-selected) — Status: DONE (`scripts/test.sh T-032` 29/29; fits within 4%; split validated at 282 tokens: lowest TTFT + A loss, A done first)
- T-036 — Mixed batches: iGPU prefill chunks ride along with decode tokens (stepper does all shared-context work) — Status: DONE (`scripts/test.sh T-034/T-035` pass; chunk 64–512 only trades A's speed during prefill vs B's TTFT, totals get worse below 256; default 256)
- T-035 — Batched decode across slots (one llama.cpp context, n sequences, stepper thread) — Status: DONE (`scripts/test.sh T-035` 5/5, T-034 10/10; 2 clients 30 vs 16 tok/s total, 4: 56 vs 16; with NPU prefill A keeps 10 vs 0.9 tok/s during B's prefill; bench/results/2026-11-25-batched-decode.md)
- T-034 — M7: OpenAI-compatible server (`python -m xdna_relay.server`): slots on iGPU, NPU prefill worker, prefix cache, placement per request, SSE streaming — Status: DONE (`scripts/test.sh T-034` 9/9; Qwen3-8B two requests via HTTP: A 10.2 vs 1.8–2.7 tok/s during B's prefill, B TTFT 4.7–4.9 vs 5.6–6.0 s)
- T-033 — Prompt-chunk GEMMs on pre-packed bfp16 weights (`models/packed.py`; replaces Q4 for prefill: GEMM re-reads B per 256 rows, Q4 would still be dequantized to bfp16) — Status: DONE (`scripts/test.sh T-033` 2/2; 8B prefill 1.33–1.5x faster on long prompts, two-request makespan −21% vs iGPU; short-prompt fixed cost fixed with a streamed-B prompt version ≤640 rows: packed now faster than bf16 at every length)
- T-032 — Prefill placement policy (`sched/placement.py`: iGPU / NPU / split by measured cost: TTFT + decode time lost by active requests) — Status: DONE (`scripts/test.sh T-032` 19/19; estimates within 15% of measured; `auto` picks the measured best on Qwen3-8B at 67/382/1702 tokens)
- T-031 — Qwen3 graph for IRON (`backends/npu/models/qwen3.py`: Llama layer + q/k RMSNorm) + handoff — Status: IN PROGRESS (`scripts/test.sh T-031` 3/3; 0.6B 64/64 identical after handoff; 8B: NPU prefill 430 tok/s with an MHA profile entry, two requests: B TTFT = iGPU's, A 8.4 vs 3 tok/s, makespan −14%)
- T-030 — Adaptive draft length in our speculative loop — Status: DONE (`scripts/test.sh T-030`; Qwen3-8B 15.5 → ~25 tok/s; open: under-picks k on code)
- T-026 — Pipelined speculation (draft k+1.. while target verifies k); adaptive draft length
  from measured contention — Status: TODO

## MC — Caching and the KV component

- T-009 — KV cache layout spec + `src/xdna_relay/kv/` (ADR 0002) — Status: TODO
- T-027 — Prefix cache: live-sequence reuse + host-RAM snapshots (LRU by bytes) — Status: DONE (`scripts/test.sh T-027` 8/8; TTFT 6.3 s → 0.09–0.34 s on 2.4k-token prompts)
- T-016 — KV handoff NPU prefill → iGPU decode (llama.cpp KV import); tokens match REF — Status: DONE (Llama 3.2 1B: 64/64 greedy tokens identical, KL 4.9e-4, handoff 23 ms; `scripts/test.sh T-016` 3/3)
- T-017 — Scheduler `HYBRID`: NPU prefills request B while iGPU decodes A — Status: IN PROGRESS (1B: A keeps 22 vs 6 tok/s, B TTFT −35%; 3B at IRON medium precision + shm blob handoff: A keeps 20 vs 6–8 tok/s, B TTFT 2.4 vs 2.1–2.2 s, makespan −8%)

## MK — Kernels (only on a measured bottleneck, ADR 0006)

- Candidate: NPU prefill utilisation (FLM ~4 TFLOP/s effective). Opens only with a profile.

## Later

- M7 — Own OpenAI-compatible server (`/v1/chat/completions`, `/v1/models`) over the scheduler; bench vs FLM and llama.cpp.
