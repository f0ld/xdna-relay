# PLAN

Status values: `TODO`, `IN PROGRESS`, `REVIEW`, `DONE`, `BLOCKED` (see PROTOCOL.md §6).
Task specs exist only for tasks marked *(spec)*; the rest are planned titles the Architect
turns into specs when their dependencies are done.

## Current

- Active task: **T-001** (M0)
- Operator TODOs: see `docs/knowledge/environment.md` → "Operator TODOs" (driver timeout before M2).

## M0 — Environment validated, an upstream mlir-aie example builds and runs on the NPU

Bootstrap (done by Architect, 2026-11-09): driver/firmware/XRT/memlock verified, `.venv` with
pinned mlir-aie + Peano + IRON installed, `scripts/env-check.sh` → `ENV PASS`.

- T-001 — Port upstream `vector_scalar_mul` and run it on the NPU *(spec)* — Status: TODO

## M1 — Q4 GEMM correct vs PyTorch reference

Approach per ADR 0004: wrap IRON's open FLM ops (Q4NX dequant → bfp16, ProjectionGEMM).

- T-002 — Op contract + backend registry skeleton (`src/xdna_relay/ops/`), `ref` backend for `gemm_bf16` — Status: TODO
- T-003 — Q4NX (Gemma 4 variant) pack/unpack/dequant in `src/xdna_relay/formats/q4nx.py`, vs IRON's `q4nx.dequantize` and a PyTorch reference — Status: TODO
- T-004 — `npu` backend for `gemm_bf16` via IRON `GEMM` — correct vs ref — Status: TODO
- T-005 — `q4_gemm` contract + `npu` backend via `flm.Dequant` + `flm.ProjectionGEMM`, correct vs PyTorch at prefill shapes (M ∈ {32,128,512}, K/N from a real model) — Status: TODO
- Decision D2 (Architect): first weight format — Q4NX-Gemma4 vs GGUF Q4_0/Q4_K repack.

## M2 — Q4 GEMM fast

- T-006 — Bench harness (`bench/`, CSV schema in `scripts/bench.sh`) + naive baseline (CPU dequant + NPU bf16 GEMM) — Status: TODO
- T-007 — Measure NPU DRAM bandwidth and dispatch overhead on this machine (replace the UNVERIFIED numbers in xdna2-architecture.md) — Status: TODO
- T-008 — Tune q4_gemm (IRON autotune profile, weight-resident B, tile_n/k_tile) and compare with `flm bench` where comparable — Status: TODO
- Requires Operator TODO 1 (tdr_timeout_ms).

## M3 — One transformer layer (prefill) correct

- T-009 — KV cache layout spec `docs/knowledge/kv-cache-layout.md` + `src/xdna_relay/kv/` (Architect writes spec; Implementer the component) — Status: TODO
- T-010 — Contracts + npu backends: RMSNorm, RoPE, SiLU/GELU-mul, residual add (IRON ops) — Status: TODO
- T-011 — Prefill attention (causal, GQA) via IRON `mha` or `flm.prefill_attn`, writing KV through the layout — Status: TODO
- T-012 — One decoder layer of the target model (Decision D3) on NPU vs HF/PyTorch layer output — Status: TODO

## M4 — Full-model prefill on the NPU

- T-013 — Model graph + REF executor end-to-end (logits match HF) — Status: TODO
- T-014 — NPU executor: full prefill, last-token logits vs REF; TTFT bench — Status: TODO
- Decision D1 (Architect): M5 path (own runtime + KV export vs ggml-xdna2 + ggml-vulkan).

## M5 — KV-cache handoff to llama.cpp decode on the iGPU

- T-015 — llama.cpp Vulkan build on the 890M, decode bench for the target model — Status: TODO
- T-016 — KV export into llama.cpp's KV tensors; decode continues and matches REF tokens — Status: TODO
- T-017 — Scheduler `HYBRID` mode end-to-end + TTFT/tok/s/energy bench vs FLM and Vulkan-only — Status: TODO

## M6 — Decode on the NPU (later; coarse)

GEMV / matrix-vector kernels (IRON `gemv`, xdna-engine int8×int4 ideas), KV stays on NPU,
`NPU_ONLY` executor for decode, fused dispatch to cut per-token commands.

## M7 — Own OpenAI-compatible server + scheduler (later; coarse)

`/v1/chat/completions` + `/v1/models`, scheduler modes `NPU_ONLY` and `HYBRID`, request queue,
prompt cache, bench vs FLM and llama.cpp.
