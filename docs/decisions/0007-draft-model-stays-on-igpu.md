# 0007 — Speculative decoding stays on one device; the NPU serves prefill and caching

Status: Accepted (2026-11-24). Refines ADR 0006 (MS milestone).

## Context
`bench/results/2026-11-24-t025-draft-placement.md`: a 0.6B draft decoding on the NPU cuts
iGPU 8B decode by 31% (shared DRAM bandwidth); a draft round on the iGPU is 116 ms of which
~42 ms is drafting. Hiding the draft on the NPU gains an estimated 0–10%, and needs our own
Qwen3 NPU decode runtime with Q4 weights. Same-device speculation already gives +13–82%,
with a strong dependence on draft length per content type (n=3 vs n=8: up to 2x).
FLM cannot serve drafts at all (~415 ms fixed per request).

## Decision
- Speculative decoding runs where the target runs (iGPU, llama.cpp + our patches).
  Our work there: adaptive draft length, model-pair choice, hybrid-model rollback.
- The NPU's roles: prefill (compute-bound, most tok/J), prefill of request B while the iGPU
  decodes A (iGPU decode −15%), and prompt/prefix caching.
- T-029 (Q4 GEMV for NPU decode) is deprioritised: it served the NPU-draft plan.
  Revisit if a measurement shows NPU decode on a critical path.

## Consequences
- MS milestone becomes "speculation policy" on the iGPU executor.
- The NPU executor's first product use is prefill + KV handoff (T-016/T-017), measured on
  multi-request workloads, where its value shows; a single request alone gets faster prefill
  on the iGPU (1B: iGPU 2523 tok/s vs IRON NPU 968).
