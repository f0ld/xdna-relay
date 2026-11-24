# 0006 — Scheduler first; full-model IRON on the NPU; kernels only for measured bottlenecks

Status: Accepted (2026-11-24). Refines ADR 0004; supersedes the op-by-op M1–M4 plan.

## Context
Operator direction (2026-11-24): the Architect implements directly (Implementer paused); use
the IRON/mlir-aie kernel library, put the project's effort into the scheduling layer
(NPU/iGPU split, speculative decoding, caching), profile first, write own kernels only for a
measured bottleneck.

Measured on this machine (Qwen3.5-4B, Q4, platform profile `low-power`, `bench/results/2026-11-24-baseline.md`):

| | iGPU decode | NPU decode | iGPU prefill | NPU prefill |
|---|---|---|---|---|
| alone | 20.4 tok/s | 14.2 tok/s | 443 tok/s | 480 tok/s |
| concurrent | 14.6 | 9.5 | 305 | 327 |

Decode is DRAM-bandwidth bound: ~2.7 GB/token, ~55 GB/s iGPU alone, ~65 GB/s for the whole
SoC with both busy. Splitting decode across devices gains ~18%; no decode kernel can move
that ceiling. The levers are fewer bytes per token (speculative decoding), skipped work
(prefix/KV caching), and overlapping compute-bound NPU prefill with iGPU decode.

IRON already ships full models (`iron.applications.llama`, npu2 profile) with a KV cache
(`iron.lm.blocks.KVCache`) and a `Session` that resumes from the first non-held token
(`iron.lm.session.Session.logits/forget/resume`) — the primitives draft verification and
prefix reuse need.

## Decision
1. Order of work: profile → baselines of existing tools → scheduler features. Every feature
   lands with a bench row against the baseline it claims to beat.
2. NPU executor = IRON's full-model `CausalLM` graphs, wrapped behind our executor
   interface, with HF safetensors weights. Op-by-op model assembly (old M1–M4) is paused.
3. iGPU executor = llama.cpp (Vulkan), driven through its API/server; we patch it where the
   scheduler needs hooks (KV import/export, draft verification).
4. FastFlowLM is a baseline and numerics reference only (closed engine; no KV or verify hooks).
5. Own kernels / IRON operators only after a profile names the bottleneck; the first
   candidate is NPU prefill utilisation (FLM reaches ~4 TFLOP/s effective, compute-bound).
6. Bring-up model: Llama 3.2 1B (IRON reference exists). The first speculative pair is
   Llama-family (shared tokenizer): 1B draft, 3B/8B target — final pick after T-021.

## Consequences
- Our own pieces replace parts of the IRON graph one at a time, each an A/B benchmark against
  what it replaces. First: Q4 weights (T-018/T-005/T-029) — IRON's Llama decode reads bf16
  weights, ~3.5x the bytes of Q4 in a bandwidth-bound phase.
- ADRs 0001–0003 stay: contracts, the KV cache as its own component, and the scheduler
  separate from model code still hold; executors now wrap whole graphs, not single ops.
- NPU-only decode (old M6) is demoted: the NPU's decode role is the draft model.
