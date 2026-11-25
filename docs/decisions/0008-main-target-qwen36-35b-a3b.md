# 0008 — Main target: Qwen3.6-35B-A3B; the Qwen3-8B NPU/iGPU path is paused, not dropped

Status: Accepted (2026-11-25, Operator decision). Extends ADR 0006/0007.

## Context
`bench/results/2026-11-25-moe-35b-a3b.md`: on the 890M iGPU Qwen3.6-35B-A3B (hybrid
linear/full attention MoE, 3B active) decodes at 20.9 tok/s vs Qwen3-8B 15.3, prefills at
471 tok/s, and with its built-in MTP draft reaches 31.9 tok/s on code (llama-server). It is
the Operator's preferred model. The 8B path (NPU prefill via IRON, placement with online
calibration, batched speculative decoding) is complete to the state in
`docs/knowledge/qwen3-8b-path.md`.

## Decision
- Main target: Qwen3.6-35B-A3B, as fast as possible on this machine, NPU included (an IRON
  prefill graph for its architecture is in scope).
- The project stays multi-model: model-specific code sits behind per-model interfaces
  (IRON graph per `model_type`, presets per model, executor capabilities such as rollback),
  so other models can be tuned with the same scheduler, cache and calibration.
- The 8B path is paused with a resume guide; its tests stay in the suite and must keep passing.

## Consequences
- New milestone MQ in PLAN.md: hybrid-state rollback (checkpoints), MTP drafting under
  BatchK, the batched engine on the 35B-A3B, then the IRON path (op inventory, graph, handoff).
- Hybrid models cannot truncate recurrent state: every component that rolls back (speculative
  verify, prefix cache) goes through an executor capability, not `truncate` alone.
