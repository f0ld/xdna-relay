# 0003 — Scheduler separate from model code

Status: Accepted (2026-11-09)

## Context
Today: prefill on NPU, decode on iGPU. Later: NPU-only, and an own server with several modes.

## Decision
Models describe the computation graph per phase (`prefill`, `decode`) as op-contract calls,
with no device names and no data movement. A scheduler (`src/xdna_relay/sched/`) picks the
executor per phase (`HYBRID`, `NPU_ONLY`, `REF`) and performs the KV handoff via the KV
cache component (ADR 0002). Executors bind op contracts to backends (ADR 0001).

## Consequences
- Adding NPU decode (M6) is adding a decode executor, not editing model files.
- `REF` mode runs any model end-to-end on PyTorch, giving a full-model oracle.
- The scheduler is the only place that knows about devices and handoff timing.
