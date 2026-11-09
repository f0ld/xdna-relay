# 0001 — Backend-agnostic op contracts

Status: Accepted (2026-11-09)

## Context
We start with NPU kernels plus PyTorch references. Later an iGPU backend for some ops may
be needed (and the NPU-only mode must not depend on model code knowing about devices).

## Decision
Every op (GEMM, Q4 GEMM, attention, norm, RoPE, activation, …) is defined **once** as an
`OpContract`: input/output `TensorSpec`s (symbolic shapes, dtype, layout), static params and
a tolerance vs the reference. Implementations register per backend (`ref`, `npu`, later
`igpu`) behind that contract. Weight repacking for a backend is an explicit `prepare()` step,
not hidden inside the op call. Tests target the contract, not a backend.

## Consequences
- An iGPU backend slots in by registering implementations; tests are reused.
- The reference backend is mandatory for every op and is the test oracle.
- Some NPU-specific fusion (e.g. GEMM + SiLU epilogue) appears as its own contract
  (`gemm_silu`) with a reference composed of the unfused ops.
