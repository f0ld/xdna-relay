# 0004 — Build NPU backends on IRON (mlir-aie + Peano), reuse its FLM ports

Status: Accepted (2026-11-09)

## Context
amd/IRON (Apache-2.0) already ships NPU2 operators (GEMM, GEMV, MHA, RMSNorm, RoPE, …),
whole-model graphs, and open ports of FastFlowLM's Gemma 4 overlays, including the Q4NX
dequant kernel and a bf16 × bfp16 ProjectionGEMM with runtime M/K/N. Writing a Q4 GEMM from
scratch would duplicate this. The project goal is an open inference path, not original kernels.

## Decision
- Toolchain = IRON + mlir-aie wheel + llvm-aie (Peano), pinned together in
  `requirements-npu.txt` (pins follow IRON's requirements.txt).
- NPU backends of our op contracts wrap IRON operators where they fit; we write own IRON
  operators / AIE C++ kernels only where none fits or a measured gap justifies it.
- Kernel-level work is still owned by us when we write it: tests vs our PyTorch references,
  not only IRON's numpy references.

## Consequences
- M1 is "Q4 GEMM via IRON flm ops behind our contract, correct vs PyTorch" — smaller and
  lower-risk than a new kernel.
- We track a nightly toolchain; bumping pins is an explicit task with a full test run.
- IRON API churn can break us; pin IRON by commit, never by branch.
