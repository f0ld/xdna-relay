# Prior art (studied 2026-11-09)

Commits read: `amd/IRON@3c47c7a` (2026-11-09), `Xilinx/mlir-aie@e5a3fa1` (2026-11-07),
`atassis/xdna-engine@35e4adf` (2026-10-25), `mistrjirka/llama.cpp-xdna2@0a0e2e7`
(2026-10-21), `FastFlowLM/FastFlowLM@f4e9a0c` (local clone at
`~/.cache/claude-scratch/npu/FastFlowLM`).

## Summary table

| Project | What | License | Reuse for us |
|---|---|---|---|
| amd/IRON | AMD's operator + graph + model library on mlir-aie. GEMM, GEMV, MHA, RMSNorm, RoPE, SiLU, softmax…; Llama 3.2, Granite 4, Gemma 4 E2B run fully on NPU2 | Apache-2.0 | **Primary dependency.** Ops, graph compiler, autotuning, FLM-port ops (Q4NX dequant, ProjectionGEMM, prefill attention) |
| Xilinx/mlir-aie | Compiler (aiecc, aie-opt), `aie.iron` Python API, kernel library (`aie_kernels/`), programming examples | Apache-2.0 WITH LLVM-exception | Toolchain; kernel library incl. `quant/q4nx_dequant.cc`; examples for M0 |
| atassis/xdna-engine | Rust inference engine + 48 hand-written AIE kernels; Qwen3-0.6B decode on NPU (18 tok/s), ASR, embeddings | Apache-2.0 | Ideas + measurements: data-movement thesis, fused decode, int8×int4 GEMM/GEMV kernels, roofline doc |
| mistrjirka/llama.cpp-xdna2 | ggml backend `ggml-xdna2`: **prefill matmuls + attention on NPU, decode on CPU**, GGUF Q4_0/Q4_1/Q4_K/Q5_K/Q6_K/Q8_0 | MIT | Most relevant to M5. Shows GGUF-native NPU prefill inside llama.cpp |
| FastFlowLM | NPU runtime/server (`flm`), models in Q4NX; runtime MIT, **NPU kernels/engine libs closed** (binary license) | MIT (runtime) + FastFlowLM NPU Binary License | Reference performance + numerics (`flm bench`), Q4NX format, AutoModel wrapper structure |

## amd/IRON

- Layout: `iron/operators/` (one class per op, `build()` lays out ObjectFIFOs/Workers),
  `iron/common/` (graphs, buffer arena, images), `iron/lm/` + `iron/applications/` (models).
- On NPU2 a graph's ops compose into **one full ELF**; a prompt chunk is one command
  (→ needs `tdr_timeout_ms=150000`, see environment.md).
- **`iron/operators/flm/`** = ports of FastFlowLM's Gemma 4 overlays:
  - `q4nx.py` — Q4NX unpack/dequantize (numpy), constants from `aie.iron.kernels.quant`.
  - `dequant/` — NPU op: q4nx → bfp16ebs8 B-operand storage (aie2p only).
  - `gemm/` — `ProjectionGEMM`: bf16 A × bf16-or-bfp16 B, fused epilogue
    (none/SiLU/Sigmoid/GELU), M/K/N are **runtime parameters** (one xclbin, many shapes).
    Grid 4×8 on NPU2. README compares against the shipped FLM overlay.
  - `prefill_attn/`, `layer/`, `lm_head/`, `gemma4_engine/` (runs the ports inside FLM's engine).
- Tests: every op has a numpy reference + `Tolerance`; `iron/tests/operators/test_catalog.py`.
  Install check: `pytest iron/tests/operators/test_catalog.py -k AXPY`.
- **Implication:** M1 (Q4 GEMM correct) = compose `flm.Dequant` + `flm.ProjectionGEMM` (or
  IRON's plain `GEMM` on dequantized weights) behind our op contract, then test vs PyTorch.
  Writing a Q4 GEMM kernel from scratch is a fallback, not the plan (ADR 0004).

## Xilinx/mlir-aie

- Python API `aie.iron`: `ObjectFifo`, `Worker`, `Runtime`, `Program`, `ExternalFunction`,
  `@iron.jit` + `CompileTime[...]` params, `iron.tensor(..., device="npu")`.
- Kernel factories in `aie.iron.kernels` (e.g. `kernels.scale`, `quant.q4nx_dequant`), each
  with a numpy `*_ref` and tolerance contract.
- `programming_examples/basic/vector_scalar_mul/` is the M0 smoke example (`-d npu2`).
- Wheels pinned by IRON: see toolchain.md.

## atassis/xdna-engine

- Rust workspace (15 crates), runs on Krackan (Ryzen AI 9 465). Prebuilt xclbins; kernel
  toolchain is a pinned mlir-aie fork (`toolchain.lock`).
- Thesis (docs/data-movement-thesis.md): workloads are **data-movement bound**; measured NPU
  DRAM read ~52.7 GB/s, fabric ceiling ~62.7 GB/s vs LPDDR theoretical 136.5 GB/s
  (docs/aie2p-architecture-and-roofline.md, measured 2026-08 on their machine — `UNVERIFIED` here).
- Kernels of interest: `dequant-int4-group`, `gemm-int8xint4(-dequant)`, `gemv-int8xint4`
  (all "unverified in this index" per their INDEX.md).
- Dispatch cost matters: Qwen3-0.6B decode went 366 → 170 configures/token by fusing.

## mistrjirka/llama.cpp-xdna2

- Backend in `ggml/src/ggml-xdna2/` (C++ host code + `kernels/` AIE sources compiled
  on first run with mlir-aie 1.4.0 + llvm-aie in `~/.local/share/ggml-xdna2/venv`, cached in
  `~/.cache/ggml-xdna2`). Docs: `docs/backend/XDNA2.md`.
- NPU does prefill weight matmuls and attention (head dim 128/256, ≥32-token batches,
  context ≥1024); norms/RoPE/linear-attention on CPU. Decode on CPU (AVX-512) because CPU
  reads DRAM faster (~70 GB/s) than the NPU (~53 GB/s) — their measurement.
- Converts GGUF quant blocks to the NPU's BFP format (`ggml-xdna2-bfp.cpp`).
- Reported (Ryzen AI 9 365, power-saver): Qwen3.5-4B prefill 560 tok/s vs Vulkan 528, FLM 479.
  Gemma 4 12B prefill: 136 tok/s vs FLM 278 — large-model prefill is where FLM still wins.
- **Implication for M5:** an alternative to our own runtime is "NPU prefill via
  ggml-xdna2 + iGPU decode via ggml-vulkan" inside one llama.cpp process, where the KV cache
  handoff is just ggml backend scheduling. Decision deferred to M4/M5 (PLAN.md decision D1).

## FastFlowLM

- Runtime (`src/`, MIT): server, tokenizer, sampler, `AutoModel` wrappers
  (`include/AutoModel/modeling_<model>.hpp`). Engine per model is a **prebuilt closed** shared
  library `lib<model>_npu.so` + `xclbins/<ModelName>/` (binary license: redistribution
  unmodified only).
- `src/create_new_model.md` = the "flm-add" path: given an engine `.so` + xclbins + `model.q4nx`,
  write the AutoModel wrapper. Adding a model whose engine does not exist needs the kernels
  (IRON's `flm/` ports are the only open ones, Gemma 4).
- Weights: `model.q4nx` (safetensors container, see flm-formats.md).
- Use as baseline: `flm bench`, `flm validate`, and comparing outputs.
