# Architecture (draft 1, 2026-11-09)

Architect-owned. Changes need an ADR in `docs/decisions/`.

## Goal and shape

Run transformer **prefill on the XDNA2 NPU** and **decode on the iGPU** (llama.cpp Vulkan)
now, with a path to **NPU-only** inference later. "Relay" = the scheduler handing a request
from one device to the next.

```
              ┌──────────────────────────── xdna_relay ─────────────────────────────┐
 request ──►  │ server (M7) ──► scheduler ──► executor(s) ──► backend ops            │
              │                   │   ▲            │             ├─ npu  (IRON/aie)  │
              │                   │   │            │             ├─ ref  (PyTorch)   │
              │      model graph ─┘   │            │             └─ igpu (later)     │
              │      (no device)      │            ▼                                 │
              │                       └──── kv cache (one layout spec) ──► llama.cpp │
              └──────────────────────────────────────────────────────────────────────┘
```

## Layers

### 1. Op contracts (ADR 0001)

Every op is declared **once** as a contract, independent of device:

```python
# src/xdna_relay/ops/contract.py (to be written in M1)
@dataclass(frozen=True)
class OpContract:
    name: str                      # "q4_gemm", "rms_norm", "rope", "attn_prefill", ...
    inputs: tuple[TensorSpec, ...] # shape symbols, dtype, layout
    outputs: tuple[TensorSpec, ...]
    params: tuple[str, ...]        # static params (eps, theta, group, ...)
    tolerance: Tolerance           # vs the reference backend

@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[str | int, ...]   # symbolic: ("M", "K")
    dtype: str                     # "bf16", "f32", "q4nx", ...
    layout: str                    # "row_major", "q4nx_gemma4", "bfp16ebs8_b", ...
```

Behind each contract: a registry of **backends** (`ref`, `npu`, later `igpu`), each a
callable `impl(*inputs, **params) -> outputs` plus optional `prepare(weights)` for one-time
weight repacking (e.g. q4nx → bfp16). The `ref` backend is PyTorch on CPU and is the
oracle for tests. Tests are written against the contract, so an `igpu` backend gets the
same tests for free.

Weight layouts are part of the contract (`layout`), with converters between them in
`src/xdna_relay/formats/` (GGUF, Q4NX, HF safetensors → internal).

### 2. Model graph (ADR 0003)

A model (`src/xdna_relay/models/<name>.py`) describes the computation as a sequence of op
contract calls over named tensors, for one **phase** (`prefill(tokens, kv)`,
`decode(token, kv)`). It never names a device and never moves data between devices.

### 3. Scheduler (ADR 0003)

Decides per request and phase which **executor** runs it, and when to hand off:

- `Mode.HYBRID` (now): prefill → NPU executor; export KV cache; decode → llama.cpp/Vulkan.
- `Mode.NPU_ONLY` (M6+): prefill and decode on the NPU executor; KV stays.
- `Mode.REF` (always): everything on the PyTorch reference (tests, debugging).

An executor binds each op contract to a backend and owns device buffers. The scheduler only
sees phases, executors and KV-cache handles.

### 4. KV cache (ADR 0002)

A component with a written layout spec (`docs/knowledge/kv-cache-layout.md`, written in M3),
not a side effect of the attention kernel:

- logical: per layer, K and V of shape `[n_kv_heads, n_ctx, head_dim]`, dtype bf16 (first
  version), positions `0..n_past-1` valid;
- physical: page/block granularity chosen to match both the NPU attention kernel's tiling
  and llama.cpp's KV tensors;
- operations: `allocate`, `append(layer, k, v, pos)`, `view(layer)`, `export(target)`,
  `import(source)`.

Hybrid mode exports it into llama.cpp's KV tensors after prefill; NPU-only mode keeps it in
NPU-visible host memory. Both use the same spec, so changing modes doesn't change the layout.

## Toolchain choice (ADR 0004)

NPU backends are built with **IRON** (operators, graphs) on **mlir-aie** + **Peano**, pinned
in `requirements-npu.txt`. We reuse IRON operators (including its open FastFlowLM ports:
Q4NX dequant, ProjectionGEMM, prefill attention) wherever they fit our contracts, and write
our own IRON operators/AIE kernels only where none fits.

## Source layout

```
src/xdna_relay/
  ops/          contract.py, registry.py, <op>.py (contract + ref backend)
  backends/     npu/ (IRON-based impls), igpu/ (later)
  formats/      q4nx.py, gguf.py, converters
  kv/           cache.py (layout spec implementation)
  models/       <model>.py (graphs)
  sched/        scheduler.py, executors
  server/       (M7)
  examples/     upstream example ports (M0)
tests/reference/  PyTorch/numpy references (protected)
tests/kernels/    per-op correctness tests (protected)
bench/            benchmark scripts, results/*.csv
```

## Open decisions

- **D1 (decide by M4):** M5 path — own runtime exporting KV to llama.cpp, **or** extend
  llama.cpp-xdna2's ggml backend (NPU prefill) and pair it with ggml-vulkan decode in one
  process. See prior-art.md.
- **D2 (decide in M1):** weight format for the first model: Q4NX (Gemma 4 variant, open
  kernels exist) vs GGUF Q4_0/Q4_K (llama.cpp ecosystem, needs our own repack to bfp16).
- **D3 (decide in M3):** first target model. Candidates: Llama 3.2 1B (IRON reference exists,
  dense, simple) or Gemma 4 E2B (open FLM ports exist).
