# FastFlowLM formats (Q4NX, model packaging)

## What is open vs closed

| Part | Status | Where |
|---|---|---|
| Runtime: server, CLI, tokenizer, sampler, chat templates, AutoModel wrappers | Open, MIT | `FastFlowLM/src/` (LICENSE_RUNTIME.txt) |
| NPU engine libs `lib<model>_npu.so` and `.xclbin` kernels | **Closed**, "FastFlowLM NPU Binary License": use and redistribute unmodified only | `flm/lib64/`, `src/xclbins/` (LICENSE_BINARY.txt) |
| Gemma 4 NPU overlays (mm, dequant, prefill attention, layer, lm_head) | **Open ports** in IRON, Apache-2.0 | `amd/IRON iron/operators/flm/` |
| Q4NX dequant kernel | Open | mlir-aie `aie_kernels/quant/q4nx_dequant.cc`, Python factory `aie.iron.kernels.quant.q4nx_dequant` |

Do not copy closed binaries into this repo. Referencing an installed `flm` for comparison is fine.

## Model package (verified on `~/.config/flm/models/Qwen3.5-4B-NPU2/`)

```
config.json             HF config
tokenizer.json, tokenizer_config.json, chat_template.jinja
model.q4nx              weights
vision_weight.q4nx      (VL models only)
```

`model.q4nx` is a **safetensors** container: 8-byte little-endian header length, JSON header,
then data. No `__metadata__`. Qwen3.5-4B: 474 tensors — 249 `I8` (quantized), 177 `BF16`
(norms, small vectors), 48 `F32`.

Quantized tensors are stored as a block grid `[out/32, in/256, BLOCK_BYTES]` (dtype `I8`
used as raw bytes). Examples (Qwen3.5-4B, hidden 2560, intermediate 9216):

| Tensor | shape | → logical (out, in) |
|---|---|---|
| `mlp.gate_proj.weight` | `[288, 10, 4736]` | (9216, 2560) |
| `mlp.down_proj.weight` | `[80, 36, 4736]` | (2560, 9216) |
| `self_attn.q_proj.weight` | `[256, 10, 4736]` | (8192, 2560) |
| `lm_head.weight` | `[7760, 10, 8704]` | (248320, 2560) — different block size |

## Q4NX block layout — Gemma 4 variant (verified from source)

Source: `aie.iron.kernels.quant` (`Q4NX_*` constants, `q4nx_unpack`, docstrings) in
mlir-aie 1.4.6.dev15, and IRON `iron/operators/flm/q4nx.py`.

- Block = `M_TILE=32` out-features (n) × `K_TILE=256` in-features (k), `GROUP=32`.
- `BLOCK_BYTES = 32*256/2 + 4*32*256/32 = 4096 + 1024 = 5120` → 5.0 bits/weight.
- Byte order inside a block:
  1. bf16 **scales**, little-endian, indexed `[k // GROUP, n]` (8 × 32 = 256 values, 512 B)
  2. bf16 **minima**, same indexing (512 B)
  3. 4-bit **codes**, unsigned, low nibble first, indexed `[n // 16, k, n % 16]` (4096 B)
- Weight: `w[k, n] = mins[k//32, n] + scales[k//32, n] * code[k, n]` (asymmetric, zero = min).
- Grid of blocks laid out `(rows // 32, cols // 256)`; `q4nx.unpack(grid)` returns codes
  `(rows, cols)` and scales/mins `(rows, cols // 32)`; `q4nx.dequantize(grid)` → float64.
- NPU path: `flm.Dequant` turns blocks into **bfp16ebs8** B-operand storage
  (`32*256*9/8 = 9216` B per block: 8 mantissas share one 8-bit exponent), then
  `flm.ProjectionGEMM(bfp16_b=True)` multiplies bf16 activations by it.

## Q4NX — Qwen3.5 variant: `UNVERIFIED`

The Qwen3.5-4B package uses **4736-byte** blocks for linear weights and 8704-byte blocks for
`lm_head`, not 5120. 4736 − 4096 = 640 B of parameters per 32×256 block, which does not
match the Gemma 4 scale+min layout (1024 B). Q4NX is a per-model family of layouts; do not
assume the Gemma 4 layout for other models. Reverse-engineering the Qwen variant is out of
scope until a task needs it (it would need FLM runtime source or the engine binary).
