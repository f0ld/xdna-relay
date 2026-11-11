# 0005 — Weights: GGUF is the source format; Q4_0/Q4_K repack to Q4NX-Gemma4 on the NPU

Status: Accepted (2026-11-11). Resolves open decision **D2** in `docs/ARCHITECTURE.md`.

## Context
D2 asked which weight format the first model uses: FastFlowLM's Q4NX (Gemma 4 variant, open
NPU kernels in IRON) or GGUF Q4_0/Q4_K (llama.cpp ecosystem). Facts, all checked on this machine:

- **Open NPU kernels exist only for the Q4NX-Gemma4 block layout**: `iron.operators.flm.Dequant`
  (q4nx → bfp16ebs8 B operand) + `flm.ProjectionGEMM(bfp16_b=True)` (flm-formats.md, prior-art.md).
- **Q4NX files are per-model and come only from FLM packages.** The local FLM models are
  Qwen3.5/3.6, which use a different, `UNVERIFIED` Q4NX variant (4736-byte blocks). No open kernel
  reads it. Using Q4NX as the source would tie D3 to Gemma 4.
- **M5 (hybrid) decodes with llama.cpp, which reads GGUF.** If prefill used a different
  quantization (FLM's Q4NX) than decode (GGUF), the KV cache handed over would come from a
  slightly different model than the one decoding.
- **The block math matches** (llama.cpp `ggml-common.h` and gguf-py `quants.py` at llama.cpp
  `6b0f75b`, read 2026-11-11):
  - Q4_0: 32 weights along K, f16 `d`, `w = d·(q − 8)`, q ∈ 0..15.
  - Q4_K: 256-weight super-block along K = 8 groups of 32, f16 `d`, `dmin`, 6-bit `sc[j]`, `m[j]`;
    `w = (d·sc[j])·q − dmin·m[j]`, q ∈ 0..15.
  - Q4NX-Gemma4: 32 (n) × 256 (k) block, groups of 32 along k per n, bf16 `scale`, `min`;
    `w = min + scale·code`, code ∈ 0..15.
  - So `scale = d·sc[j]`, `min = −dmin·m[j]` (Q4_K) and `scale = d`, `min = −8d` (Q4_0), with
    codes copied as-is. The Q4_K super-block (256 along K) equals Q4NX's `K_TILE`. In float32
    these are exact: `scale·code + min` reproduces gguf-py's dequantization bit for bit (T-003 tests).
    The **only loss** is rounding scale and min to bf16 for Q4NX. Measured on
    `Qwen3.5-9B-Q4_K_M` `blk.0.ffn_gate.weight` (12288×4096): max error 0.088 of one quantization
    step, relative RMS weight error 5.0e-3. `min` dominates because |min| ≫ scale. This is small
    against 4-bit quantization noise (±0.5 step) but **not bit-exact**.
- **A real "Q4_K_M" file is not all Q4_K.** Measured on `Qwen3.5-9B-Q4_K_M.gguf` (by parameter
  count): Q4_K 65.8%, Q6_K 20.6% (`ffn_down`, some `attn_v`, `output`), Q5_K 13.5%, Q8_0 0.1%.
  Q5_K/Q6_K codes do not fit in 4 bits.

## Decision
1. **GGUF is the only on-disk weight source.** The same file serves the NPU prefill and the
   llama.cpp decode (M5). No FLM Q4NX files are read.
2. **Per-tensor NPU format is chosen by the backend's `prepare()` (ADR 0001) from the GGUF type:**
   - `Q4_0`, `Q4_K` → repack on the host into the **Q4NX-Gemma4 block layout** (5.0 bits/weight)
     → `q4_gemm` npu backend (`flm.Dequant` + `flm.ProjectionGEMM`).
   - every other type (`Q5_K`, `Q6_K`, `Q8_0`, `F16`, `BF16`, `F32`) → dequantize on the host to
     bf16 → existing `gemm_bf16` npu backend (T-004). This fallback costs memory and bandwidth
     (16 bits/weight); a bfp16 repack for these types is an M2 optimization, not M1.
   - We never requantize a tensor to fewer bits than GGUF stores (that would change the model).
3. **Q4NX-Gemma4 is our internal NPU layout, not an interchange format.** Our code packs it
   (`src/xdna_relay/formats/q4nx.py`) and validates it against IRON's `iron.operators.flm.q4nx`
   unpack/dequantize.
4. **gguf-py (llama.cpp, MIT, pinned `gguf==0.19.0` in `requirements-npu.txt`) is a runtime
   dependency for the container (`GGUFReader`) and for dequantizing fallback types
   (`gguf.quants.dequantize`).** Our own code unpacks Q4_0/Q4_K into codes, scale and min (the
   repack needs the fields, not floats). It is tested bit-exact against `gguf.quants.dequantize`.
   (Amended 2026-11-11, same day: the first draft had gguf-py as a test-only oracle.
   Reimplementing every GGUF dequant would duplicate it without adding an independent check.)

## Consequences
- D3 (first model) is no longer constrained by weight format: any model with a GGUF works.
- M1 order: T-003 GGUF reader + reference dequant → T-018 Q4NX pack + GGUF→Q4NX repack →
  T-005 `q4_gemm` on the NPU. T-005's weight input layout is `q4nx_gemma4`.
- The NPU prefill model differs from the GGUF model by bf16 rounding of Q4 scale/min (above).
  T-005 must measure it (`q4_gemm` npu vs GGUF-exact dequant + float64 GEMM). If the logits
  drift in M3/M4 is too large, revisit (e.g. f32 scales via a modified dequant kernel).
- `q4_gemm` needs its own tolerance. `flm.ProjectionGEMM` multiplies in bfp16, and IRON's
  `GEMM` at `MEDIUM` already misses the `gemm_bf16` contract (pitfalls.md).
- In a Q4_K_M file about a third of the weights take the bf16 fallback until M2 adds a
  bfp16 path. Picking Q4_0/"pure" quantizations for the first model reduces that.
- `UNVERIFIED`: that `flm.Dequant` + `ProjectionGEMM` accept every K/N our models need.
  K % 256 == 0 holds for Q4_K tensors by construction (super-block size). N % 32 == 0 and Q4_0's
  K % 256 == 0 do not; padding rules go in the T-018/T-005 specs.
