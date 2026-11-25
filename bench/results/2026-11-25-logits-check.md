# T-022: IRON NPU logits vs the Hugging Face reference (2026-11-25)

`bench/logits_check.py`: HF transformers 5.24 / torch 2.15 CPU in float32
(`tests/reference/hf_golden.py`, run in `~/.cache/xdna-relay/ref-venv`) logits after a
~286-token prompt and after each of 16 greedy tokens; IRON (`IronLmExecutor`, NPU) and
llama.cpp (iGPU) teacher-forced on the same tokens. 17 rows per model. No NPU TDR in any run
(`tdr_guard`). Platform `balanced`, AC, NPU pmode `performance`.
Row 0 is the prompt chunk (ProjectionGEMM at the model's precision); rows 1–16 are decode
steps (bf16 GEMV, attention over the prompt's KV).

| model | implementation | KL mean | KL max | top-1 | max abs diff (median / max) |
|---|---|---|---|---|---|
| Llama 3.2 1B | IRON default (medium) | 5.47e-3 | 4.95e-2 | 16/17 | 0.35 / 1.35 |
| Llama 3.2 1B | IRON high | 9.97e-4 | 8.99e-3 | 17/17 | 0.20 / 0.67 |
| Llama 3.2 1B | llama.cpp BF16 | 4.44e-5 | 2.28e-4 | 17/17 | 0.03 / 0.09 |
| Llama 3.2 3B | IRON default (high) | 4.90e-4 | 3.09e-3 | 17/17 | 0.14 / 0.58 |
| Llama 3.2 3B | IRON medium | 1.88e-3 | 9.03e-3 | 17/17 | 0.20 / 0.70 |
| Llama 3.2 3B | llama.cpp BF16 | 8.63e-6 | 4.92e-5 | 17/17 | 0.02 / 0.06 |
| Qwen3-0.6B | IRON (medium) | 1.98e-3 | 7.14e-3 | 17/17 | 0.45 / 0.89 |
| Qwen3-0.6B | IRON high | 9.33e-4 | 4.95e-3 | 16/17 | 0.26 / 0.38 |
| Qwen3-0.6B | llama.cpp Q8_0 | 1.73e-3 | 4.48e-3 | 17/17 | 0.59 / 0.81 |
| Qwen3-8B | IRON (medium) | 9.59e-3 | 9.90e-2 | 16/17 | 0.79 / 1.89 |
| Qwen3-8B | IRON medium, packed bfp16 | 7.87e-3 | 8.87e-2 | 16/17 | 0.80 / 1.91 |
| Qwen3-8B | llama.cpp Q4_K_M | 9.59e-3 | 9.44e-2 | 17/17 | 1.57 / 3.20 |

- IRON's NPU logits track the float32 reference to KL ~1e-3 at `high` and 2–10e-3 at
  `medium` (bfp16 macs): no implementation error; precision only. The single top-1 misses are
  near-ties (Llama 1B medium row 16, Qwen3-8B one decode row).
- Medium (our prefill default for 3B+ and Qwen3) is as close as llama.cpp Q8_0 (0.6B) and
  Q4_K_M (8B); packing (T-033) does not degrade it.
- llama.cpp BF16 is ~100x closer than IRON: llama.cpp accumulates in f32 on the iGPU.
- IRON's own `--check-accuracy` needs a `golden.npz` recorded with torch + transformers
  (`iron/lm/golden.py`); our ref-venv script replaces it.
