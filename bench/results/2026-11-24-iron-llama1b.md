# T-022 — IRON Llama 3.2 1B on the NPU vs FLM and llama.cpp (same model)

Platform profile `performance`, on battery. Weights: `unsloth/Llama-3.2-1B-Instruct` (bf16
safetensors) for IRON; FLM `llama3.2:1b` (Q4NX); `Llama-3.2-1B-Instruct-Q4_K_M.gguf` for
llama.cpp (patched clone, Vulkan). IRON driven by `bench/iron_llama_probe.py`
(`Session.logits` per step, full ELF, max_seq_len 2048).

| path | prefill | decode |
|---|---|---|
| NPU IRON bf16 | 968 tok/s (481 tok) | 13.4 tok/s (74.7 ms/step, p90 75.8) |
| NPU FLM Q4NX | 945 (466 tok) / 1802 (1711 tok) | 59–64 tok/s |
| iGPU llama.cpp Q4_K_M | pp512 2523 | tg64 86 |

IRON rollback: `Session.forget(n-4)` then 1 token = 73.9 ms (one decode step); 3 tokens at
once = 276.9 ms (goes through the 2048-row prompt-chunk graph).

Reading: IRON decode moves ~2.47 GB bf16 per step → ~33 GB/s, at the NPU's measured DRAM
ceiling. FLM's Q4 is 4.7x faster on the same model and device. This is the measured
bottleneck that opens T-028/T-029 (Q4 weights + Q4 GEMV in the IRON decode graph).
