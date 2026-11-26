# Qwen3.6-35B-A3B: iGPU per-op profile, prefill ubatch sweep, T-044 measurements

Machine: Ryzen AI 9 HX 370, AC (online 1), platform profile `balanced`. Model
`Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf` (21.1 GiB). llama.cpp fork 6b0f75b, Vulkan, fa on.

## Per-op profile (`GGML_VK_PERF_LOGGER=1 llama-bench`; ops serialised, sums ≈ wall time)

Decode, one token (tg8 19.55 tok/s; one graph's ops 46.9 ms):

| op | ms/token | note |
|---|---|---|
| Q8_0 MUL_MAT_VEC (attn_qkv 8192, attn_gate/ssm_out 4096, attn_q, out, shexp) | 21.3 | 75–80 GB/s |
| experts MUL_MAT_ID_VEC (q4_K gate/up, q5_K/q6_K down) | 9.2 | 8 experts |
| LM head q6_K 248320×2048 | 5.2 | 80 GB/s |
| GET_ROWS + CPY (recurrent state in/out, embedding) | 3.7 | S = 2 MiB per layer |
| GATED_DELTA_NET | 2.0 | |
| router/beta/alpha f32 GEMV, norms, elementwise, conv, attention | ~4.6 | |

Prefill, 512 tokens (pp512 491 tok/s; ops 1036 ms):

| group | ms | % |
|---|---|---|
| experts MUL_MAT_ID (1.7–2.4 TFLOPS) | 476 | 46 |
| dense MUL_MAT (Q8_0 6.5–6.9 TFLOPS; F32 router 2.3) | 251 | 24 |
| elementwise / norms | 162 | 16 |
| CONCAT (GDN conv state) | 65 | 6 |
| GATED_DELTA_NET | 47 | 5 |
| SSM_CONV | 25 | 2 |
| FLASH_ATTN_EXT | 9 | 1 |

## Prefill vs ubatch (pp2048, `-b 2048 -ub ...`)

| ubatch | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|
| tok/s | 305 | 394 | **481** | 460 | 431 |

The server's batched prefill used 256-token chunks. T-044: a prompt prefilling with no
sequence decoding takes the whole n_batch (512). Engine TTFT, 2000-token prompt, 35B MTP,
2 slots: 5.90 / 5.90 s (256) → 4.78 / 4.73 s (512): **339 → 420 tok/s (+24%)**.

## MTP state in snapshots (T-044)

Fixed k=3, 160 tokens, 1 slot, ~250-token system prompt; warm = served from the prefix
cache (snapshot at prompt[:-1]).

| prompt | run | before T-044: acc, tok/s | T-044: acc, tok/s |
|---|---|---|---|
| CSV code | cold | 0.85, 31.3 | 0.85, 31.3 |
| CSV code | warm ×2 | 0.78 / 0.77, 29.5 / 29.1 | 0.83 / 0.86, 30.6 / 31.4 |
| hash map prose+code | cold | 0.65, 25.8 | 0.61, 25.1 |
| hash map prose+code | warm ×2 | 0.64 / 0.62, 25.6 / 25.2 | 0.64 / 0.63, 25.7 / 25.4 |

Before: a restore cleared the MTP context's KV and hidden state (drafts from zeros, −6% on
code). Now the blob carries them and warm = cold. Shim-level check: the 6-token drafts of a
restored sequence equal the live one's (T-044 test; the old shim gave different tokens).
