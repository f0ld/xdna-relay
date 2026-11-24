# T-025 — Where should the draft model run? (Qwen3-8B target)

Platform profile `performance`, **on AC**. Target `Qwen3-8B-Q4_K_M.gguf` (iGPU, patched
llama.cpp clone), draft `Qwen3-0.6B-Q8_0.gguf` (iGPU) or FLM `qwen3:0.6b` (NPU).

## FLM as an NPU draft source: not viable
- Every request costs ~480 ms wall, ~415 ms TTFT, independent of draft length (4 tokens).
- Prefix reuse only across chat turns (`cached_tokens` > 0), and TTFT stays ~415 ms even then
  → a fixed per-request cost (likely prefill/decode overlay switch). A draft round on the
  iGPU is ~116 ms in total.

## Same-device speculation (iGPU only), greedy, 256 tokens
| config | code | explain | rewrite | story |
|---|---|---|---|---|
| no spec | 15.65 | 15.61 | 15.54 | 15.62 |
| draft n=3 | **27.50** (.74) | 19.48 (.43) | **28.36** (.80) | 17.67 (.36) |
| draft n=5 | 27.17 (.62) | 16.47 (.31) | 29.44 (.70) | 15.37 (.27) |
| draft n=8 | 17.84 (.46) | 9.36 (.18) | 21.79 (.59) | 8.75 (.16) |

Outputs: code/rewrite identical to no-spec; explain/story diverge mid-text (batched-verify
float near-ties, as with Qwen3.5).

## Contention from an NPU draft-class load
| | 8B decode on iGPU |
|---|---|
| alone | 15.9 tok/s |
| FLM qwen3:0.6b decoding continuously on the NPU (106 tok/s alone, ~77 under load) | 11.0 tok/s (−31%) |

## Per-round accounting (code, n=3)
Round = 3.19 tokens × 36.4 ms = 116 ms: draft on iGPU ~42 ms (0.6B at 95.8 tok/s × 3 + draft
re-processing the verify batch), verify ~69 ms.
- NPU draft, sequential: 38 ms draft + 69 ms verify ≈ 107 ms → ~4% better.
- NPU draft, pipelined (draft r+1 during verify r): verify slowed by NPU traffic (up to −31%
  while active), redraft on any rejection → estimated 0–10% better.
Conclusion: on this SoC, decode-side cross-device speculation is near break-even; the shared
DRAM bandwidth eats the parallelism. Same-device speculation (+13–82%) is the lever.
