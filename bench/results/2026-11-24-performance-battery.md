# Contention + power, platform profile `performance`, ON BATTERY (AC 0, 86% at start)

Same setup as `2026-11-24-baseline.md` (Qwen3.5-4B; FLM v1.0.9 `--pmode performance`;
llama.cpp 6b0f75b Vulkan). Not yet measured on AC power — battery limits may differ.

## Contention matrix (`bench/contention.py`)
    label,aggressor,aggressor_tps,victim,victim_tps
    qwen3.5-4b-performance-battery,none,0.0,npu-prefill,497.27
    qwen3.5-4b-performance-battery,none,0.0,npu-decode,14.78
    qwen3.5-4b-performance-battery,none,0.0,igpu-prefill,630.09
    qwen3.5-4b-performance-battery,none,0.0,igpu-decode,21.24
    qwen3.5-4b-performance-battery,npu-prefill,574.1,igpu-prefill,572.21
    qwen3.5-4b-performance-battery,npu-prefill,550.5,igpu-decode,17.32
    qwen3.5-4b-performance-battery,npu-decode,14.2,igpu-prefill,458.45
    qwen3.5-4b-performance-battery,npu-decode,12.9,igpu-decode,12.78
    qwen3.5-4b-performance-battery,igpu-prefill,605.1,npu-prefill,339.69
    qwen3.5-4b-performance-battery,igpu-prefill,548.9,npu-decode,11.50
    qwen3.5-4b-performance-battery,igpu-decode,19.7,npu-prefill,324.29
    qwen3.5-4b-performance-battery,igpu-decode,18.3,npu-decode,11.90

## Power (`bench/sample_power.sh`)
| workload | PPT avg | sclk avg | result | low-power for comparison |
|---|---|---|---|---|
| iGPU decode alone | 28.7 W | 2577 MHz | tg 21.2 | 13.2 W, 1590 MHz, tg 20.8 |
| iGPU prefill alone | 28.4 W | 2171 MHz | pp1024 652 | – |
| NPU prefill alone | 17.7 W | idle | 559–571 tok/s | 12.6 W, ~485 tok/s |
| NPU decode alone | 13.0 W | idle | 15.0 tok/s | 11.8 W, 15.1 tok/s |
| iGPU decode + NPU prefill | 31.5 W | 2286 MHz | tg 18.1 (−15%) | 14.9 W, 1143 MHz, tg 15.3 (−26%) |
| iGPU decode + NPU decode | 31.7 W | 2470 MHz | tg 13.2 (−38%) | 14.7 W, 1354 MHz, tg 13.4 (−36%) |

## Reading
- Decode vs decode: ~−37% in both profiles, iGPU clock not throttled in performance →
  **DRAM bandwidth** bound. Splitting decode across devices gains ~20% at best.
- Prefill vs prefill: −32% (low-power) → −9% (performance) → under low-power that was the
  **package power cap** (~15 W).
- NPU prefill next to iGPU decode: −15% (performance) / −24% (low-power) on iGPU decode.
- Efficiency (tok/s per W of PPT): iGPU decode 1.58 low-power vs 0.74 performance (2x power
  for +3% speed); prefill NPU 32–38 vs iGPU 23 → the NPU is the most efficient prefill device.
