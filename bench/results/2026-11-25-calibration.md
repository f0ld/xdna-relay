# T-038: online calibration of placement costs, validated on hardware (2026-11-25)

`bench/calibration_run.py`, 32 rounds: A streams 120 tokens; B (82–1718 tokens, cycled)
arrives 0.5 s after A's first token. Server: Qwen3-8B, batched (chunk 256), NPU packed,
`--placement auto`, started with a **deliberately wrong prior** (`--preset llama3.2-3b`: iGPU
~1.8x and NPU ~1.3x/block faster than this pair). Platform `balanced`, on AC, NPU pmode
`performance`; no NPU TDR in either run. A alone ~15.5 tok/s.

## Run 1 (first calibrator): converged, but to some wrong decisions
Estimates went from −53/−57% to within ±12% after ~4 rounds; decisions: 279 and 398 tokens
iGPU (measured best: split / NPU), 968–1718 split. Two model errors visible in the fits:
- one iGPU line for all sizes: its intercept (0.37 s, from long prefills) was charged to the
  1-token / small tail every NPU or split option leaves the iGPU (really ~1 decode step);
- the prior had one NPU curve, so the calibrator kept one NPU fit, mixing the server's short
  (≤640 rows, streamed B) and long (resident B) prompt versions.
Fix: own fit for iGPU parts ≤ 64 tokens; NPU fits always split at 640 rows.

## Run 2 (fixed calibrator)
| round(s) | B tokens | choice | est vs measured TTFT | A tok/s during B |
|---|---|---|---|---|
| 0–4 | 82–1717 | mixed | +3 … −53% | |
| 12–31 | 82–83 | iGPU | +38 … +63% (0.15–0.2 s) | 4.7–4.8 |
| 13–29 | 138–139 | iGPU | +6 … +25% | 3.0–4.3 |
| 15–31 | 279 | split 256+23 | −8 … −11% | 9.5–10.5 |
| 19–27 | 398 | iGPU | −1 … 0% | 2.2 |
| 12–28 | 967–968 | NPU | 0 … +1% | 10.3–10.4 |
| 14–30 | 1327–1328 | split 1280+48 | −1 … −2% | 9.4–9.7 |
| 18–26 | 1717–1718 | NPU | +6 … +8% | 9.7–10.1 |
Final fits: iGPU bulk (decoding) 0.375 s + 2.49 ms/token; iGPU small tails 0.226 s + 1.21
ms/token; NPU short 0.346 s + 0.643 s/block, long 0.167 s + 0.653 s/block (incl. handoff);
keep iGPU 0.29, keep NPU 0.70; plain decode step 64 ms.

Against the best measured placements (2026-11-25-batched-decode.md): 82/139 iGPU ✓, 279 split
✓, 968 NPU ✓, 1718 NPU ✓; **398 iGPU ✗** (measured NPU: TTFT 1.68 s with A at 10.1 vs iGPU
1.39 s with A at 2.2; TTFT + A lost 2.28 vs ~2.58). A near-tie (~4% in the model) decided by
keep_iGPU: from step times 0.29, from the client 2.2/15.5 = 0.14. The step-time estimate
misses what A loses outside the stepper's steps (B's request handling in Python: template,
tokenize, prepare, all under the GIL with the stepper). Open: measure keep from decoders'
token timestamps instead of step times.

## T-039: closing the remaining misses (runs 3–8, same wrong prior, 32–40 rounds each, on AC)
Each run exposed one modelling error; each fix has a T-038 test (14 tests now):
| run | what it showed | fix |
|---|---|---|
| 3 | per-step keep (0.29) ≠ client's (0.14): steps with 1-token tails counted as "prefill" | keep from whole request windows (arrival → first token), per placement |
| 4 | one keep fraction cannot fit: iGPU keep 0.31–0.46 at 82–139 tok, 0.13–0.19 at 398–1327; client gap was my script's A-alone (15.5 vs 14.7 tok/s) | others' **lost decode-seconds** fitted per device and size; windows start at HTTP arrival |
| 5 | NPU at 2 blocks never tried (pessimistic extrapolation, 1.71 vs ~1.35 s) | bounded exploration: options within 20% never observed at a size are tried (≤2×) |
| 6 | NPU option charged its 1-token tail's loss twice (window already covers it); then short prompts went to the NPU: the iGPU line (intercept 0.44 s) misfits 83 tokens (0.41 s) | no double count; split adds only the tail beyond one token |
| 7 | batched iGPU cost is not linear at the short end | per-bin residual corrections on every fit (128-token / one-block bins) |
| 8 | split's loss composed (0.42 s at 398) vs measured 0.81 s: its tail is a real prefill chunk | split's own loss fit from its windows |

## Run 8 (final calibrator), rounds 16–39 (converged; exploration stopped after round 20)
| B tokens | choice | est vs measured TTFT | A tok/s during | measured best |
|---|---|---|---|---|
| 83 / 139 | iGPU | −13 … +3% | 3.2–7.1 | iGPU ✓ |
| 279 | split 256+23 | −11 … −1% | 9.2–10.6 | split ✓ |
| 398 | NPU | +9 … +12% | 9.8 | NPU ✓ |
| 968 | NPU | +2 … +4% | 10.4–10.6 | NPU ✓ |
| 1328 | split 1280+48 | −5 … −4% | 9.7–9.8 | split ✓ (TTFT+loss 4.83 vs NPU 4.94) |
| 1718 | NPU | +3 … +5% | 10.3–10.5 | NPU ✓ |
Round 39 (279 tokens, 1.84 s TTFT) is an unexplained outlier. Starting from a preset for another
model, the server reaches the measured-best placement at every size within ~20 requests.
