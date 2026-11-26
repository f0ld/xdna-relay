# T-046: Qwen3.6-35B-A3B experts on the NPU (IRON), gate 1

Machine: Ryzen AI 9 HX 370, AC, platform `balanced`, NPU pmode `performance`. Expert shapes:
gate+up K=2048 N=1024, down K=512 N=2048, packed bfp16 B (`bfp16_b=True`, Precision.MEDIUM).
Gate 1 (docs/knowledge/qwen35moe-iron-plan.md): 40 layers of experts on the NPU >= 1.5x
faster than the iGPU's (0.93 ms/token: T=2048 -> 1.9 s).

## 1. One ProjectionGEMM step per expert (`bench/moe_expert_probe.py`)
Per-expert slope of a runlist of E experts in one full ELF:

| rows per expert (M) | gate/up | down |
|---|---|---|
| 256 (1 unit), valid 1 or 256 | 125-135 us | 90-103 us |
| 512 | 162 us | 120 us |
| 1024 | 330 us | 244 us |

~80 us of fixed work per step whatever its rows (a unit is 256 rows: M_TILE 64 x 4 core
rows; the row bound only skips whole units): 256 x 2 x 40 steps -> ~2.1 s. Fails.

## 2. GroupedGEMM: all experts in one call (`src/xdna_relay/backends/npu/grouped_gemm.py`)
ProjectionGEMM's streamed-B transfer with the unit dimension stepping one expert's packed
weight (stride 0 in IRON). Correct per group (`scripts/test.sh T-046`, 6/6; a wrong stride
fails). A unit's rows are split across the two core-row halves (`unit_rows`).

256 experts, one call each:

| variant | gate/up | down | per layer | 40 layers |
|---|---|---|---|---|
| 4 core rows, tile_n 64 | 22.9 ms | 17.7 ms | 40.6 | 1.62 s |
| 4 core rows, tile_n 128 | 17.0 | 14.0 | 31.0 | 1.24 s |
| 2 core rows (128-row units), tile_n 128 | 15.8 | 9.2 | 25.0 | 1.00 s |

All at 50-59 GB/s, the NPU's DRAM ceiling: bytes = packed weights (906 MB/layer, the floor:
~17 ms) + A/C rows including each unit's padding. tile_n 128 reads A once per 1024 columns
instead of twice; 2 core rows halve the padding (needed a smaller B pool: one more A buffer
per memtile than IRON's budget counts).

## 3. Real routing (`bench/native/moe_routing.cpp`: cb_eval on `ffn_moe_argsort`, top 8)
Code (8192 tokens) / prose (5740), every layer but the last (it runs on output rows only);
GPU and CPU backends agree exactly.

| T | experts used | busiest expert | 128-row units needed | 17th busiest (cap. after 16 hot) |
|---|---|---|---|---|
| 512 | 185-207 | 257-287 (mean 16) | 190-214 | |
| 1024 | 206-227 | 476-554 | 226-249 | 104-114 |
| 2048 | 222-240 | 906-1083 (mean 64) | 287-303 | 206-220 (worst 312) |
| 4096 | 236-248 | 1766-2104 | 411-414 | 402-416 |

Heavy-tailed: the top 8 experts take ~30% of a layer's rows; nearly all experts get some.

## Gate 1 verdict
- Uniform capacity per expert (what GroupedGEMM does) cannot hold this routing: capacity
  256-320 for the tail plus ~16 hot-expert calls -> ~42 ms/layer at T=2048 (~1.1x). **Fails.**
- A layout with each expert's own unit count (units128 ~ 300 at T=2048) would cost about
  (300/256) x 25 ms + SiLU*up ~2.5 ms ~ 32 ms/layer -> 1.28 s (**~1.5x**), 4k ~2.1x, 8k ~2.7x,
  if one call can take a per-unit weight offset table (one B shim task per unit per column:
  ~2-5k tasks a call; command-processor cost UNVERIFIED). Not built.
- iGPU observation: its expert GEMMs run at ~2 TFLOPS vs 6.5 for dense GEMMs; with the
  routing this skewed (hot experts with ~1000 rows of 2048), a MoE kernel that runs hot
  experts as dense GEMMs could gain on the iGPU directly.

## 4. Experiment A: each expert gets the units its tokens need (unit -> expert table)
`GroupedGEMM(unit_groups=...)`: one B shim transfer per unit per column, each at its
expert's weights (`_split_blocks`: A fills, arming, C drains, then the B fills, so a full B
lane queue never stands ahead of what the cores need). Correct for repeated, skipped and
reordered experts (`scripts/test.sh T-046`, 9/9). Tables from real routing
(`MOE_DUMP=bench/results/moe_counts_code.csv bench/native/moe_routing`, layer 13 of the code
text; units of 128 rows, 2 core rows, tile_n 128):

| T | units | gate/up | down | per layer |
|---|---|---|---|---|
| 2048 | 306 | 19.35 ms | 11.22 ms | 30.6 ms |
| 4096 | 420 (gate/up as 2 calls of 210) | 25.62 | 15.25 | 40.9 |

~61-63 us (gate/up) + ~36 us (down) per unit, the same as the uniform layout: thousands of B
transfers per call cost the command processor nothing measurable.

NPU timeouts (TDR, 10:55 and 10:56): gate/up with 418 units (836 B ring passes per column
block) and 420 units (560 passes) in ONE call. 306 units (408 passes) ran. Not retried
further; `GroupedGEMM` refuses a table over 512 passes (`MAX_TABLE_PASSES`, 2 queue pushes of
256; cause UNVERIFIED: likely the 3rd memtile push waits on the 1st, whose B fills come
after it in `_split_blocks`) and the probe splits larger tables (`max_units`: 384 at K=2048).

40-layer projection from every layer's real unit count (per unit 97 us + SiLU*up ~7.5 us
estimated + 0.6 ms per call):

| T | mean units | NPU experts | iGPU experts | speedup |
|---|---|---|---|---|
| 1024 | 227 | 1.00 s | 0.95 s | 0.95x |
| 2048 | 303 | 1.32 s | 1.90 s | 1.45x |
| 4096 | 418 | 1.82 s | 3.81 s | 2.09x |
| 8192 | 659 | 2.84 s | 7.62 s | 2.69x |

Gate 1 (>= 1.5x at T=2048): 1.45x, a marginal miss at 2k; passes from ~2.2k tokens on, 2.1x
at 4k, 2.7x at 8k. Still open for production: the table is static here (a rebuild per
routing, ~5 s); a per-call table needs one per-call offset word per unit
(`offset_parameter` takes a single `Value`; UNVERIFIED whether a `per_call(count=...)`
vector element can be one).
