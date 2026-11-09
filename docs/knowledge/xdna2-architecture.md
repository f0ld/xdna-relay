# XDNA2 (AIE2P) on Strix Point

## Array geometry — verified

Source: `aie.dialects.aie.get_target_model(AIEDevice.npu2)` in mlir-aie 1.4.6.dev15, and
`xrt-smi examine` (`NPU Strix`, `aie2p`, topology `6x8`).

| Row | Tile type | Per tile |
|---|---|---|
| 0 | Shim (NoC) — DMA to/from host DDR | 16 BDs |
| 1 | Memory tile (L2) | **512 KiB**, 48 BDs, 64 locks |
| 2–5 | Compute (core) tiles (L1) | **64 KiB** data memory in 4 banks, 16 BDs, 16 locks, default stack 1 KiB |

- 8 columns × 4 compute rows = **32 AIE cores**, 8 memtiles (4 MiB L2 total), 8 shim tiles.
- DMA task queue depth 4. Partial-array devices `npu2_1col` … `npu2_7col` exist in mlir-aie.
- FLM and IRON both use the full 4×8 grid for GEMM.

## Data movement

- Host buffers (XRT BOs in DDR) ↔ shim DMA ↔ memtile (L2) ↔ core L1, via ObjectFifos.
- Memtiles can broadcast/split/join; IRON's GEMM: A broadcast along compute rows, B down
  columns, C joined through the memtile.
- Neighbouring cores share memory (north/south/east/west affinity); cores also have
  a cascade stream for accumulator forwarding.

## Compute (from xdna-engine docs/aie2p-architecture-and-roofline.md — `UNVERIFIED` here)

- Vector register 512-bit (32 bf16 / 64 int8 / 128 int4 lanes); accumulator 2048-bit.
- Peak MAC/cycle/core: bf16 128 (emulated), **bfp16 512 (native)**, int8 1024,
  int8×int4 1024 (`4x16x16` mmul).
- mmul shapes: bf16 `8x8x8`/`4x8x8`; bfp16 `8x8x8`/`8x8x16`; int8 `8x8x8`/`4x8x8`.
- → This is why FLM/IRON's Q4NX path dequantizes to **bfp16ebs8** (block FP: 8 mantissas share
  one exponent) and runs bf16 × bfp16 GEMM, not int4 math.

## Bandwidth (third-party measurements — `UNVERIFIED` on this machine)

- NPU DRAM read ~52.7 GB/s, fabric ceiling ~62.7 GB/s (xdna-engine, Krackan).
- CPU reads ~70 GB/s vs NPU ~53 GB/s (llama.cpp-xdna2, Ryzen AI 9 365) → why decode
  (bandwidth-bound) is not a natural fit for the NPU; prefill (compute-bound) is.
- Measure on our machine in M2 (bench task) before relying on these.

## Driver/runtime behaviour

- `amdxdna` kills a command after ~2 × `tdr_timeout_ms` without progress (default 2000 ms →
  ~4 s) with `ERT_CMD_STATE_TIMEOUT` (IRON docs/installation.md §2).
- XRT locks a 64 MiB buffer when opening the NPU (needs memlock; ours is unlimited).
- NPU2 runs a whole IRON graph as one ELF; xclbin count is limited (FLM budgets 16 per model).
