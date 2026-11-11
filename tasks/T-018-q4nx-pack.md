# T-018 — Q4NX-Gemma4 pack + GGUF Q4_0/Q4_K → Q4NX repack
Status: TODO
Milestone: M1
Depends on: T-003 (DONE), ADR 0005

## Goal
Turn a GGUF Q4_0/Q4_K weight (unpacked by T-003 into codes, scale, min) into the byte blob
that IRON's NPU dequant operator `DequantBFP` reads. T-005 feeds that blob to the NPU. This
needs three steps: zero-pad to the operator's alignment, pack 32×256 **blocks** (the Q4NX-Gemma4
layout, scale/min rounded to bf16), and order the blocks the way `DequantBFP` streams them.
Pure numpy on the CPU. No NPU code, and no `aie`/`iron` imports in `src/`.

## Files you may change
- `src/xdna_relay/formats/q4nx.py` (create)
- `docs/knowledge/pitfalls.md` (append only, if you hit something new)
- `PLAN.md` (only the T-018 status line), `handoff/T-018-iN.md`

## Files to read first
- `tests/kernels/test_t018_q4nx_pack.py`: the tests you must pass (do not edit)
- `src/xdna_relay/formats/gguf.py`: `Q4Unpacked`, `GGUFTensor`, `unpack_q4` (you reuse them)
- `.venv/lib64/python3.14/site-packages/mlir_aie/python/aie/iron/kernels/quant.py`: function
  `q4nx_unpack` (the exact inverse of your `pack_grid`; read its docstring)
- `.venv/lib64/python3.14/site-packages/iron/operators/flm/dequant/op.py`: method `DequantBFP.grid`
  (the exact inverse of your `grid_to_blob`)

## Terms
- `n` = out-feature (row of the PyTorch weight, `N` rows), `k` = in-feature (column, `K` columns).
- **block**: 32 rows × 256 columns of the weight = 5120 bytes.
- **grid**: `uint8` array `(N // 32, K // 256, 5120)`, block `[r, c]` covers rows `32r..32r+31`,
  columns `256c..256c+255`. This is what IRON's `iron.operators.flm.q4nx.unpack(grid)` reads.
- **blob**: 1-D `uint8`, the same blocks reordered for `DequantBFP` (see `grid_to_blob`).

## Interface / contract — `src/xdna_relay/formats/q4nx.py`
Imports: `numpy`, `ml_dtypes.bfloat16`, `dataclasses`, and from `.gguf`: `GGUFTensor`,
`Q4Unpacked`, `unpack_q4`. Nothing else (tested: no `aie`/`iron`/`pyxrt`).

```python
M_TILE, K_TILE, GROUP = 32, 256, 32
BLOCK_BYTES = 5120            # 512 B scales + 512 B mins + 4096 B codes
N_ALIGN, K_ALIGN = 64, 512    # DequantBFP needs N % 64 == 0 and K % 512 == 0

def pad_q4(u: Q4Unpacked, n_align: int = N_ALIGN, k_align: int = K_ALIGN) -> Q4Unpacked:
    """Zero-pad codes (N, K) up to multiples of (n_align, k_align), and scale/min
    (N, K // 32) to match. Padded entries are 0 (so padded weights are 0). Same dtypes
    (uint8 / float32 / float32). An already aligned input comes back with equal values."""

def pack_grid(u: Q4Unpacked) -> np.ndarray:
    """uint8 grid (N // 32, K // 256, 5120). ValueError if N % 32 or K % 256.
    Scale and min are rounded to bf16 round-to-nearest-even: x.astype(np.float32).astype(bfloat16)."""

def unpack_grid(grid: np.ndarray) -> Q4Unpacked:
    """Inverse of pack_grid: codes uint8 (N, K), scale/min float32 (N, K // 32), exact bf16 values."""

def grid_to_blob(grid: np.ndarray) -> np.ndarray:
    """1-D uint8 in DequantBFP's order. ValueError if grid rows (N // 32) or grid columns
    (K // 256) are odd, i.e. N % 64 or K % 512."""

def blob_to_grid(blob: np.ndarray, n: int, k: int) -> np.ndarray:
    """Inverse of grid_to_blob for an (n, k) weight (n % 64 == 0, k % 512 == 0)."""

@dataclass(frozen=True)
class Q4NXWeight:
    blob: np.ndarray   # 1-D uint8, n_pad * k_pad * 5120 // 8192 bytes
    n: int             # logical out_features (GGUF tensor shape[0])
    k: int             # logical in_features (GGUF tensor shape[1])
    n_pad: int         # n rounded up to N_ALIGN
    k_pad: int         # k rounded up to K_ALIGN

def from_gguf(t: GGUFTensor) -> Q4NXWeight:
    """unpack_q4(t) -> pad_q4 -> pack_grid -> grid_to_blob. ValueError (from unpack_q4) for non-Q4 types."""
```

### Byte layout of ONE block (what `pack_grid` writes; `q4nx_unpack` reads it back)
Inside a block, `n` = 0..31 (row within the block), `k` = 0..255 (column within the block),
`g = k // 32` = 0..7.
1. bytes `0 .. 511`: 256 bf16 **scales**, little-endian, value index `g * 32 + n`.
2. bytes `512 .. 1023`: 256 bf16 **mins**, same indexing.
3. bytes `1024 .. 5119`: 8192 4-bit **codes**. Nibble index `j = (n // 16) * 4096 + k * 16 + (n % 16)`;
   nibble `j` is in byte `1024 + j // 2`, the **low** nibble when `j` is even, the high one when odd.

Vectorized (`nr = N // 32`, `nc = K // 256`):
```python
def _sm_bytes(x):                         # x: float32 (N, K // 32), the scales or the mins
    y = x.reshape(nr, 32, nc, 8).transpose(0, 2, 3, 1)       # (nr, nc, g, n)
    y = np.ascontiguousarray(y.astype(np.float32).astype(bfloat16))
    return y.view(np.uint8).reshape(nr, nc, 512)
c = u.codes.reshape(nr, 2, 16, nc, 256).transpose(0, 3, 1, 4, 2).reshape(nr, nc, 8192)
qs = (c[..., 0::2] | (c[..., 1::2] << 4)).astype(np.uint8)  # (nr, nc, 4096)
grid = np.concatenate([_sm_bytes(u.scale), _sm_bytes(u.min), qs], axis=-1)
```
For `unpack_grid`, reverse each step. Turn the bf16 bits back into float32 with
`(np.ascontiguousarray(grid[..., :1024]).view(np.uint16).astype(np.uint32) << 16).view(np.float32)`,
then undo the transposes.

### Block order of the blob (what `grid_to_blob` writes; `DequantBFP.grid` reads it back)
```python
nr, nc, _ = grid.shape
blob = grid.reshape(nr // 2, 2, nc // 2, 2, BLOCK_BYTES).transpose(0, 2, 3, 1, 4)
return np.ascontiguousarray(blob).reshape(-1)
```
In words: column blocks of 64 rows; inside each, k-tiles of 512 columns; inside each, the two
256-column halves; inside each, the two 32-row halves. `blob_to_grid` reshapes to
`(n // 64, k // 512, 2, 2, BLOCK_BYTES)`, applies `transpose(0, 3, 1, 2, 4)`, and reshapes to
`(n // 32, k // 256, BLOCK_BYTES)`.

## Steps
1. `git switch -c task/T-018-q4nx-pack main`; set T-018 to `IN PROGRESS` in `PLAN.md`.
2. `scripts/test.sh T-018`. Expect `FAIL 0/24` (`cannot import name 'q4nx'`). This is the baseline.
3. Write the constants, `pad_q4`, `pack_grid`, `unpack_grid`.
   `scripts/test.sh T-018 -k "layout or grid"` → must print `PASS 11/11`.
4. Write `grid_to_blob`, `blob_to_grid`. `scripts/test.sh T-018 -k "blob"` → `PASS 6/6`.
5. Write `Q4NXWeight`, `from_gguf`. `scripts/test.sh T-018` → must print `PASS 24/24`.
   Also `scripts/test.sh T-003` (must stay `PASS 25/25`).
6. End-of-iteration steps in AGENTS.md §5 with `@DONE`.

## Done-command
scripts/test.sh T-018   # must print PASS 24/24

## Tolerances / perf target
- Layout tests are exact: codes equal, and scale/min equal the bf16 round-to-nearest-even values.
- Values: `|Q4NX weight − GGUF weight| ≤ 2^-8 · (|scale|·code + |min|)` elementwise (one bf16
  rounding on each). The Architect's version reaches 0.996× this bound. Truncating or flooring to
  bf16 instead of round-to-nearest-even fails it.
- `from_gguf` on a 4096×4096 Q4_K tensor: < 5 s. numpy vector ops only, no Python loops over blocks.

## Known pitfalls
- **Use `DequantBFP`'s block order, not `pack_q4nx`'s.** IRON's
  `iron/operators/flm/layer/reference.py` has a `pack_q4nx` with a *different* block order (pairs
  of row blocks per 256-column tile, for the fused layer kernel). Do not copy its last two lines.
- **Do not import `aie` or `iron` in `src/xdna_relay/formats/`.** The tests use IRON only as an oracle.
  Copy the constants as plain numbers (a test checks that they equal IRON's).
- **Round to nearest even.** `float32.astype(ml_dtypes.bfloat16)` does RNE. Do not shift the float32 bits
  right by 16 (that truncates).
- `.view(np.uint8)` / `.view(np.uint16)` need a contiguous array: call `np.ascontiguousarray` after
  any transpose and before `.view`.
- `<<` on uint8 overflows silently above 255. `c[..., 1::2] << 4` is safe only because codes are ≤ 15.
- The `-k` filter matches the file name too (`test_t018_q4nx_pack.py`). Use the exact expressions in
  the steps.
