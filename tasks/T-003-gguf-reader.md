# T-003 — GGUF reader, reference dequant, Q4_0/Q4_K field unpacking
Status: TODO
Milestone: M1
Depends on: ADR 0005 (GGUF is the weight source)

## Goal
Read weight tensors from a GGUF file (the same file llama.cpp decodes from). Dequantize any of
them to float32 (fallback path), and unpack Q4_0/Q4_K tensors into 4-bit **codes**, per-group
**scale** and per-group **min**. T-018 repacks those fields into the NPU's Q4NX layout.
Pure numpy + torch on the CPU. No NPU code.

## Files you may change
- `src/xdna_relay/formats/__init__.py` (create)
- `src/xdna_relay/formats/gguf.py` (create)
- `docs/knowledge/pitfalls.md` (append only, if you hit something new)
- `PLAN.md` (only the T-003 status line), `handoff/T-003-iN.md`

## Files to read first
- `tests/kernels/test_t003_gguf_reader.py`: the tests you must pass (do not edit)
- `tests/reference/gguf_files.py`: how the test files are built (do not edit)
- `docs/decisions/0005-weights-gguf-source-q4nx-on-npu.md`: why, and the Q4 block math
- `.venv/lib64/python3.14/site-packages/gguf/quants.py`: classes `Q4_0` and `Q4_K`
  (`dequantize_blocks`, `get_scale_min`). You port parts of these (MIT, llama.cpp).

## Verified API (gguf==0.19.0, installed in `.venv`; checked 2026-11-11)
- `from gguf import GGUFReader, GGMLQuantizationType, GGML_QUANT_SIZES`
- `from gguf.quants import dequantize` (rename on import: `as _gguf_dequantize`, since our
  module defines its own `dequantize`).
- `GGUFReader(path)` memory-maps the file. `reader.tensors` is a list of `ReaderTensor` with
  `.name` (str), `.tensor_type` (a `GGMLQuantizationType`; `.name` gives `"Q4_K"`, `"F32"`, …),
  `.shape` (numpy array of GGUF dims in **GGUF order** `[in_features, out_features]`, numpy uint64), and
  `.data` (numpy array; for quantized types `uint8` rows of shape `(out_features, row_bytes)`).
- `GGML_QUANT_SIZES[GGMLQuantizationType.Q4_K] == (256, 144)` = (weights per block, bytes per block);
  `Q4_0` → `(32, 18)`.
- `gguf.quants.dequantize(raw, qtype)` → float32 array `(out_features, in_features)` for 2-D tensors.

## Interface / contract — `src/xdna_relay/formats/gguf.py`

```python
GROUP = 32                       # weights per (scale, min) group along K, both Q4 types
Q4_TYPES = ("Q4_0", "Q4_K")

@dataclass(frozen=True)
class GGUFTensor:
    name: str
    qtype: str                   # GGMLQuantizationType member name: "Q4_K", "Q6_K", "F32", ...
    shape: tuple[int, ...]       # numpy/PyTorch order = REVERSED GGUF dims: (out_features N, in_features K)
                                 # for 2-D, (K,) for 1-D. Plain Python ints (not numpy ints).
    raw: np.ndarray              # exactly reader_tensor.data (may be a read-only memmap)

def read_tensors(path: str | os.PathLike) -> dict[str, GGUFTensor]:
    """All tensors of the file by name. shape = tuple(int(d) for d in reversed(t.shape)),
    qtype = t.tensor_type.name, raw = t.data."""

def dequantize(t: GGUFTensor) -> torch.Tensor:
    """float32 tensor of shape t.shape, a writable copy:
    a = _gguf_dequantize(np.asarray(t.raw), GGMLQuantizationType[t.qtype])
    return torch.from_numpy(np.array(a, dtype=np.float32, copy=True).reshape(t.shape))"""

@dataclass(frozen=True)
class Q4Unpacked:
    codes: np.ndarray            # uint8 (N, K), values 0..15
    scale: np.ndarray            # float32 (N, K // GROUP)
    min: np.ndarray              # float32 (N, K // GROUP)
    # weight[n, k] = scale[n, k // 32] * codes[n, k] + min[n, k // 32]

def unpack_q4(t: GGUFTensor) -> Q4Unpacked:
    """ValueError unless t.qtype in Q4_TYPES and len(t.shape) == 2. Vectorized numpy only."""

def q4_to_float(u: Q4Unpacked) -> np.ndarray:
    """float32 (N, K): np.repeat(scale, GROUP, axis=1) * codes.astype(np.float32)
    + np.repeat(min, GROUP, axis=1), all in float32."""
```

### Block layouts for `unpack_q4` (llama.cpp `ggml-common.h`)
First view the rows as blocks: `n, k = t.shape`; `bs, ts = GGML_QUANT_SIZES[...]`;
`b = np.asarray(t.raw, dtype=np.uint8).reshape(n * (k // bs), ts)`; `nb = b.shape[0]`.
Reading f16 from bytes: `b[:, 0:2].copy().view(np.float16).astype(np.float32)`. The `.copy()` makes
the slice contiguous, which `.view` needs.

**Q4_0** (18 bytes, 32 weights; one block = one group):
- bytes 0..1: f16 `d`.
- bytes 2..17: 16 bytes of nibbles. The **low** nibble of byte i is element i, the **high**
  nibble is element i + 16 (port `Q4_0.dequantize_blocks`, without its `- 8`).
- `scale = d`, `min = -8 * d`, `codes` = the nibbles (0..15).

**Q4_K** (144 bytes, 256 weights = 8 groups of 32):
- bytes 0..1: f16 `d`; bytes 2..3: f16 `dmin`.
- bytes 4..15: 12 bytes of packed 6-bit `sc[8]` and `m[8]`. Port `Q4_K.get_scale_min`
  **unchanged**: reshape to `(nb, 3, 4)`, split, `sc = concat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)])`, …
- bytes 16..143: 128 bytes of nibbles in 4 chunks of 32 bytes. Chunk c holds group 2c in its
  low nibbles and group 2c + 1 in its high nibbles (port `Q4_K.dequantize_blocks`' `qs` lines).
- `scale[j] = d * sc[j]`, `min[j] = -(dmin * m[j])` (all float32), `codes` = the nibbles.

Finally reshape: `codes` → `(n, k)` as uint8, `scale`/`min` → `(n, k // 32)` as float32. Blocks are
row-major (all blocks of row 0 first), so plain `reshape` keeps the order.

## Steps
1. `git switch -c task/T-003-gguf-reader main`; set T-003 to `IN PROGRESS` in `PLAN.md`.
2. `scripts/test.sh T-003`. Expect `FAIL 0/25` with `ModuleNotFoundError: No module named
   'xdna_relay.formats'`. This is the baseline.
3. Create `formats/__init__.py` (docstring `"""Weight file formats and converters."""`) and in
   `formats/gguf.py` write `GROUP`, `Q4_TYPES`, `GGUFTensor`, `read_tensors`, `dequantize`.
   `scripts/test.sh T-003 -k "open or dequant"` → must print `PASS 13/13`.
4. Write `Q4Unpacked`, `unpack_q4`, `q4_to_float`.
5. `scripts/test.sh T-003` → must print `PASS 25/25`. Also run `scripts/test.sh T-002`
   (must stay `PASS 35/35`).
6. End-of-iteration steps in AGENTS.md §5 with `@DONE`.

## Done-command
scripts/test.sh T-003   # must print PASS 25/25

## Tolerances / perf target
- `dequantize` and `q4_to_float` must equal gguf-py **bit for bit** (`assert_array_equal`). This
  holds only if the arithmetic is float32, in the order given above. Do not compute in float64.
- `unpack_q4` on a 4096×4096 Q4_K tensor: < 5 s (Architect's version: well under 1 s). Use numpy
  vector ops only, no Python loops over blocks or rows.

## Known pitfalls
- **GGUF dims are reversed.** `ReaderTensor.shape` is `[K, N]`, and our `shape` is `(N, K)`.
  `ReaderTensor.shape` holds numpy uint64. Convert each dim with `int(...)` (tested).
- **The `-k` filter matches the file name** (`test_t003_gguf_reader.py` contains `gguf`, `read`).
  Use the expression from step 3 exactly.
- `raw` can be a read-only memmap. `torch.from_numpy` on it warns or breaks in-place ops, so
  `dequantize` must copy (`np.array(..., copy=True)`, tested).
- `uint8 >> np.array([0, 4], dtype=np.uint8)` stays uint8. Shifting by a Python int list can
  upcast. Copy the gguf-py expressions exactly.
- `min` is a Python builtin. As a dataclass field name it's fine; as a local variable use `mn`.
- Do not import `gguf` from `src/xdna_relay/ops/` or anything the ops layer imports. Only
  `formats/` uses it.
- If `import gguf` fails: `scripts/test.sh` uses the project `.venv`; the Architect installed
  `gguf==0.19.0` there (pinned in `requirements-npu.txt`). Do not install anything. `@ESCALATE` instead.
