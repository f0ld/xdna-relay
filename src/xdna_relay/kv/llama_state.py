"""llama.cpp sequence-state blobs (``llama_state_seq_get_data`` / ``set_data``) for a plain
KV cache, parsed and built in Python (T-016).

Format at llama.cpp 6b0f75b (src/llama-context.cpp ``state_seq_get_data``,
src/llama-kv-cache.cpp ``state_write`` / ``state_write_meta`` / ``state_write_data``):

    u32 io_magic, i32 seq_id,
    u32 n_stream; per stream: u32 cell_count; if cell_count:
      cell_count x { i32 pos, u32 n_seq_id, [cell ext if has_cell_ext], n_seq_id x i32 seq_id }
      u32 v_trans, u32 n_layer, u32 n_rot_k, u32 n_rot_v
      n_layer x { i32 k_type, u64 k_row_bytes, cell_count * k_row_bytes }
      if not v_trans: n_layer x { i32 v_type, u64 v_row_bytes, cell_count * v_row_bytes }

Supported: one stream, no cell ext (no M-RoPE / PLE), ``v_trans == 0`` (flash attention on).
Anything else raises, so a format change fails loudly instead of corrupting a cache.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_BF16 = 30


@dataclass
class Layer:
    type: int
    row_bytes: int
    rows: np.ndarray  # uint8 [cells, row_bytes]


@dataclass
class KVState:
    magic: int
    seq_id: int
    positions: np.ndarray  # int32 [cells]
    v_trans: int = 0
    n_rot_k: int = 0
    n_rot_v: int = 0
    k: list[Layer] = field(default_factory=list)
    v: list[Layer] = field(default_factory=list)

    @property
    def n_cells(self) -> int:
        return int(self.positions.size)


class _R:
    def __init__(self, b: bytes):
        self.b, self.o = memoryview(b), 0

    def take(self, fmt: str):
        v = struct.unpack_from("<" + fmt, self.b, self.o)
        self.o += struct.calcsize("<" + fmt)
        return v[0] if len(v) == 1 else v

    def raw(self, n: int) -> np.ndarray:
        a = np.frombuffer(self.b, dtype=np.uint8, count=n, offset=self.o)
        self.o += n
        return a


def parse(blob) -> KVState:
    r = _R(bytes(blob))
    magic, seq_id = r.take("I"), r.take("i")
    n_stream = r.take("I")
    if n_stream != 1:
        raise ValueError(f"{n_stream} KV streams: only 1 supported")
    cells = r.take("I")
    pos = np.empty(cells, np.int32)
    for i in range(cells):
        pos[i], n_seq = r.take("i"), r.take("I")
        if n_seq != 1:
            raise ValueError(f"cell {i} belongs to {n_seq} sequences")
        r.take("i")
    st = KVState(magic, seq_id, pos)
    if cells == 0:
        return st
    st.v_trans, n_layer, st.n_rot_k, st.n_rot_v = r.take("I"), r.take("I"), r.take("I"), r.take("I")
    if st.v_trans:
        raise ValueError("transposed V (flash attention off) is not supported")
    for dst in (st.k, st.v):
        for _ in range(n_layer):
            t, rb = r.take("i"), r.take("Q")
            dst.append(Layer(t, rb, r.raw(cells * rb).reshape(cells, rb)))
    if r.o != len(r.b):
        raise ValueError(f"{len(r.b) - r.o} trailing bytes: format differs (cell ext? hybrid memory?)")
    return st


def build(st: KVState) -> bytes:
    n = st.n_cells
    out = [struct.pack("<IiII", st.magic, st.seq_id, 1, n)]
    for p in st.positions:
        out.append(struct.pack("<iIi", int(p), 1, st.seq_id))
    if n:
        out.append(struct.pack("<IIII", st.v_trans, len(st.k), st.n_rot_k, st.n_rot_v))
        for layers in (st.k, st.v):
            for L in layers:
                if L.rows.shape != (n, L.row_bytes):
                    raise ValueError(f"layer rows {L.rows.shape} != ({n}, {L.row_bytes})")
                out.append(struct.pack("<iQ", L.type, L.row_bytes))
                out.append(np.ascontiguousarray(L.rows).tobytes())
    return b"".join(out)


def halves_to_interleaved(x: np.ndarray) -> np.ndarray:
    """Hugging Face RoPE layout (pairs j, j+D/2) -> llama.cpp/Meta layout (pairs 2j, 2j+1).

    Applies to the last axis (head_dim). GGUF conversion permutes q/k weights this way
    (convert_hf_to_gguf.py ``LlamaModel.permute``), so llama.cpp's cached K rows are in the
    interleaved order while a HF-layout model (IRON) produces the halves order.
    """
    D = x.shape[-1]
    out = np.empty_like(x)
    out[..., 0::2] = x[..., : D // 2]
    out[..., 1::2] = x[..., D // 2:]
    return out


def from_kv(template: KVState, keys: list[np.ndarray], values: list[np.ndarray], start_pos: int = 0) -> KVState:
    """A state for ``len(keys[0])`` cells at positions ``start_pos..``, header from ``template``.

    ``keys`` / ``values``: per layer ``[cells, n_kv_heads * head_dim]`` in llama.cpp's channel
    order (apply ``halves_to_interleaved`` to K first if needed), any float dtype; converted to
    the template's cache type (F16 / BF16 / F32).
    """
    n = keys[0].shape[0]
    st = KVState(template.magic, template.seq_id, np.arange(start_pos, start_pos + n, dtype=np.int32),
                 template.v_trans, template.n_rot_k, template.n_rot_v)
    for src, tl, dst in ((keys, template.k, st.k), (values, template.v, st.v)):
        if len(src) != len(tl):
            raise ValueError(f"{len(src)} layers given, template has {len(tl)}")
        for x, L in zip(src, tl):
            if L.type == GGML_TYPE_F16:
                a = np.asarray(x, dtype=np.float32).astype(np.float16)
            elif L.type == GGML_TYPE_F32:
                a = np.asarray(x, dtype=np.float32)
            elif L.type == GGML_TYPE_BF16:
                import ml_dtypes
                a = np.asarray(x, dtype=np.float32).astype(ml_dtypes.bfloat16)
            else:
                raise ValueError(f"cache type {L.type} not supported (quantized KV?)")
            rows = np.ascontiguousarray(a).reshape(n, -1).view(np.uint8)
            if rows.shape[1] != L.row_bytes:
                raise ValueError(f"row of {rows.shape[1]} bytes, cache expects {L.row_bytes}")
            dst.append(Layer(L.type, L.row_bytes, rows))
    return st


def to_kv(st: KVState) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per-layer float32 K and V ``[cells, row_elems]`` (for tests / inspection)."""
    def dec(L: Layer):
        if L.type == GGML_TYPE_F16:
            return L.rows.view(np.float16).astype(np.float32)
        if L.type == GGML_TYPE_F32:
            return L.rows.view(np.float32).copy()
        import ml_dtypes
        return L.rows.view(ml_dtypes.bfloat16).astype(np.float32)
    return [dec(L) for L in st.k], [dec(L) for L in st.v]
