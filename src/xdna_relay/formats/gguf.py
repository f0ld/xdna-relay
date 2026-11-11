"""GGUF reader, reference dequant, and Q4_0/Q4_K field unpacking.

Oracle: llama.cpp's gguf-py (gguf==0.19.0) `gguf.quants.dequantize`.
Pure numpy + torch on the CPU. No NPU code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from gguf import GGUFReader, GGML_QUANT_SIZES, GGMLQuantizationType
from gguf.quants import dequantize as _gguf_dequantize


# ---------------------------------------------------------------- constants

GROUP = 32  # weights per (scale, min) group along K, both Q4 types
Q4_TYPES = ("Q4_0", "Q4_K")

# ---------------------------------------------------------------- GGUFTensor

@dataclass(frozen=True)
class GGUFTensor:
    """A weight tensor read from a GGUF file."""
    name: str
    qtype: str               # GGMLQuantizationType member name: "Q4_K", "F32", ...
    shape: tuple[int, ...]   # numpy/PyTorch order (out, in)
    raw: np.ndarray          # exactly reader_tensor.data (may be read-only memmap)


# ---------------------------------------------------------------- read_tensors

def read_tensors(path: str | os.PathLike) -> dict[str, GGUFTensor]:
    """All tensors of the file by name.

    * ``shape`` = ``tuple(int(d) for d in reversed(t.shape))`` (reverses GGUF dims).
    * ``qtype`` = ``t.tensor_type.name``.
    * ``raw`` = ``t.data``.
    """
    reader = GGUFReader(path)
    result: dict[str, GGUFTensor] = {}
    for t in reader.tensors:
        result[t.name] = GGUFTensor(
            name=t.name,
            qtype=t.tensor_type.name,
            shape=tuple(int(d) for d in reversed(t.shape)),
            raw=t.data,
        )
    return result


# ---------------------------------------------------------------- dequantize

def dequantize(t: GGUFTensor) -> torch.Tensor:
    """float32 tensor of shape ``t.shape``, a writable copy.

    Uses the gguf-py oracle (``gguf.quants.dequantize``) and reshapes to our dim order.
    """
    a = _gguf_dequantize(np.asarray(t.raw), GGMLQuantizationType[t.qtype])
    return torch.from_numpy(np.array(a, dtype=np.float32, copy=True).reshape(t.shape))


# ---------------------------------------------------------------- Q4 unpacking

@dataclass(frozen=True)
class Q4Unpacked:
    """Per-group unpacked Q4 fields.

    weight[n, k] = ``scale[n, k // 32] * codes[n, k] + min[n, k // 32]``.
    """
    codes: np.ndarray   # uint8 (N, K), values 0..15
    scale: np.ndarray   # float32 (N, K // GROUP)
    min: np.ndarray     # float32 (N, K // GROUP)


def unpack_q4(t: GGUFTensor) -> Q4Unpacked:
    """ValueError unless ``t.qtype`` in ``Q4_TYPES`` and ``len(t.shape) == 2``.

    Vectorised numpy only — no Python loops over blocks or rows.
    """
    if t.qtype not in Q4_TYPES:
        raise ValueError(f"unpack_q4: expected Q4 type, got {t.qtype!r}")
    if len(t.shape) != 2:
        raise ValueError(f"unpack_q4: expected 2-D tensor, got {len(t.shape)}-D")

    n, k = t.shape
    bs, ts = GGML_QUANT_SIZES[GGMLQuantizationType[t.qtype]]
    raw = np.asarray(t.raw, dtype=np.uint8)
    nb = n * (k // bs)  # total blocks

    # View all rows as a flat list of blocks: (nb, ts)
    b = raw.reshape(nb, ts)

    if t.qtype == "Q4_0":
        return _unpack_q4_0(b, n, k, nb)
    else:
        return _unpack_q4_k(b, n, k, nb)


# ---------------------------------------------------------------- Q4_0

def _unpack_q4_0(b: np.ndarray, n: int, k: int, nb: int) -> Q4Unpacked:
    # d = bytes 0..1, f16 → float32
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32)  # (nb, 1)
    # nibbles: bytes 2..17 = 16 bytes = 32 nibbles
    # reshape to (nb, 1, 1, 16), shift by [0,4], mask → (nb, 1, 2, 16) → flatten → (nb, 32)
    qs = b[:, 2:].reshape((nb, 1, 1, 16))
    qs = (qs >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))) & np.uint8(0x0F)
    qs = qs.reshape(nb, -1)  # (nb, 32)

    # codes: uint8 (N, K)
    codes = qs.reshape(n, k)
    # scale: (N, K//32), min = -8 * d
    scale = d.reshape(n, k // 32)
    mn = -8.0 * d.reshape(n, k // 32)

    return Q4Unpacked(codes=codes, scale=scale, min=mn)


# ---------------------------------------------------------------- Q4_K

def _unpack_q4_k(b: np.ndarray, n: int, k: int, nb: int) -> Q4Unpacked:
    # d = bytes 0..1, dmin = bytes 2..3
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32)       # (nb, 1)
    dmin = b[:, 2:4].copy().view(np.float16).astype(np.float32)   # (nb, 1)

    # --- scales: bytes 4..15, 12 bytes → 8 sc + 8 m (6-bit each) ---
    scales_raw = b[:, 4:16].reshape((nb, 3, 4))
    d_s = scales_raw[:, 0, :]   # EEAAAAAA
    m_s = scales_raw[:, 1, :]   # FFBBBBBB
    m_d = scales_raw[:, 2, :]   # GGCCCCCC / HHDDDDDD

    sc = np.concatenate(
        [d_s & np.uint8(0x3F), (m_d & np.uint8(0x0F)) | ((d_s >> 2) & np.uint32(0x30))],
        axis=-1,
    ).reshape(nb, 8)
    mn = np.concatenate(
        [m_s & np.uint8(0x3F), (m_d >> 4) | ((m_s >> 2) & np.uint32(0x30))],
        axis=-1,
    ).reshape(nb, 8)

    # Final scale and min in float32
    scale = (d * sc.astype(np.float32)).reshape(nb, 8)   # (nb, 8)
    mn = -(dmin * mn.astype(np.float32)).reshape(nb, 8)  # (nb, 8)

    # --- codes: bytes 16..143, 128 bytes = 4 chunks of 32 bytes ---
    # each byte gives 2 nibbles → 128 bytes = 256 nibbles per block
    qs = b[:, 16:144].reshape((nb, 4, 1, 32))
    qs = (qs >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))) & np.uint8(0x0F)
    qs = qs.reshape(nb, -1).astype(np.uint8)  # (nb, 256)

    codes = qs.reshape(n, k)

    # Reshape scale / min to (n, k//32).
    # nb = n * (k//256), 8 groups per block → (nb, 8) → (n, k//32)
    return Q4Unpacked(codes=codes.reshape(n, k), scale=scale.reshape(n, k // 32), min=mn.reshape(n, k // 32))


# ---------------------------------------------------------------- q4_to_float

def q4_to_float(u: Q4Unpacked) -> np.ndarray:
    """float32 (N, K): ``scale * codes + min``, broadcasting groups."""
    s = np.repeat(u.scale, GROUP, axis=1)
    m = np.repeat(u.min, GROUP, axis=1)
    return s * u.codes.astype(np.float32) + m
