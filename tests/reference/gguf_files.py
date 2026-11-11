"""Helpers that build small GGUF files for tests. Architect-owned (protected).

Uses llama.cpp's gguf-py (pinned gguf==0.19.0) as the independent writer/oracle.
Q4_K, Q5_K and Q6_K cannot be quantized by gguf-py, so their blocks are random bytes with
finite f16 super-block scales. Every such byte pattern is a valid block.
"""

import numpy as np
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType, GGUFWriter

# Byte offsets of the f16 scale fields inside one block (llama.cpp ggml-common.h).
_F16_FIELDS = {
    "Q4_0": (0,),  # d
    "Q8_0": (0,),  # d
    "Q4_K": (0, 2),  # d, dmin
    "Q5_K": (0, 2),  # d, dmin
    "Q6_K": (208,),  # d is the last field (ql[128], qh[64], scales[16], d)
}


def random_raw(qtype: str, n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Random but valid raw rows for an (n, k) weight: uint8 (n, k // block * type_size)."""
    q = GGMLQuantizationType[qtype]
    bs, ts = GGML_QUANT_SIZES[q]
    assert k % bs == 0, (qtype, k, bs)
    raw = rng.integers(0, 256, size=(n, k // bs, ts), dtype=np.uint8)
    for off in _F16_FIELDS[qtype]:
        vals = rng.uniform(1e-3, 2e-2, size=(n, k // bs)).astype(np.float16)
        raw[..., off : off + 2] = vals.view(np.uint8).reshape(n, k // bs, 2)
    return raw.reshape(n, k // bs * ts)


def write_gguf(path, tensors: dict) -> None:
    """tensors: {name: (array, qtype)}. Quantized types take raw uint8 rows from random_raw();
    F32/F16 take a float array in numpy order (out, in)."""
    w = GGUFWriter(str(path), "llama")
    for name, (arr, qtype) in tensors.items():
        q = GGMLQuantizationType[qtype]
        if qtype in ("F32", "F16"):
            w.add_tensor(name, arr.astype(np.float32 if qtype == "F32" else np.float16))
        else:
            w.add_tensor(name, arr, raw_shape=arr.shape, raw_dtype=q)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
