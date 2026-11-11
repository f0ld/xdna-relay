"""T-003: GGUF reader, reference dequant, and Q4_0/Q4_K field unpacking.

Contract under test (see tasks/T-003-gguf-reader.md for the full interface):
    src/xdna_relay/formats/gguf.py
        GROUP, Q4_TYPES, GGUFTensor, read_tensors(path), dequantize(t),
        Q4Unpacked, unpack_q4(t), q4_to_float(u)
Oracle: llama.cpp's gguf-py (gguf==0.19.0) `gguf.quants.dequantize`. No hardware needed.
"""

import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType
from gguf.quants import dequantize as oracle_dequantize

from tests.reference.gguf_files import random_raw, write_gguf


def _mod():
    from xdna_relay.formats import gguf

    return gguf


def _oracle(raw, qtype):
    return oracle_dequantize(raw, GGMLQuantizationType[qtype]).astype(np.float32)


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    """One GGUF file with every tensor kind the tests need. Shapes are (out, in) = (N, K)."""
    rng = np.random.default_rng(3)
    t = {
        "q40": (random_raw("Q4_0", 64, 512, rng), "Q4_0"),
        "q4k": (random_raw("Q4_K", 96, 768, rng), "Q4_K"),
        "q5k": (random_raw("Q5_K", 32, 256, rng), "Q5_K"),
        "q6k": (random_raw("Q6_K", 32, 512, rng), "Q6_K"),
        "q80": (random_raw("Q8_0", 16, 64, rng), "Q8_0"),
        "f16": (rng.standard_normal((8, 32)), "F16"),
        "f32": (rng.standard_normal((5, 7)), "F32"),
        "norm": (rng.standard_normal(48), "F32"),
    }
    path = tmp_path_factory.mktemp("gguf") / "sample.gguf"
    write_gguf(path, t)
    return path, t


# ---------------------------------------------------------------- read_tensors


def test_open_constants():
    m = _mod()
    assert m.GROUP == 32
    assert m.Q4_TYPES == ("Q4_0", "Q4_K")


def test_open_reads_names_types_shapes(sample):
    path, t = sample
    ts = _mod().read_tensors(path)
    assert set(ts) == set(t)
    expect = {
        "q40": ("Q4_0", (64, 512)),
        "q4k": ("Q4_K", (96, 768)),
        "q5k": ("Q5_K", (32, 256)),
        "q6k": ("Q6_K", (32, 512)),
        "q80": ("Q8_0", (16, 64)),
        "f16": ("F16", (8, 32)),
        "f32": ("F32", (5, 7)),
        "norm": ("F32", (48,)),
    }
    for name, (qtype, shape) in expect.items():
        x = ts[name]
        assert x.name == name
        assert x.qtype == qtype, name
        assert x.shape == shape and all(type(d) is int for d in x.shape), (name, x.shape)


def test_open_raw_bytes_are_as_written(sample):
    path, t = sample
    ts = _mod().read_tensors(path)
    for name in ("q40", "q4k", "q6k"):
        assert ts[name].raw.dtype == np.uint8
        np.testing.assert_array_equal(np.asarray(ts[name].raw).reshape(t[name][0].shape), t[name][0])


def test_open_tensor_is_frozen(sample):
    x = _mod().read_tensors(sample[0])["q40"]
    with pytest.raises(Exception):
        x.qtype = "F32"


# ---------------------------------------------------------------- dequantize


@pytest.mark.parametrize("name", ["q40", "q4k", "q5k", "q6k", "q80", "f16", "f32", "norm"])
def test_dequant_matches_oracle(sample, name):
    path, t = sample
    x = _mod().read_tensors(path)[name]
    d = _mod().dequantize(x)
    assert isinstance(d, torch.Tensor) and d.dtype is torch.float32
    assert tuple(d.shape) == x.shape
    expected = _oracle(np.asarray(x.raw), x.qtype).reshape(x.shape)
    np.testing.assert_array_equal(d.numpy(), expected)
    if name in ("f16", "f32", "norm"):
        np.testing.assert_allclose(d.numpy(), t[name][0], rtol=1e-3, atol=1e-3)


def test_dequant_result_is_writable_copy(sample):
    x = _mod().read_tensors(sample[0])["f32"]
    d = _mod().dequantize(x)
    d.add_(1.0)  # must not raise (read-only mmap) and must not change the file view
    np.testing.assert_array_equal(_mod().dequantize(x).numpy() + 1.0, d.numpy())


# ---------------------------------------------------------------- unpack_q4


@pytest.mark.parametrize("name", ["q40", "q4k"])
def test_unpack_shapes_and_ranges(sample, name):
    x = _mod().read_tensors(sample[0])[name]
    u = _mod().unpack_q4(x)
    n, k = x.shape
    assert u.codes.dtype == np.uint8 and u.codes.shape == (n, k)
    assert u.codes.max() <= 15
    assert u.scale.dtype == np.float32 and u.scale.shape == (n, k // 32)
    assert u.min.dtype == np.float32 and u.min.shape == (n, k // 32)
    assert len(np.unique(u.codes)) == 16  # random data uses every code


@pytest.mark.parametrize("name", ["q40", "q4k"])
def test_unpack_is_bit_exact_vs_oracle(sample, name):
    """scale * code + min in float32 must reproduce gguf-py's dequantization bit for bit."""
    m = _mod()
    x = m.read_tensors(sample[0])[name]
    w = m.q4_to_float(m.unpack_q4(x))
    assert w.dtype == np.float32 and w.shape == x.shape
    np.testing.assert_array_equal(w, _oracle(np.asarray(x.raw), x.qtype).reshape(x.shape))


def test_unpack_q4_0_fields():
    """Q4_0: scale = d, min = -8 * d, codes = stored nibbles (low nibbles are elements 0..15)."""
    m = _mod()
    raw = np.zeros((1, 18), dtype=np.uint8)
    raw[0, 0:2] = np.array([0.5], dtype=np.float16).view(np.uint8)
    raw[0, 2] = 0x21  # element 0 -> 1, element 16 -> 2
    raw[0, 17] = 0xF3  # element 15 -> 3, element 31 -> 15
    u = m.unpack_q4(m.GGUFTensor("t", "Q4_0", (1, 32), raw))
    assert u.scale[0, 0] == np.float32(0.5) and u.min[0, 0] == np.float32(-4.0)
    assert (u.codes[0, 0], u.codes[0, 16], u.codes[0, 15], u.codes[0, 31]) == (1, 2, 3, 15)
    assert u.codes[0, 1:15].sum() == 0


def test_unpack_q4_k_fields():
    """Q4_K: group j has scale = d * sc[j], min = -dmin * m[j] (6-bit sc/m, llama.cpp packing)."""
    m = _mod()
    raw = np.zeros((1, 144), dtype=np.uint8)
    raw[0, 0:4] = np.array([0.25, 0.125], dtype=np.float16).view(np.uint8)  # d, dmin
    # scales[12]: j < 4 -> sc = scales[j] & 63, m = scales[j + 4] & 63
    #             j >= 4 -> sc = (scales[j + 4] & 15) | ((scales[j - 4] >> 6) << 4),
    #                       m  = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)
    raw[0, 4 + 0] = 5  # sc[0] = 5
    raw[0, 4 + 4] = 7  # m[0] = 7
    raw[0, 4 + 8] = 0x93  # sc[4] = 3, m[4] = 9
    raw[0, 4 + 0] |= 0x40  # adds 1 << 4 to sc[4]  -> sc[4] = 19
    raw[0, 4 + 4] |= 0x80  # adds 2 << 4 to m[4]   -> m[4] = 41
    # qs[128]: chunk c (32 bytes) holds group 2c in low nibbles, group 2c + 1 in high nibbles
    raw[0, 16 + 0] = 0xA1  # group 0 element 0 -> 1, group 1 element 0 -> 10
    raw[0, 16 + 64 + 31] = 0x0C  # group 4 element 31 -> 12
    u = m.unpack_q4(m.GGUFTensor("t", "Q4_K", (1, 256), raw))
    assert u.scale[0, 0] == np.float32(0.25 * 5) and u.min[0, 0] == np.float32(-0.125 * 7)
    assert u.scale[0, 4] == np.float32(0.25 * 19) and u.min[0, 4] == np.float32(-0.125 * 41)
    assert u.scale[0, 1] == 0 and u.min[0, 1] == 0
    assert (u.codes[0, 0], u.codes[0, 32], u.codes[0, 4 * 32 + 31]) == (1, 10, 12)


@pytest.mark.parametrize("name", ["q5k", "q6k", "q80", "f16", "norm"])
def test_unpack_rejects_other_types(sample, name):
    x = _mod().read_tensors(sample[0])[name]
    with pytest.raises(ValueError):
        _mod().unpack_q4(x)


def test_unpack_handles_many_rows_without_python_loops(sample):
    """A 4096 x 4096 Q4_K tensor (real layer size) unpacks in well under a few seconds."""
    import time

    m = _mod()
    raw = random_raw("Q4_K", 4096, 4096, np.random.default_rng(7))
    x = m.GGUFTensor("big", "Q4_K", (4096, 4096), raw)
    t0 = time.perf_counter()
    u = m.unpack_q4(x)
    dt = time.perf_counter() - t0
    assert u.codes.shape == (4096, 4096)
    assert dt < 5.0, f"unpack took {dt:.1f}s; vectorize with numpy"
