"""T-018: Q4NX-Gemma4 pack + GGUF Q4_0/Q4_K -> Q4NX repack (ADR 0005).

Contract under test (see tasks/T-018-q4nx-pack.md for the full interface):
    src/xdna_relay/formats/q4nx.py
        M_TILE, K_TILE, GROUP, BLOCK_BYTES, N_ALIGN, K_ALIGN,
        pad_q4(u), pack_grid(u), unpack_grid(grid), grid_to_blob(grid), blob_to_grid(blob, n, k),
        Q4NXWeight, from_gguf(t)
Oracles (CPU only, no hardware): IRON's iron.operators.flm.q4nx.unpack/dequantize (block
layout) and iron.operators.flm.dequant.op.DequantBFP.grid (blob order the NPU op reads),
gguf-py dequantize (values).
"""

import os
import subprocess
import sys

import numpy as np
import pytest
from gguf import GGMLQuantizationType
from gguf.quants import dequantize as oracle_dequantize
from ml_dtypes import bfloat16

from tests.reference.gguf_files import random_raw


def _mod():
    from xdna_relay.formats import q4nx

    return q4nx


def _gguf():
    from xdna_relay.formats import gguf

    return gguf


def _rand_unpacked(n, k, seed):
    rng = np.random.default_rng(seed)
    return _gguf().Q4Unpacked(
        codes=rng.integers(0, 16, (n, k), dtype=np.uint8),
        scale=rng.uniform(1e-3, 5e-2, (n, k // 32)).astype(np.float32),
        min=rng.uniform(-0.5, 0.5, (n, k // 32)).astype(np.float32),
    )


def _bf16(x):
    """Round-to-nearest-even to bf16, back as float32 (ml_dtypes rounds RNE)."""
    return np.asarray(x, np.float32).astype(bfloat16).astype(np.float32)


# ---------------------------------------------------------------- layout constants


def test_layout_constants_match_iron():
    from iron.operators.flm import q4nx as iq
    from iron.operators.flm.gemm.design import K_TILE as K_TILE_B
    from iron.operators.flm.gemm.design import N_TILE_DEFAULT

    m = _mod()
    assert (m.M_TILE, m.K_TILE, m.GROUP, m.BLOCK_BYTES) == (iq.M_TILE, iq.K_TILE, iq.GROUP, iq.BLOCK_BYTES)
    assert (m.M_TILE, m.K_TILE, m.GROUP, m.BLOCK_BYTES) == (32, 256, 32, 5120)
    assert (m.N_ALIGN, m.K_ALIGN) == (N_TILE_DEFAULT, K_TILE_B) == (64, 512)


def test_layout_module_stays_npu_free():
    code = (
        "import sys, xdna_relay.formats.q4nx; "
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('aie', 'iron', 'pyxrt')); "
        "assert not bad, bad"
    )
    env = dict(os.environ, PYTHONPATH="src")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_layout_pad():
    m = _mod()
    u = _rand_unpacked(40, 96, seed=1)
    p = m.pad_q4(u)
    assert p.codes.shape == (64, 512) and p.scale.shape == (64, 16) and p.min.shape == (64, 16)
    assert p.codes.dtype == np.uint8 and p.scale.dtype == np.float32 and p.min.dtype == np.float32
    np.testing.assert_array_equal(p.codes[:40, :96], u.codes)
    np.testing.assert_array_equal(p.scale[:40, :3], u.scale)
    np.testing.assert_array_equal(p.min[:40, :3], u.min)
    assert not p.codes[40:].any() and not p.codes[:, 96:].any()
    assert not p.scale[40:].any() and not p.scale[:, 3:].any()
    assert not p.min[40:].any() and not p.min[:, 3:].any()
    a = _rand_unpacked(128, 1024, seed=2)
    q = m.pad_q4(a)
    np.testing.assert_array_equal(q.codes, a.codes)
    np.testing.assert_array_equal(q.scale, a.scale)


# ---------------------------------------------------------------- grid (block layout)


@pytest.mark.parametrize("n,k", [(32, 256), (96, 768), (64, 1024)])
def test_grid_matches_iron_unpack(n, k):
    from iron.operators.flm import q4nx as iq

    m = _mod()
    u = _rand_unpacked(n, k, seed=n + k)
    g = m.pack_grid(u)
    assert g.dtype == np.uint8 and g.shape == (n // 32, k // 256, 5120)
    codes, scales, mins = iq.unpack(g)
    np.testing.assert_array_equal(codes, u.codes)
    np.testing.assert_array_equal(scales, _bf16(u.scale))
    np.testing.assert_array_equal(mins, _bf16(u.min))


def test_grid_roundtrip():
    m = _mod()
    u = _rand_unpacked(96, 768, seed=5)
    v = m.unpack_grid(m.pack_grid(u))
    assert v.codes.dtype == np.uint8 and v.scale.dtype == np.float32 and v.min.dtype == np.float32
    np.testing.assert_array_equal(v.codes, u.codes)
    np.testing.assert_array_equal(v.scale, _bf16(u.scale))
    np.testing.assert_array_equal(v.min, _bf16(u.min))


def test_grid_byte_layout_of_one_block():
    """Scales then mins (bf16 LE, [k // 32, n]), then codes as nibbles [n // 16, k, n % 16], low first."""
    m = _mod()
    g = _gguf()
    codes = np.zeros((32, 256), np.uint8)
    scale = np.zeros((32, 8), np.float32)
    mn = np.zeros((32, 8), np.float32)
    codes[17, 5] = 9  # nibble 1*4096 + 5*16 + 1 = 4177 -> byte 1024 + 2088, high nibble
    codes[0, 0] = 7  # nibble 0 -> byte 1024, low nibble
    scale[3, 0] = 0.5  # [g=0, n=3] -> bytes 6..7, bf16 0x3F00
    mn[31, 7] = -2.0  # [g=7, n=31] -> bytes 512 + 2*(7*32 + 31) = 1022..1023, bf16 0xC000
    b = m.pack_grid(g.Q4Unpacked(codes, scale, mn)).reshape(-1)
    assert b.shape == (5120,)
    assert (b[6], b[7]) == (0x00, 0x3F)
    assert (b[1022], b[1023]) == (0x00, 0xC0)
    assert b[1024] == 0x07
    assert b[1024 + 2088] == 0x90
    assert int(b.astype(np.int64).sum()) == 0x3F + 0xC0 + 0x07 + 0x90


def test_grid_rounds_to_nearest_even():
    m = _mod()
    g = _gguf()
    vals = np.array([1 + 2.0**-8, 1 + 3 * 2.0**-8, 1 + 2.0**-8 + 2.0**-20, -(1 + 2.0**-8 + 2.0**-20)], np.float32)
    expect = np.array([1.0, 1 + 2.0**-6, 1 + 2.0**-7, -(1 + 2.0**-7)], np.float32)
    scale = np.zeros((32, 8), np.float32)
    scale[:4, 0] = vals
    v = m.unpack_grid(m.pack_grid(g.Q4Unpacked(np.zeros((32, 256), np.uint8), scale, scale.copy())))
    np.testing.assert_array_equal(v.scale[:4, 0], expect)
    np.testing.assert_array_equal(v.min[:4, 0], expect)


@pytest.mark.parametrize("n,k", [(48, 256), (32, 384)])
def test_grid_rejects_misaligned(n, k):
    with pytest.raises(ValueError):
        _mod().pack_grid(_rand_unpacked(n, k, seed=0))


# ---------------------------------------------------------------- blob (DequantBFP order)


@pytest.mark.parametrize("n,k", [(64, 512), (128, 1024), (192, 1536)])
def test_blob_matches_dequant_op_order(n, k):
    from iron.operators.flm.dequant.op import DequantBFP

    m = _mod()
    rng = np.random.default_rng(n * k)
    grid = rng.integers(0, 256, (n // 32, k // 256, 5120), dtype=np.uint8)
    blob = m.grid_to_blob(grid)
    op = DequantBFP(K=k, N=n)
    assert blob.dtype == np.uint8 and blob.ndim == 1 and blob.size == op.quantized_size()
    np.testing.assert_array_equal(op.grid(blob), grid)
    np.testing.assert_array_equal(m.blob_to_grid(blob, n, k), grid)


@pytest.mark.parametrize("n,k", [(32, 512), (64, 256), (96, 512)])
def test_blob_rejects_misaligned(n, k):
    grid = np.zeros((n // 32, k // 256, 5120), np.uint8)
    with pytest.raises(ValueError):
        _mod().grid_to_blob(grid)


# ---------------------------------------------------------------- from_gguf


@pytest.mark.parametrize(
    "qtype,n,k,n_pad,k_pad",
    [("Q4_K", 96, 768, 128, 1024), ("Q4_0", 40, 96, 64, 512), ("Q4_K", 128, 1024, 128, 1024)],
)
def test_gguf_repack_values(qtype, n, k, n_pad, k_pad):
    """Q4NX weights == GGUF weights up to bf16 (RNE) rounding of scale and min; padding is 0."""
    from iron.operators.flm import q4nx as iq
    from iron.operators.flm.dequant.op import DequantBFP

    m = _mod()
    raw = random_raw(qtype, n, k, np.random.default_rng(n + k))
    t = _gguf().GGUFTensor("w", qtype, (n, k), raw)
    w = m.from_gguf(t)
    assert (w.n, w.k, w.n_pad, w.k_pad) == (n, k, n_pad, k_pad)
    assert w.blob.dtype == np.uint8 and w.blob.ndim == 1
    assert w.blob.size == n_pad * k_pad * 5120 // (32 * 256)
    got = iq.dequantize(DequantBFP(K=k_pad, N=n_pad).grid(w.blob))  # float64 (n_pad, k_pad)
    assert not got[n:].any() and not got[:, k:].any()
    exact = oracle_dequantize(raw, GGMLQuantizationType[qtype]).astype(np.float64)
    u = _gguf().unpack_q4(t)
    s = np.repeat(np.abs(u.scale.astype(np.float64)), 32, axis=1)
    mn = np.repeat(np.abs(u.min.astype(np.float64)), 32, axis=1)
    # RNE to bf16 (8 significant bits) moves a value by at most 2^-8 of itself: one such
    # rounding on scale (times code) and one on min.
    bound = 2.0**-8 * (s * u.codes + mn) + 1e-30
    err = np.abs(got[:n, :k] - exact)
    assert (err <= bound).all(), f"worst {np.max(err / bound):.3f}x the bf16 rounding bound"


@pytest.mark.parametrize("qtype", ["Q6_K", "Q8_0"])
def test_gguf_rejects_non_q4(qtype):
    raw = random_raw(qtype, 32, 256, np.random.default_rng(0))
    with pytest.raises(ValueError):
        _mod().from_gguf(_gguf().GGUFTensor("w", qtype, (32, 256), raw))


def test_gguf_weight_is_frozen():
    raw = random_raw("Q4_K", 64, 512, np.random.default_rng(0))
    w = _mod().from_gguf(_gguf().GGUFTensor("w", "Q4_K", (64, 512), raw))
    with pytest.raises(Exception):
        w.n = 1


def test_gguf_repack_speed():
    """A 4096 x 4096 Q4_K layer repacks in a few seconds (numpy only, no Python loops)."""
    import time

    raw = random_raw("Q4_K", 4096, 4096, np.random.default_rng(7))
    t = _gguf().GGUFTensor("big", "Q4_K", (4096, 4096), raw)
    t0 = time.perf_counter()
    w = _mod().from_gguf(t)
    dt = time.perf_counter() - t0
    assert w.blob.size == 4096 * 4096 * 5120 // 8192
    assert dt < 5.0, f"repack took {dt:.1f}s; vectorize with numpy"
