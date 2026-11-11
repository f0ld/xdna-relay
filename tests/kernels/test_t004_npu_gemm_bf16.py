"""T-004: `npu` backend for gemm_bf16 via IRON's GEMM operator.

Contract under test (see tasks/T-004-npu-gemm.md for the full interface):
    src/xdna_relay/backends/npu/gemm_bf16.py
        PAD, padded_shape(m, k, n), gemm_op(m_pad, k_pad, n_pad),
        to_numpy_bf16(t), to_torch_bf16(a), gemm_bf16_npu(a, b)
    `import xdna_relay.backends.npu` registers ("gemm_bf16", "npu") in ops.registry.DEFAULT.
Hardware tests take the `npu` fixture.
"""

import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from tests.reference.gemm import gemm_ref


def _mod():
    from xdna_relay.backends.npu import gemm_bf16

    return gemm_bf16


def _registry():
    import xdna_relay.backends.npu  # noqa: F401  (import side effect registers "npu")
    from xdna_relay.ops.registry import DEFAULT

    return DEFAULT


def _data(m, k, n, seed):
    g = torch.Generator().manual_seed(seed)
    # scale so |C| stays O(1) for every K
    a = (torch.randn(m, k, generator=g) / k**0.5).to(torch.bfloat16)
    b = torch.randn(k, n, generator=g).to(torch.bfloat16)
    return a, b


def _check(c, a, b):
    """c vs the float64 oracle: the contract tolerance, and IRON's own HIGHEST bound
    (2 bf16 units of |A| @ |B|, iron/operators/gemm.py GEMM.tolerance)."""
    from xdna_relay.ops.gemm_bf16 import GEMM_BF16

    m, n = a.shape[0], b.shape[1]
    assert c.dtype is torch.bfloat16 and tuple(c.shape) == (m, n)
    expected = gemm_ref(a, b)
    err = (c.double() - expected).abs()
    assert GEMM_BF16.tolerance.allclose(c, expected), f"max abs err {err.max().item():.3e}"
    bound = 2.0 * 2.0**-8 * (a.double().abs() @ b.double().abs())
    worst = (err / bound.clamp_min(1e-30)).max().item()
    assert worst <= 1.0, f"error is {worst:.2f}x IRON's HIGHEST bound (wrong precision?)"


# ---------------------------------------------------------------- no hardware


def test_ops_import_stays_npu_free():
    """Importing the op layer must not pull in IRON/mlir-aie (ref-only machines)."""
    code = (
        "import sys, xdna_relay.ops; "
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('aie', 'iron', 'pyxrt')); "
        "assert not bad, bad"
    )
    env = dict(os.environ, PYTHONPATH="src")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_padded_shape():
    m = _mod()
    assert m.PAD == 64
    assert m.padded_shape(1, 1, 1) == (64, 64, 64)
    assert m.padded_shape(64, 64, 64) == (64, 64, 64)
    assert m.padded_shape(100, 300, 70) == (128, 320, 128)
    assert m.padded_shape(65, 2048, 1) == (128, 2048, 64)


def test_bf16_conversion_roundtrip_is_bit_exact():
    from ml_dtypes import bfloat16

    m = _mod()
    t = torch.tensor(
        [[0.0, -0.0, 1.0, -2.5], [3.0e38, -1.0e-38, float("inf"), 0.0078125]],
        dtype=torch.bfloat16,
    )
    a = m.to_numpy_bf16(t)
    assert a.dtype == bfloat16 and a.shape == (2, 4)
    back = m.to_torch_bf16(a)
    assert back.dtype is torch.bfloat16 and back.shape == (2, 4)
    assert torch.equal(back.view(torch.int16), t.view(torch.int16))
    # a non-contiguous view converts by value, not by memory
    assert torch.equal(m.to_torch_bf16(m.to_numpy_bf16(t.T)).view(torch.int16),
                       t.T.contiguous().view(torch.int16))


def test_registered_in_default():
    from xdna_relay.ops.gemm_bf16 import GEMM_BF16

    r = _registry()
    assert r.contract("gemm_bf16") is GEMM_BF16
    assert r.backends("gemm_bf16") == ("npu", "ref")
    assert r.backend("gemm_bf16", "npu").impl is _mod().gemm_bf16_npu


def test_gemm_op_is_cached_and_highest_precision():
    from iron.common import Precision
    from iron.operators.gemm import GEMM

    m = _mod()
    op = m.gemm_op(64, 128, 64)
    assert isinstance(op, GEMM)
    assert (op.M, op.K, op.N) == (64, 128, 64)
    assert op.precision == Precision.HIGHEST
    assert m.gemm_op(64, 128, 64) is op


def test_rejects_empty_dims():
    r = _registry()
    with pytest.raises(ValueError):
        r.run("gemm_bf16", "npu", torch.zeros(0, 8, dtype=torch.bfloat16),
              torch.zeros(8, 4, dtype=torch.bfloat16))
    with pytest.raises(ValueError):
        r.run("gemm_bf16", "npu", torch.zeros(4, 0, dtype=torch.bfloat16),
              torch.zeros(0, 4, dtype=torch.bfloat16))


# ---------------------------------------------------------------- hardware


@pytest.mark.npu
@pytest.mark.parametrize(
    "m,k,n",
    [
        (1, 1, 1),  # pads to 64^3
        (64, 64, 64),  # no padding
        (100, 300, 70),  # every dim padded
        (64, 2048, 96),  # long K
        (256, 2048, 1024),  # prefill-like, 8 columns
        (512, 4096, 2048),
    ],
)
def test_npu_matches_oracle(npu, m, k, n):
    a, b = _data(m, k, n, seed=1000 * m + k + n)
    c = _registry().run("gemm_bf16", "npu", a, b)
    _check(c, a, b)


@pytest.mark.npu
def test_npu_repeated_calls_use_new_data(npu):
    """Same padded shape twice (cached operator): the second result must be from the second inputs."""
    r = _registry()
    a1, b1 = _data(100, 300, 70, seed=1)
    a2, b2 = _data(120, 320, 128, seed=2)  # pads to the same (128, 320, 128)
    c1 = r.run("gemm_bf16", "npu", a1, b1)
    c2 = r.run("gemm_bf16", "npu", a2, b2)
    _check(c1, a1, b1)
    _check(c2, a2, b2)


@pytest.mark.npu
def test_npu_leaves_inputs_untouched_and_takes_views(npu):
    a, b = _data(128, 256, 192, seed=3)
    a_copy, b_copy = a.clone(), b.clone()
    at = a.T.contiguous().T  # non-contiguous view with a's values
    bs = torch.stack([b, -b], dim=2).reshape(b.shape[0], -1)[:, ::2]  # strided view with b's values
    assert torch.equal(bs, b)
    assert not at.is_contiguous() and not bs.is_contiguous()
    c = _registry().run("gemm_bf16", "npu", at, bs)
    _check(c, a, b)
    assert torch.equal(a, a_copy) and torch.equal(b, b_copy)


@pytest.mark.npu
def test_npu_agrees_with_ref_backend(npu):
    """Both round an f32-accumulated sum once. They differ by at most 1 bf16 ulp plus the
    f32 accumulation-order error (K * 2^-24 of |A| @ |B|, which matters only where the sum
    cancels to near zero), and only rarely."""
    r = _registry()
    m, k, n = 256, 2048, 1024
    a, b = _data(m, k, n, seed=4)
    c_npu = r.run("gemm_bf16", "npu", a, b).double()
    c_ref = r.run("gemm_bf16", "ref", a, b).double()
    diff = (c_npu - c_ref).abs()
    ulp = 2.0 ** (torch.floor(torch.log2(c_ref.abs().clamp_min(2.0**-126))) - 7)
    f32_order = k * 2.0**-24 * (a.double().abs() @ b.double().abs())
    allowed = ulp + f32_order
    assert (diff <= allowed).all(), f"max diff {(diff / allowed).max().item():.2f}x allowed"
    assert (diff > 0).float().mean().item() < 0.01, "more than 1% of outputs differ from ref"
