"""T-001: upstream vector_scalar_mul runs on the NPU and matches the reference.

Contract under test (src/xdna_relay/examples/vector_scalar_mul.py):
    vector_scalar_mul(a: np.ndarray, factor: int) -> np.ndarray
    - a: 1-D np.int16 or np.int32, len(a) * itemsize a multiple of 128 bytes and >= 1024 bytes
    - returns a new array, same shape and dtype as a
    - raises ValueError for any other dtype or size
"""

import numpy as np
import pytest

from tests.reference.eltwise import vector_scalar_mul_ref


def _impl():
    from xdna_relay.examples.vector_scalar_mul import vector_scalar_mul

    return vector_scalar_mul


def _data(n, dtype, seed):
    return np.random.default_rng(seed).integers(-1000, 1000, size=n, dtype=dtype)


def test_reference_sanity():
    a = np.array([1, -2, 3, 1000], dtype=np.int16)
    np.testing.assert_array_equal(vector_scalar_mul_ref(a, -7), [-7, 14, -21, -7000])


@pytest.mark.parametrize("n,dtype", [(1024, np.int16), (4096, np.int16), (2048, np.int32)])
def test_rejects_bad_input(n, dtype):
    f = _impl()
    with pytest.raises(ValueError):
        f(np.zeros(n, dtype=np.float32), 3)
    with pytest.raises(ValueError):
        f(np.zeros(100, dtype=dtype), 3)  # 200/400 bytes: below 1024 bytes


@pytest.mark.npu
@pytest.mark.parametrize(
    "n,dtype,factor",
    [
        (4096, np.int16, 3),  # upstream default shape
        (4096, np.int16, -7),
        (2048, np.int32, 5),
        (8192, np.int32, -11),
    ],
)
def test_npu_matches_reference(npu, n, dtype, factor):
    a = _data(n, dtype, seed=n + factor)
    out = _impl()(a.copy(), factor)
    assert out.dtype == a.dtype and out.shape == a.shape
    np.testing.assert_array_equal(out, vector_scalar_mul_ref(a, factor))


@pytest.mark.npu
def test_npu_repeat_calls_stable(npu):
    """Second call reuses the cached JIT artifact and must still be exact."""
    f = _impl()
    a = _data(4096, np.int16, seed=1)
    first = f(a, 3)
    second = f(a, 3)
    np.testing.assert_array_equal(first, second)
