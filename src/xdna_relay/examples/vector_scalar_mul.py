"""vector_scalar_mul — port of the upstream mlir-aie smoke example onto the NPU.

Upstream: mlir-aie/programming_examples/basic/vector_scalar_mul/vector_scalar_mul.py
(verified against commit e5a3fa1 of the pinned mlir-aie wheel).
"""

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, Out, kernels
from aie.iron.algorithms import transform


@iron.jit
def _vsm_design(
    A: In,
    C: Out,
    F: In,
    *,
    in1_size: CompileTime[int],
    int_bit_width: CompileTime[int] = 16,
):
    """IRON JIT design for vector × scalar multiplication on the NPU.

    Copied from the upstream `vector_scalar_mul` example (toolchain.md, §"Smoke example"),
    simplified by removing ``trace_size``, ``use_chess``, ``vectorized`` parameters
    (hard-coded to ``vectorized=True``, ``use_chess=False``, ``trace_size=0``).

    Argument order: (A=input, C=output, F=scalar-factor).
    """
    in1_dtype = np.int16 if int_bit_width == 16 else np.int32
    tensor_size = in1_size // np.dtype(in1_dtype).itemsize  # in1_size is BYTES
    tile_size = tensor_size // 4
    tensor_ty = np.ndarray[(tensor_size,), np.dtype[in1_dtype]]
    scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]
    scale = kernels.scale(
        tile_size=tile_size,
        dtype=in1_dtype,
        vectorized=True,
        use_chess=False,
    )
    return transform(scale, tensor_ty, scalar_ty, tile_size=tile_size, trace_size=0)


def vector_scalar_mul(a: np.ndarray, factor: int) -> np.ndarray:
    """Return ``a * factor`` computed on the NPU, same shape and dtype as ``a``.

    Parameters
    ----------
    a : np.ndarray
        1-D array of ``np.int16`` or ``np.int32``.
        ``a.nbytes`` must be a multiple of 128 and >= 1024.
    factor : int
        Scalar multiplier (sent to the NPU as one int32).

    Returns
    -------
    np.ndarray
        New array with the same shape and dtype as ``a``.

    Raises
    ------
    ValueError
        If ``a`` has the wrong dtype, is not 1-D, or violates the size constraint.
    """
    # --- validation (before touching the NPU) ---
    if a.ndim != 1:
        raise ValueError(f"vector_scalar_mul expects 1-D array, got {a.ndim}-D")
    if a.dtype not in (np.int16, np.int32):
        raise ValueError(
            f"vector_scalar_mul expects int16 or int32, got {a.dtype}"
        )
    if a.nbytes % 128 != 0:
        raise ValueError(
            f"a.nbytes ({a.nbytes}) must be a multiple of 128"
        )
    if a.nbytes < 1024:
        raise ValueError(
            f"a.nbytes ({a.nbytes}) must be >= 1024"
        )

    # --- build NPU tensors ---
    a_t = iron.tensor(a, dtype=a.dtype, device="npu")
    f_t = iron.full((1,), int(factor), dtype=np.int32, device="npu")
    c_t = iron.zeros(a.size, dtype=a.dtype, device="npu")

    # --- run the design ---
    _vsm_design(
        a_t, c_t, f_t,
        in1_size=a.nbytes,
        int_bit_width=8 * a.itemsize,
    )

    # --- read back ---
    return c_t.numpy().copy()
