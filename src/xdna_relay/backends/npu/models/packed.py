"""Projections whose prompt-chunk GEMMs read pre-packed bfp16 weights (T-033).

IRON's ``F.linear`` hands ``flm.ProjectionGEMM`` the bf16 ``(out, in)`` checkpoint matrix
(``b_col_maj=True``); at ``Precision.MEDIUM`` the cores round it to bfp16ebs8 on every pass,
and the GEMM re-reads B from DDR for every 256-row unit of the prompt (``_units_valid``), so
a prompt chunk is weight-bandwidth bound (~22 GB/s; bench/results/2026-11-25-placement.md).
Packing B once on the host (``ProjectionGEMM.pack_b``, bfp16ebs8: 9 bytes per 8 values,
``bfp16_b=True``) halves those bytes and skips the in-core rounding: standalone, the 8B's
GEMMs run 1.3-2.0x faster (bench/results/2026-11-25-packed-gemm.md).

The packed matrix is an ``iron.State`` of bfp16ebs8 blocks (a ``param`` cannot hold a
block-float dtype: numpy has none), filled after load (``fill_packed``). Decode (one row)
keeps GEMV on the bf16 ``weight``, so both copies are resident.
"""
from __future__ import annotations

import numpy as np
from aie.helpers.util import v8bfp16ebs8
from aie.utils import bfp

import iron
from iron import Precision
from iron.lm.blocks import Attention, GatedMLP, Linear
from iron.operators.flm.gemm.design import Epilogue
from iron.operators.flm.gemm.op import ProjectionGEMM


def _patch_dma_bd_fits() -> None:
    """mlir-aie's ``Device.dma_bd_fits`` calls ``np.dtype(dtype)``, which raises for the
    block-float marker class: a large packed B (the 8B's; per-call-sized transfers) fails
    to sequence. Same check, the element type mapped as the fifo types map it
    (``!aiex.bfp<"v8bfp16ebs8">``). UNVERIFIED upstream; docs/knowledge/pitfalls.md."""
    from aie.iron.device import device as d

    if getattr(d.Device.dma_bd_fits, "_xdna_relay", False):
        return
    orig = d.Device.dma_bd_fits

    def dma_bd_fits(self, col, row, tap, dtype):
        if not bfp.is_bfp(dtype):
            return orig(self, col, row, tap, dtype)
        self._validate_coordinates(col, row)
        if self._tm.is_shim_noc_or_pl_tile(col, row):
            if tap.padding is not None:
                return False
            offset, sizes, strides = d.shim_dims(tap)
        else:
            pad = 4 - len(tap.sizes)
            offset = tap.offset
            sizes = [1] * pad + list(tap.sizes)
            strides = [0] * pad + list(tap.strides)
        if len(sizes) > 4:
            return False
        context = d.Device._bd_context
        if context is None:
            context = d.Device._bd_context = d.ir.Context()
        with context, d.ir.Location.unknown():
            return d.dma_bd_fits(self._device.value, col, row, d.np_dtype_to_mlir_type(dtype),
                                 offset, sizes, strides, d.math.prod(sizes[1:]))

    dma_bd_fits._xdna_relay = True
    d.Device.dma_bd_fits = dma_bd_fits


_patch_dma_bd_fits()


def _gemm(rows: int, k: int, n: int, profile=None) -> ProjectionGEMM:
    given = dict(M=rows, K=k, N=n, epilogue_modes=(Epilogue.NONE,), precision=Precision.MEDIUM,
                 b_col_maj=False, bfp16_b=True)
    return ProjectionGEMM(**given) if profile is None else profile.make(ProjectionGEMM, **given)


class PackedLinear(Linear):
    """``Linear`` whose several-row calls run ``ProjectionGEMM`` on a packed bfp16 B."""

    def __init__(self, in_features: int, out_features: int, bias: bool, *, precision: Precision):
        if precision != Precision.MEDIUM:
            raise ValueError("PackedLinear: bfp16 B multiplies at precision='medium' only")
        super().__init__(in_features, out_features, bias, precision=precision)
        self.packed = iron.state((in_features * out_features // bfp.BLOCK,), v8bfp16ebs8)

    def body(self, x):
        if len(x.shape) == 1 or x.shape[0] == 1:
            return super().body(x)
        y = ProjectionGEMM.step(x, self.packed, N=self.out_features, b_col_maj=False,
                                bfp16_b=True, epilogue_modes=(Epilogue.NONE,),
                                precision=self.precision)
        if self.bias is None:
            return y
        return y + self.bias


def _swap(owner, name: str) -> None:
    lin = getattr(owner, name)
    if lin is None or isinstance(lin, PackedLinear):
        return
    p = PackedLinear(lin.in_features, lin.out_features, lin.bias is not None, precision=lin.precision)
    setattr(owner, name, p)


class PackedAttention(Attention):
    """``Attention`` with packed q/k/v/o; ``project`` calls the projections (not F.linear on
    their weights) for several rows."""

    def __init__(self, c, i: int, norms: str = "", o_bias: bool = False):
        super().__init__(c, i, norms=norms, o_bias=o_bias)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            _swap(self, name)

    def project(self, x, **gemv) -> list:
        if len(x.shape) == 2 and x.shape[0] > 1 and not gemv:
            return [None if p is None else p(x) for p in (self.q_proj, self.k_proj, self.v_proj)]
        return super().project(x, **gemv)


class PackedGatedMLP(GatedMLP):
    def __init__(self, c):
        super().__init__(c)
        for name in ("gate_proj", "up_proj", "down_proj"):
            _swap(self, name)


def _packed_linears(model):
    for i, layer in enumerate(model.model.layers):
        for part in ("self_attn", "mlp"):
            block = getattr(layer, part)
            for name, x in vars(block).items():
                if isinstance(x, PackedLinear):
                    yield f"layers.{i}.{part}.{name}", x


def fill_packed(model, log=None) -> int:
    """Pack every ``PackedLinear``'s loaded bf16 weight into its state on the device.

    Call after ``load_versions``. The layout follows the GEMM the prompt graph runs (same
    profile, so the same tiles). Returns the bytes written.
    """
    graph = model.prompt_chunk
    profile = model._profile()
    rows = model.config.prefill_chunk
    total = 0
    for path, x in _packed_linears(model):
        w = np.asarray(x.weight)  # (out, in) bf16
        op = _gemm(rows, x.in_features, x.out_features, profile)
        packed = np.ascontiguousarray(op.pack_b(np.ascontiguousarray(w.T))).reshape(-1).view(np.uint8)
        buf = graph._storage(x.packed)
        view = buf.numpy_view().reshape(-1).view(np.uint8)
        if view.size < packed.size:
            raise ValueError(f"{path}: state of {view.size} bytes, packed B {packed.size}")
        view[: packed.size] = packed
        buf.to("npu")
        total += packed.size
        if log:
            log(f"packed {path}: {packed.size / 2**20:.1f} MiB")
    return total
