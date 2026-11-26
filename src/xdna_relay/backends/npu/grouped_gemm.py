"""GroupedGEMM: one IRON ``ProjectionGEMM`` call over many experts (T-046).

A MoE layer is ``groups`` small GEMMs, each expert's routed rows times its own weight. As
separate ``ProjectionGEMM`` steps each costs ~80-100 us of fixed per-call work (the
sequence arming B's memtile pool, the cores' parameters, the drain) whatever its rows: 256
experts x 2 GEMMs x 40 layers is ~1.6 s per prompt pass on that alone
(bench/results/2026-11-26-moe-expert-probe.md).

``ProjectionGEMM`` already streams B from DDR once per 256-row unit when B is not resident
(``b_tap``: the unit dimension of B's transfer has stride 0, "B does not depend on the
row"). GroupedGEMM gives that dimension the stride of one expert's packed weight: unit ``g``
(rows ``256 g .. 256 g + 255`` of A and C) multiplies by expert ``g``'s B. One call, one
arming, all experts. A unit's rows are not contiguous: the two halves of the core rows
(``JOINS``) each take their own half of M, so unit ``g`` is rows ``128 g .. 128 g + 127``
and ``M/2 + 128 g .. M/2 + 128 g + 127`` of A and C (``unit_rows``). Rows of a unit past an
expert's routed tokens are computed and ignored (their C rows are garbage).

B: the experts' ``pack_b`` outputs concatenated, ``groups * K * N / 8`` bfp16ebs8 blocks
(``bfp16_b=True`` at ``Precision.MEDIUM`` only). Always streamed (one DDR read per unit),
never resident.

Implementation: ``sequence`` is ``ProjectionGEMM.sequence``'s source with three textual
edits (``_EDITS``), each asserted to apply exactly once, so an IRON update that moves them
fails at import rather than computing with the wrong weights. Derived from IRON
(Apache-2.0, Advanced Micro Devices).
"""
from __future__ import annotations

import inspect
import textwrap

import numpy as np
from aie.utils import bfp
from aie.utils.compile.jit.params import param

from iron import Precision
from iron.operators.flm.gemm import op as _gemm_op
from iron.operators.flm.gemm.design import Epilogue
from iron.operators.flm.gemm.op import ProjectionGEMM

UNIT = 256  # rows per unit at the device's 4 core rows: M_TILE (64) x rows; one unit per group

_EDITS = [
    # never resident: B differs per unit
    ("            if k_iters <= B_SLOTS and units <= pool.lock_max:\n",
     "            if False:  # GroupedGEMM: B differs per unit, never resident\n"),
    # the slab search otherwise insists on residency where k_iters fits the pool
    ("                k_iters > B_SLOTS or all(s.b_resident for s in slabs)\n",
     "                True  # GroupedGEMM: streamed by design\n"),
    # B: one packed matrix per group, the unit dimension steps a group
    ("            B = TensorAccessPattern.full((N // N_TILE, k_iters, b_slot_elems))[\n"
     "                :, :, None\n"
     "            ]\n",
     "            B = TensorAccessPattern.full((self.groups, N // N_TILE, k_iters, b_slot_elems))[\n"
     "                :, :, :, None\n"
     "            ]\n"),
    ("            sweeps = 1 if slab.b_resident else slab.units\n"
     "            return B[mega_col * COLS + c].repeat(sweeps)\n",
     "            return B[slab.first : slab.first + slab.units, mega_col * COLS + c]\n"),
    # a unit -> group table: one B transfer per unit per column (_split_blocks)
    ("            for bi, (mega_col, active_cols) in enumerate(blocks):\n",
     "            if self.unit_groups is not None:\n"
     "                _split_blocks(self, blocks, a_legs, B, slab, set_up, push_b_mt, c_tap, n_work, last_c, COLS, JOINS)\n"
     "            for bi, (mega_col, active_cols) in enumerate(blocks if self.unit_groups is None else []):\n"),
]


def _split_blocks(op, blocks, a_legs, B, slab, set_up, push_b_mt, c_tap, n_work, last_c, COLS, JOINS):
    """Unit u multiplies by group ``op.unit_groups[u]``: one B fill per unit per column.

    Issued after the block's A fills, the arming (``set_up``) and its C drains, so that a
    B lane's full queue (the compiler waits for a slot) never stands ahead of what the
    cores need to make progress."""
    for bi, (mega_col, active_cols) in enumerate(blocks):
        for r, (tap, _) in enumerate(a_legs):
            op.A.fill(tap, lane=r, managed=False)
        if bi == 0:
            set_up()
        else:
            push_b_mt(bi)
        for c in range(active_cols):
            last = bi == n_work(c) - 1
            for half in range(JOINS):
                task = op.C.drain(c_tap(mega_col, c, half, slab), lane=c * JOINS + half, wait=last,
                                  managed=False)
                if last:
                    last_c.append(task)
        for u in range(slab.first, slab.first + slab.units):
            g = op.unit_groups[u]
            for c in range(active_cols):
                op.B.fill(B[g, mega_col * COLS + c], lane=c, managed=False)


def _grouped_sequence():
    src = textwrap.dedent(inspect.getsource(ProjectionGEMM.sequence))
    for old, new in _EDITS:
        # the source was dedented by one level (4 spaces)
        old_d = "\n".join(l[4:] if l.startswith("    ") else l for l in old.split("\n"))
        new_d = "\n".join(l[4:] if l.startswith("    ") else l for l in new.split("\n"))
        n = src.count(old_d)
        if n != 1:
            raise ImportError(f"GroupedGEMM: IRON's ProjectionGEMM.sequence changed: edit found {n}x:\n{old_d}")
        src = src.replace(old_d, new_d)
    ns = dict(vars(_gemm_op))
    ns["_split_blocks"] = _split_blocks
    exec(compile(src, f"<GroupedGEMM.sequence from {_gemm_op.__file__}>", "exec"), ns)
    return ns["sequence"]


class GroupedGEMM(ProjectionGEMM):
    """``groups`` GEMMs ``C[rows g] = A[rows g] @ B[g]`` in one call (rows g = ``unit_rows(g)``)."""

    groups: int = param(default=1)
    # Core rows the GEMM uses (the device has 4): a unit is M_TILE x core_rows rows, so 2
    # halves the padding rows a sparsely routed expert drags through DDR.
    core_rows: int = param(default=4, array=True)
    # Unit -> group (expert) table: unit u's rows multiply by group unit_groups[u]; M is
    # len(unit_groups) units. None: unit g is group g. Static here (a rebuild per table): the
    # T-046 probe of what one B transfer per unit costs the command processor.
    unit_groups: tuple | None = param(default=None)
    packed_blocks: int = param(derive=lambda op: op.groups * op.K * op.N // bfp.BLOCK, repr=False)

    def validate(self) -> None:
        super().validate()
        if not self.bfp16_b or self.precision != Precision.MEDIUM:
            raise ValueError("GroupedGEMM: packed bfp16 B (bfp16_b=True) at precision='medium' only")
        if self.core_rows not in (2, 4):
            raise ValueError(f"GroupedGEMM: core_rows ({self.core_rows}) must be 2 or 4")
        n_units = self.groups if self.unit_groups is None else len(self.unit_groups)
        if self.unit_groups is not None and not all(0 <= g < self.groups for g in self.unit_groups):
            raise ValueError("GroupedGEMM: unit_groups entries must be group indices")
        if self.core_rows == 2 and n_units % 2:
            raise ValueError("GroupedGEMM: core_rows=2 needs an even unit count (IRON checks M % 256; pad the table)")
        if self.M != n_units * self.unit:
            raise ValueError(f"GroupedGEMM: M ({self.M}) must be units ({n_units}) x {self.unit}")
        if self.m_chunk not in (None, 1):
            raise ValueError("GroupedGEMM: m_chunk must be 1")

    sequence = _grouped_sequence()

    @property
    def unit(self) -> int:
        return 64 * self.core_rows  # M_TILE x core rows

    @property
    def _b_slots(self) -> int:
        """At 2 core rows a memtile holding an A forward also holds a second A buffer the
        parent's budget does not count (aiecc: 64 KiB over the 512 KiB memtile): leave it room."""
        n = super()._b_slots
        if self.core_rows == 4:
            return n
        from iron.operators.flm.gemm.design import K_TILE, _b_bytes
        extra = 2 * 64 * self.k_tile  # one A buffer, bf16
        per = _b_bytes(self.k_tile * self.tile_n, self.bfp16_b)
        return max(1, n - -(-extra // per))

    # B memtile ring passes per column-block a call may take with a unit table: 2 queue
    # pushes of 256. Measured 2026-11-26 (gate/up, 2 core rows, tile_n 128): 306 units = 408
    # passes ran; 418 (836 passes) and 420 (560) hit an NPU command timeout (TDR). Likely the
    # 3rd push waits on the 1st, which needs B fills _split_blocks issues after it. UNVERIFIED
    # cause; the bound keeps calls inside what ran. Split a larger table over several calls.
    MAX_TABLE_PASSES = 512

    def resolve(self, dev):
        r = super().resolve(dev)
        if self.core_rows != r.rows:
            from iron.operators.flm.gemm.design import C_JOINS, M_TILE
            r = r.replace(rows=self.core_rows, c_l2=M_TILE * r.tile_n * self.core_rows // C_JOINS)
        if r.unit_groups is not None:
            units, k_iters = len(r.unit_groups), r._k_iters
            ring = max(s for s in range(1, r._b_slots + 1) if (units * k_iters) % s == 0)
            if units * k_iters // ring > self.MAX_TABLE_PASSES:
                raise ValueError(f"GroupedGEMM: {units} units x {k_iters} k tiles over a {ring}-slot ring is "
                                 f"{units * k_iters // ring} B passes, over {self.MAX_TABLE_PASSES} (NPU timeout); "
                                 f"split the unit table over several calls (max_units)")
        return r

    @classmethod
    def max_units(cls, K: int, k_tile: int = 512, slots: int = 3) -> int:
        """The most units one table call takes at this K (a multiple of 6), for splitting."""
        n = cls.MAX_TABLE_PASSES * slots // (K // k_tile)
        return n - n % 6

    @classmethod
    def of(cls, groups: int, K: int, N: int, epilogue_modes=(Epilogue.NONE,), core_rows: int = 4,
           unit_groups=None, **tunables) -> "GroupedGEMM":
        units = groups if unit_groups is None else len(unit_groups)
        if unit_groups is not None:
            tunables["unit_groups"] = tuple(int(g) for g in unit_groups)
        return cls(M=units * 64 * core_rows, K=K, N=N, groups=groups, core_rows=core_rows,
                   epilogue_modes=epilogue_modes,
                   precision=Precision.MEDIUM, b_col_maj=False, bfp16_b=True, **tunables)

    def unit_rows(self, g: int) -> np.ndarray:
        """The ``unit`` rows of A and C that unit ``g`` owns (group g without a table), in slot order."""
        half, h = self.M // 2, self.unit // 2
        return np.r_[g * h:(g + 1) * h, half + g * h:half + (g + 1) * h]

    def pack_groups(self, Bs) -> np.ndarray:
        """Concatenate each group's ``pack_b`` of its ``(K, N)`` weight."""
        return np.concatenate([np.asarray(self.pack_b(b)).reshape(-1) for b in Bs])

    def ops(self) -> int:
        return 2 * self.M * self.K * self.N

    def reference(self, A, B):
        raise NotImplementedError("GroupedGEMM: compare per group with ProjectionGEMM.reference")


def patch_standalone_bfp() -> None:
    """``iron.tune.measure.Standalone`` seeds inputs with ``_sample``, which calls np.dtype()
    on the block-float marker of a packed B and raises. Seed those with valid bfp16ebs8
    blocks of small values (random bytes would be random shared exponents). For benches and
    tests that build a Standalone around a packed-B GEMM. Idempotent."""
    import iron.tune.measure as m
    if getattr(m._sample, "_xdna_relay", False):
        return
    orig = m._sample

    def _sample(dtype, nbytes, rng):
        if not bfp.is_bfp(dtype):
            return orig(dtype, nbytes, rng)
        blocks = nbytes // bfp.BLOCK_BYTES
        out = np.empty((blocks, bfp.BLOCK_BYTES), np.uint8)
        out[:, 0] = 120  # shared exponent ~2^-7
        out[:, 1:] = rng.integers(0, 256, (blocks, bfp.BLOCK_BYTES - 1), dtype=np.uint8)
        return out.reshape(-1)

    _sample._xdna_relay = True
    m._sample = _sample
