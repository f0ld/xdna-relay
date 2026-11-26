"""T-046: GroupedGEMM, many experts' GEMMs in one IRON ProjectionGEMM call.

Unit g (A and C rows ``op.unit_rows(g)``: 128 rows in each half of M) multiplies by group g's packed bfp16 weight. Each
group's result must match its own product (float32 reference, bf16 output) within IRON's
ProjectionGEMM bound for bfp16 macs (``tolerance()``: (2 + K 2^-16 + 2) units of
2^-8 |A| @ |B|), and must NOT match a neighbour's weight: a wrong B stride (stride 0 = the
first expert's weights for every unit, an off-by-one group) fails the test.
"""
import numpy as np
import pytest
from ml_dtypes import bfloat16


def _run(op, A, Bs):
    from iron.tune.measure import Standalone
    from xdna_relay.backends.npu.grouped_gemm import patch_standalone_bfp
    patch_standalone_bfp()
    sa = Standalone(f"t046_{op.groups}_{op.K}_{op.N}_{op.core_rows}_{abs(hash(op.unit_groups)) % 10**8}", [op], load=False)
    packed = op.pack_groups(Bs).view(np.uint8)
    sa._bytes("s0_A")[:A.nbytes] = A.view(np.uint8).reshape(-1)
    sa._bytes("s0_B")[:packed.nbytes] = packed
    sa.load()
    sa.callable()
    return sa.written(0)["C"].astype(np.float32)


def _ref(A, B):
    return (A.astype(np.float32) @ B.astype(np.float32)).astype(bfloat16).astype(np.float32)


def _bound(op, A, B):
    units = 2.0 + op.K * 2.0 ** -16 + 2.0
    return units * 2.0 ** -8 * (np.abs(A.astype(np.float32)) @ np.abs(B.astype(np.float32))) + 1e-6


@pytest.mark.parametrize("groups,K,N,core_rows", [(4, 2048, 1024, 4), (4, 512, 2048, 4), (16, 2048, 1024, 4),
                                                   (4, 2048, 1024, 2), (16, 512, 2048, 2)])
def test_grouped_matches_each_group(npu, groups, K, N, core_rows):
    from xdna_relay.backends.npu.grouped_gemm import GroupedGEMM
    rng = np.random.default_rng(groups + K)
    op = GroupedGEMM.of(groups, K, N, core_rows=core_rows)
    A = (rng.standard_normal((op.M, K)) * 0.5).astype(bfloat16)
    Bs = [(rng.standard_normal((K, N)) * 0.05).astype(bfloat16) for _ in range(groups)]
    C = _run(op, A, Bs)
    for g in range(groups):
        rows = op.unit_rows(g)
        err = np.abs(C[rows] - _ref(A[rows], Bs[g]))
        bound = _bound(op, A[rows], Bs[g])
        assert (err <= bound).all(), (g, float((err / bound).max()))
        wrong = (g + 1) % groups
        assert not (np.abs(C[rows] - _ref(A[rows], Bs[wrong])) <= _bound(op, A[rows], Bs[wrong])).all()


def test_refuses_resident_shapes_or_bad_m():
    from xdna_relay.backends.npu.grouped_gemm import GroupedGEMM
    with pytest.raises(ValueError):
        GroupedGEMM(M=512, K=2048, N=1024, groups=4, precision="medium", bfp16_b=True).resolved()


@pytest.mark.parametrize("table,K,N,core_rows", [((2, 0, 2, 3, 1, 0), 2048, 1024, 2), ((3, 3, 3, 1), 512, 2048, 2),
                                                 ((1, 0, 1), 2048, 1024, 4)])
def test_unit_table(npu, table, K, N, core_rows):
    """Units map to groups through a table: repeated and skipped groups, any order."""
    from xdna_relay.backends.npu.grouped_gemm import GroupedGEMM
    groups = 4
    rng = np.random.default_rng(len(table) + K)
    op = GroupedGEMM.of(groups, K, N, core_rows=core_rows, unit_groups=table)
    A = (rng.standard_normal((op.M, K)) * 0.5).astype(bfloat16)
    Bs = [(rng.standard_normal((K, N)) * 0.05).astype(bfloat16) for _ in range(groups)]
    C = _run(op, A, Bs)
    for u, g in enumerate(table):
        rows = op.unit_rows(u)
        err = np.abs(C[rows] - _ref(A[rows], Bs[g]))
        assert (err <= _bound(op, A[rows], Bs[g])).all(), (u, g)
