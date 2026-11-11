"""T-002: op contract + backend registry skeleton, `ref` backend for gemm_bf16.

Contracts under test (see tasks/T-002-op-registry.md for the full interface):
    src/xdna_relay/ops/contract.py  DTYPES, TensorSpec, Tolerance, OpContract
    src/xdna_relay/ops/registry.py  Backend, Registry, DEFAULT
    src/xdna_relay/ops/gemm_bf16.py GEMM_BF16, gemm_bf16_ref (registered in DEFAULT)
No hardware needed.
"""

import pytest
import torch

from tests.reference.gemm import gemm_ref


# ---------------------------------------------------------------- helpers


def _contract_mod():
    from xdna_relay.ops import contract

    return contract


def _registry_mod():
    from xdna_relay.ops import registry

    return registry


def _add_contract():
    """Toy contract: out = x + y * alpha, x/y (N, D) f32, out (N, D) f32."""
    c = _contract_mod()
    return c.OpContract(
        name="toy_add",
        inputs=(
            c.TensorSpec("x", ("N", "D"), "f32", "row_major"),
            c.TensorSpec("y", ("N", "D"), "f32", "row_major"),
        ),
        outputs=(c.TensorSpec("out", ("N", "D"), "f32", "row_major"),),
        params=("alpha",),
        tolerance=c.Tolerance(atol=1e-6, rtol=1e-6),
    )


def _toy_impl(x, y, *, alpha):
    return x + y * alpha


# ---------------------------------------------------------------- contract.py


def test_dtypes_table():
    c = _contract_mod()
    assert c.DTYPES["bf16"] is torch.bfloat16
    assert c.DTYPES["f32"] is torch.float32
    assert c.DTYPES["f16"] is torch.float16
    assert c.DTYPES["i32"] is torch.int32


def test_tensorspec_is_frozen():
    c = _contract_mod()
    s = c.TensorSpec("a", ("M", 4), "bf16", "row_major")
    with pytest.raises(Exception):
        s.name = "b"


@pytest.mark.parametrize(
    "shape,dtype,layout",
    [
        (("M", 0), "bf16", "row_major"),  # non-positive int dim
        (("M", -3), "bf16", "row_major"),
        (("M", ""), "bf16", "row_major"),  # empty symbol
        (("M", 2.5), "bf16", "row_major"),  # neither str nor int
        (("M", "K"), "bfloat17", "row_major"),  # unknown dtype
        (("M", "K"), "bf16", ""),  # empty layout
    ],
)
def test_tensorspec_rejects_bad_fields(shape, dtype, layout):
    c = _contract_mod()
    with pytest.raises(ValueError):
        c.TensorSpec("a", shape, dtype, layout)


def test_tolerance_allclose():
    c = _contract_mod()
    t = c.Tolerance(atol=0.1, rtol=0.0)
    e = torch.tensor([1.0, 2.0, 3.0])
    assert t.allclose(e + 0.05, e)
    assert not t.allclose(e + 0.2, e)
    # works across dtypes (compares in float32 or wider)
    assert t.allclose(e.to(torch.bfloat16), e.to(torch.float64))
    # shape mismatch is "not close", not an exception
    assert not t.allclose(torch.zeros(3), torch.zeros(4))
    with pytest.raises(ValueError):
        c.Tolerance(atol=-1.0, rtol=0.0)


def test_contract_rejects_output_only_symbol():
    c = _contract_mod()
    with pytest.raises(ValueError):
        c.OpContract(
            name="bad",
            inputs=(c.TensorSpec("x", ("N",), "f32", "row_major"),),
            outputs=(c.TensorSpec("y", ("Q",), "f32", "row_major"),),
            params=(),
            tolerance=c.Tolerance(atol=0.0, rtol=0.0),
        )


def test_contract_rejects_duplicate_tensor_names():
    c = _contract_mod()
    with pytest.raises(ValueError):
        c.OpContract(
            name="bad",
            inputs=(
                c.TensorSpec("x", ("N",), "f32", "row_major"),
                c.TensorSpec("x", ("N",), "f32", "row_major"),
            ),
            outputs=(c.TensorSpec("y", ("N",), "f32", "row_major"),),
            params=(),
            tolerance=c.Tolerance(atol=0.0, rtol=0.0),
        )


def test_bind_shapes_ok():
    k = _add_contract()
    x = torch.zeros(3, 5)
    assert k.bind_shapes(x, x) == {"N": 3, "D": 5}
    assert k.output_shapes({"N": 3, "D": 5}) == ((3, 5),)


def test_bind_shapes_literal_dims():
    c = _contract_mod()
    k = c.OpContract(
        name="lit",
        inputs=(c.TensorSpec("x", ("N", 4), "f32", "row_major"),),
        outputs=(c.TensorSpec("y", (4, "N"), "f32", "row_major"),),
        params=(),
        tolerance=c.Tolerance(atol=0.0, rtol=0.0),
    )
    assert k.bind_shapes(torch.zeros(7, 4)) == {"N": 7}
    assert k.output_shapes({"N": 7}) == ((4, 7),)
    with pytest.raises(ValueError):
        k.bind_shapes(torch.zeros(7, 5))  # literal dim mismatch


@pytest.mark.parametrize(
    "x,y",
    [
        (torch.zeros(3, 5), torch.zeros(3, 6)),  # symbol D bound twice differently
        (torch.zeros(3, 5), torch.zeros(3, 5, 1)),  # wrong ndim
        (torch.zeros(3, 5), torch.zeros(3, 5, dtype=torch.bfloat16)),  # wrong dtype
        (torch.zeros(3, 5), [[0.0] * 5] * 3),  # not a tensor
    ],
)
def test_bind_shapes_rejects(x, y):
    k = _add_contract()
    with pytest.raises(ValueError):
        k.bind_shapes(x, y)


def test_bind_shapes_rejects_wrong_count():
    k = _add_contract()
    with pytest.raises(ValueError):
        k.bind_shapes(torch.zeros(3, 5))


# ---------------------------------------------------------------- registry.py


def test_registry_basic_flow():
    r = _registry_mod().Registry()
    k = _add_contract()
    r.add_contract(k)
    assert r.ops() == ("toy_add",)
    assert r.contract("toy_add") is k
    assert r.backends("toy_add") == ()
    r.register("toy_add", "ref", _toy_impl)
    r.register("toy_add", "alt", _toy_impl, prepare=lambda w: w)
    assert r.backends("toy_add") == ("alt", "ref")  # sorted
    b = r.backend("toy_add", "ref")
    assert b.name == "ref" and b.impl is _toy_impl and b.prepare is None
    assert r.backend("toy_add", "alt").prepare is not None


def test_registry_errors():
    reg = _registry_mod()
    r = reg.Registry()
    k = _add_contract()
    r.add_contract(k)
    with pytest.raises(ValueError):
        r.add_contract(k)  # duplicate contract name
    with pytest.raises(KeyError):
        r.contract("nope")
    with pytest.raises(KeyError):
        r.register("nope", "ref", _toy_impl)  # unknown op
    r.register("toy_add", "ref", _toy_impl)
    with pytest.raises(ValueError):
        r.register("toy_add", "ref", _toy_impl)  # duplicate backend
    with pytest.raises(KeyError):
        r.backend("toy_add", "npu")
    with pytest.raises(KeyError):
        r.backends("nope")


def test_registries_are_independent():
    reg = _registry_mod()
    r1, r2 = reg.Registry(), reg.Registry()
    r1.add_contract(_add_contract())
    assert r2.ops() == ()


def test_run_validates_and_returns_single_output():
    r = _registry_mod().Registry()
    r.add_contract(_add_contract())
    r.register("toy_add", "ref", _toy_impl)
    x, y = torch.ones(2, 3), torch.full((2, 3), 2.0)
    out = r.run("toy_add", "ref", x, y, alpha=0.5)
    assert isinstance(out, torch.Tensor)
    torch.testing.assert_close(out, torch.full((2, 3), 2.0))


def test_run_rejects_bad_params():
    r = _registry_mod().Registry()
    r.add_contract(_add_contract())
    r.register("toy_add", "ref", _toy_impl)
    x = torch.ones(2, 3)
    with pytest.raises(ValueError):
        r.run("toy_add", "ref", x, x)  # missing alpha
    with pytest.raises(ValueError):
        r.run("toy_add", "ref", x, x, alpha=1.0, beta=2.0)  # extra param
    with pytest.raises(ValueError):
        r.run("toy_add", "ref", x, torch.ones(2, 4), alpha=1.0)  # bad shape


@pytest.mark.parametrize(
    "bad_impl",
    [
        lambda x, y, *, alpha: x[:1],  # wrong shape
        lambda x, y, *, alpha: x.to(torch.float64),  # wrong dtype
        lambda x, y, *, alpha: (x, y),  # wrong output count
        lambda x, y, *, alpha: x.tolist(),  # not a tensor
    ],
)
def test_run_rejects_bad_backend_output(bad_impl):
    r = _registry_mod().Registry()
    r.add_contract(_add_contract())
    r.register("toy_add", "bad", bad_impl)
    x = torch.ones(2, 3)
    with pytest.raises(RuntimeError):
        r.run("toy_add", "bad", x, x, alpha=1.0)


def test_run_multi_output_returns_tuple():
    c = _contract_mod()
    r = _registry_mod().Registry()
    r.add_contract(
        c.OpContract(
            name="split",
            inputs=(c.TensorSpec("x", ("N",), "f32", "row_major"),),
            outputs=(
                c.TensorSpec("a", ("N",), "f32", "row_major"),
                c.TensorSpec("b", ("N",), "f32", "row_major"),
            ),
            params=(),
            tolerance=c.Tolerance(atol=0.0, rtol=0.0),
        )
    )
    r.register("split", "ref", lambda x: (x * 2, x * 3))
    out = r.run("split", "ref", torch.ones(4))
    assert isinstance(out, tuple) and len(out) == 2
    torch.testing.assert_close(out[1], torch.full((4,), 3.0))


# ---------------------------------------------------------------- gemm_bf16


def test_gemm_bf16_contract_shape():
    from xdna_relay.ops.gemm_bf16 import GEMM_BF16

    assert GEMM_BF16.name == "gemm_bf16"
    assert [(s.name, s.shape, s.dtype, s.layout) for s in GEMM_BF16.inputs] == [
        ("A", ("M", "K"), "bf16", "row_major"),
        ("B", ("K", "N"), "bf16", "row_major"),
    ]
    assert [(s.name, s.shape, s.dtype, s.layout) for s in GEMM_BF16.outputs] == [
        ("C", ("M", "N"), "bf16", "row_major"),
    ]
    assert GEMM_BF16.params == ()
    assert GEMM_BF16.tolerance.atol == 1e-2
    assert GEMM_BF16.tolerance.rtol == 2e-2


def test_importing_ops_registers_gemm_bf16_ref():
    import xdna_relay.ops  # noqa: F401  (import side effect fills DEFAULT)
    from xdna_relay.ops.gemm_bf16 import GEMM_BF16, gemm_bf16_ref
    from xdna_relay.ops.registry import DEFAULT

    assert "gemm_bf16" in DEFAULT.ops()
    assert DEFAULT.contract("gemm_bf16") is GEMM_BF16
    assert "ref" in DEFAULT.backends("gemm_bf16")
    assert DEFAULT.backend("gemm_bf16", "ref").impl is gemm_bf16_ref
    # importing twice must not raise (no double registration)
    import importlib

    importlib.import_module("xdna_relay.ops")


@pytest.mark.parametrize(
    "m,k,n",
    [(1, 1, 1), (32, 64, 48), (128, 256, 128), (64, 2048, 96)],
)
def test_gemm_bf16_ref_matches_oracle(m, k, n):
    import xdna_relay.ops  # noqa: F401
    from xdna_relay.ops.gemm_bf16 import GEMM_BF16
    from xdna_relay.ops.registry import DEFAULT

    g = torch.Generator().manual_seed(1000 * m + k + n)
    # scale so |C| stays O(1) for every K
    a = (torch.randn(m, k, generator=g) / k**0.5).to(torch.bfloat16)
    b = torch.randn(k, n, generator=g).to(torch.bfloat16)
    c = DEFAULT.run("gemm_bf16", "ref", a, b)
    assert c.dtype is torch.bfloat16 and c.shape == (m, n)
    expected = gemm_ref(a, b)
    assert GEMM_BF16.tolerance.allclose(c, expected), (
        f"max abs err {(c.double() - expected).abs().max().item():.3e}"
    )
    # ref accumulates in float32 or wider: error is only the final bf16 rounding
    rel = ((c.double() - expected).abs() / expected.abs().clamp_min(1e-3)).max().item()
    assert rel <= 2**-8, f"ref backend error {rel:.3e} > one bf16 ulp; accumulate in f32"


def test_gemm_bf16_ref_rejects_bad_input():
    import xdna_relay.ops  # noqa: F401
    from xdna_relay.ops.registry import DEFAULT

    a = torch.zeros(4, 8, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        DEFAULT.run("gemm_bf16", "ref", a, torch.zeros(7, 3, dtype=torch.bfloat16))  # K mismatch
    with pytest.raises(ValueError):
        DEFAULT.run("gemm_bf16", "ref", a, torch.zeros(8, 3))  # f32 B
