"""T-038: online calibration of the placement cost model (sched/online.py).

Pure logic, except the last test (engine on Qwen3-0.6B, iGPU): it checks the server engine
feeds measured prefill parts and decode-step times into the calibrator.
"""
import math

import numpy as np
import pytest

from xdna_relay.sched.online import LinFit, OnlineCosts
from xdna_relay.sched.placement import PlacementPolicy, costs_for

PRESET = "qwen3-8b-packed-batched"


def test_linfit_prior_then_data():
    f = LinFit(1.0, 2.0)
    assert f(3.0) == pytest.approx(7.0)  # no data: the prior
    for _ in range(200):  # true line 0.5 + 1.0 x, two lengths
        for x in (1.0, 5.0):
            f.add(x, 0.5 + 1.0 * x)
    c0, c1 = f.coef()
    assert c0 == pytest.approx(0.5, abs=0.05) and c1 == pytest.approx(1.0, abs=0.02)


def test_linfit_single_length_stays_finite():
    f = LinFit(1.0, 2.0)
    for _ in range(500):
        f.add(4.0, 5.0)  # only one x: the ridge keeps the solution determined
    assert f(4.0) == pytest.approx(5.0, abs=0.05)
    assert all(math.isfinite(c) for c in f.coef())


def test_prior_reproduces_static_preset():
    st = PlacementPolicy(costs_for(PRESET))
    on = PlacementPolicy.calibrating(costs_for(PRESET))
    for n in (85, 401, 1721):
        for d in ("igpu", "npu"):
            assert on.estimate(d, n, 1).ttft_s == pytest.approx(st.estimate(d, n, 1).ttft_s, rel=0.02), (d, n)


def test_learns_a_slower_npu_and_flips_the_decision():
    """A machine whose NPU part is 2x the preset's: after observing it, long prompts go to the
    iGPU (whose own estimate stays right)."""
    truth = OnlineCosts(costs_for(PRESET))  # the preset's own parts, as 'true' functions
    true_npu = lambda k: 2.0 * truth.npu_s(k, 1)  # noqa: E731
    true_igpu = lambda n: truth.igpu_s(n, 1)  # noqa: E731
    p = PlacementPolicy.calibrating(costs_for(PRESET))
    assert p.choose(1721, active=1).device in ("npu", "split")
    rng = np.random.default_rng(0)
    for _ in range(40):
        k = int(rng.integers(300, 2000))
        p.observe_part("npu", k, true_npu(k) * rng.normal(1, 0.03), active=1)
        n = int(rng.integers(1, 2000))
        p.observe_part("igpu", n, true_igpu(n) * rng.normal(1, 0.03), active=1)
    for k in (700, 1500):
        assert p.online.npu_s(k, 1) == pytest.approx(true_npu(k), rel=0.12), k
    assert p.choose(1721, active=1).device == "igpu"


def test_keep_from_request_windows():
    """Keep per placement from whole prefill windows; steps with prefill chunks (the tails of
    NPU/split requests) do not move it, only set no baseline."""
    o = OnlineCosts(costs_for(PRESET))
    assert o.observe_window("igpu", 1.0, 5) is None  # no baseline yet
    for _ in range(20):
        o.observe_step(0.065, prefill_chunk=False, npu_busy=False)
        o.observe_step(0.070, prefill_chunk=True, npu_busy=False)  # a tail step: ignored
    for _ in range(60):
        o.observe_window("igpu", 1.39, 3)  # client-measured case: ~2.2 tok/s of 15.5
        o.observe_window("npu", 1.68, 17)
    assert o.plain_step_s == pytest.approx(0.065, abs=1e-3)
    assert o.keep_igpu.value == pytest.approx(3 * 0.065 / 1.39, abs=0.02)
    assert o.keep_npu.value == pytest.approx(17 * 0.065 / 1.68, abs=0.02)
    assert o.observe_window("split", 1.3, 15) is None


def test_engine_feeds_the_calibrator(igpu):
    from tests.reference.models import QWEN3_06B_Q8, require
    from xdna_relay.server.engine import Engine

    e = Engine(require(QWEN3_06B_Q8), slots=2, n_ctx=4096)
    try:
        assert e.policy.online is not None  # default policy calibrates
        for i in range(3):
            e.complete(e.tokenize(f"<|im_start|>user\\nTell me about rivers, part {i}.<|im_end|>\\n"
                                  "<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n"), 24)
        s = e.policy.online.summary()
        assert s["igpu"][0] + s["igpu:small"][0] >= 3  # three iGPU prefills observed
        assert s["plain_step_s"][0] > 0  # decode steps timed
    finally:
        e.close()


def test_small_tails_have_their_own_fit():
    """Tails after a handoff (1 token, a split's rest) cost about a decode step; learning that
    must not drag the bulk prefill line, and vice versa."""
    p = PlacementPolicy.calibrating(costs_for(PRESET))
    for _ in range(30):
        p.observe_part("igpu", 1, 0.08, active=1)
        p.observe_part("igpu", 26, 0.12, active=1)
        p.observe_part("igpu", 1700, 5.4, active=1)
        p.observe_part("igpu", 400, 1.4, active=1)
    o = p.online
    assert o.igpu_s(1, 1) == pytest.approx(0.08, abs=0.03)
    assert o.igpu_s(26, 1) == pytest.approx(0.12, abs=0.03)
    assert o.igpu_s(1700, 1) == pytest.approx(5.4, rel=0.05)
    assert o.igpu_s(400, 1) == pytest.approx(1.4, rel=0.08)


def test_npu_versions_split_even_with_a_single_curve_prior():
    p = PlacementPolicy.calibrating(costs_for("llama3.2-3b"))  # prior without a short version
    for _ in range(30):
        p.observe_part("npu", 300, 0.9, active=1)  # short version: 2 blocks
        p.observe_part("npu", 1700, 3.2, active=1)  # long: 7 blocks
    assert p.online.npu_s(300, 1) == pytest.approx(0.9, rel=0.08)
    assert p.online.npu_s(1700, 1) == pytest.approx(3.2, rel=0.08)
