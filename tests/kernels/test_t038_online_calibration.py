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
    o = p.online
    for _ in range(5):
        o.observe_step(0.065, prefill_chunk=False, npu_busy=False)
    keep = {"npu": 0.64, "igpu": 0.067}  # the preset's: the others keep this share
    for _ in range(40):  # what the engine reports per request: parts and the whole window
        k = int(rng.integers(300, 2000))
        t = true_npu(k) * rng.normal(1, 0.03)
        p.observe_part("npu", k, t, active=1)
        o.observe_window("npu", t, round(t * keep["npu"] / 0.065), n=k)
        n = int(rng.integers(65, 2000))
        t = true_igpu(n) * rng.normal(1, 0.03)
        p.observe_part("igpu", n, t, active=1)
        o.observe_window("igpu", t, round(t * keep["igpu"] / 0.065), n=n)
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
    assert o.observe_window("split", 1.3, 15, n=256) is None  # tails vary: not used


def test_lost_time_fitted_per_size():
    """Short iGPU prefills cost the others little, long ones a lot: fitted per size, so both
    are right (one keep fraction could not be)."""
    o = OnlineCosts(costs_for(PRESET))
    for _ in range(5):
        o.observe_step(0.065, prefill_chunk=False, npu_busy=False)
    for _ in range(40):
        o.observe_window("igpu", 0.45, 3, n=83)  # lost 0.255
        o.observe_window("igpu", 1.45, 4, n=398)  # lost 1.19
        o.observe_window("igpu", 5.40, 11, n=1718)  # lost 4.685
        o.observe_window("npu", 1.68, 17, n=397)  # lost 0.575
    assert o.igpu_lost_s(398) == pytest.approx(1.19, rel=0.1)
    assert o.igpu_lost_s(1718) == pytest.approx(4.685, rel=0.05)
    assert o.npu_lost_s(397) == pytest.approx(0.575, rel=0.1)


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


def test_explores_a_near_tie_it_never_observed():
    """A pessimistic, unobserved option within the margin is tried; once observed, the
    estimated best wins again unless the observation says otherwise."""
    p = PlacementPolicy.calibrating(costs_for(PRESET))
    c = p.choose(1721, active=1)
    best = min(("igpu", "npu", "split"), key=lambda d: p.estimate(d, 1721, 1).cost)
    near = [d for d in ("igpu", "npu", "split") if d != best
            and p.estimate(d, 1721, 1).cost <= 1.2 * p.estimate(best, 1721, 1).cost]
    if near:
        assert c.explore and c.device in near
    for _ in range(3):  # observe every part at this size
        p.observe_part("igpu", 1721, 5.4, active=1)
        p.observe_part("npu", 1720, 4.4, active=1)
        p.observe_part("npu", 1536, 4.0, active=1)
    c = p.choose(1721, active=1)
    assert not c.explore


def test_no_exploration_without_decoders_or_calibration():
    st = PlacementPolicy(costs_for(PRESET))
    assert not st.choose(1721, active=1).explore
    on = PlacementPolicy.calibrating(costs_for(PRESET))
    assert not on.choose(1721, active=0).explore


def test_npu_option_loss_not_double_counted():
    """Measured at 398 tokens: NPU TTFT 1.55 s, others lose 0.48 s; iGPU 1.36 s, others lose
    1.17 s. After observing both, the NPU option must win (the tail's loss is inside the NPU
    window and must not be added from the never-observed iGPU tail prior)."""
    p = PlacementPolicy.calibrating(costs_for("llama3.2-3b"), explore_margin=0.0)
    o = p.online
    for _ in range(5):
        o.observe_step(0.065, prefill_chunk=False, npu_busy=False)
    for _ in range(20):
        p.observe_part("npu", 397, 1.32, active=1)
        p.observe_part("igpu", 1, 0.23, active=1)
        o.observe_window("npu", 1.55, round((1.55 - 0.48) / 0.065), n=397)
        p.observe_part("igpu", 398, 1.36, active=1)
        o.observe_window("igpu", 1.36, round((1.36 - 1.17) / 0.065), n=398)
    c = p.choose(398, active=1)
    assert c.device == "npu", (c, p.estimate("split", 398, 1))


def test_bins_correct_a_nonlinear_short_end():
    """Batched iGPU prefill in 256-token chunks: 83 tokens took 0.41 s although the line
    through long prompts says ~0.6; once observed, the bin predicts what was measured, and
    unobserved sizes still follow the line."""
    p = PlacementPolicy.calibrating(costs_for(PRESET))
    meas = {83: 0.41, 398: 1.37, 968: 3.05, 1718: 5.45}
    for _ in range(15):
        for n, t in meas.items():
            p.observe_part("igpu", n, t, active=1)
    o = p.online
    for n, t in meas.items():
        assert o.igpu_s(n, 1) == pytest.approx(t, rel=0.08), n
    line = o.igpu[(True, False)].line(1300 / 1000)
    assert o.igpu_s(1300, 1) == pytest.approx(line)  # bin never observed: the line
