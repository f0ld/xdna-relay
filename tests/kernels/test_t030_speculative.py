"""T-030: speculative greedy decoding (xdna_relay.sched.speculative) and the AdaptiveK policy.

Equality rule: greedy speculative output must equal plain greedy output token for token,
except that it may diverge at a near-tie: at the first differing position, the two candidate
tokens' target logits (prefix evaluated in one batch) differ by < NEAR_TIE. Batched verify
and single-token decode round differently (measured gaps at divergences 0.03–0.17 on
Qwen3-8B, bench/results/2026-11-24-relay-spec.md; stock llama.cpp shows the same).
"""
import numpy as np
import pytest

from xdna_relay.sched.speculative import AdaptiveK, generate
from tests.reference.models import QWEN3_06B_Q8, QWEN3_8B_Q4, qwen3_chat, require

NEAR_TIE = 0.25
PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Explain how TCP slow start works in one paragraph.",
]


# ---- policy (pure Python) -------------------------------------------------------------

def test_expected_tokens():
    pol = AdaptiveK(k_max=4, p=[1.0, 1.0, 1.0, 1.0])
    assert pol.expected_tokens(0) == 1.0
    assert pol.expected_tokens(3) == 4.0
    pol = AdaptiveK(k_max=3, p=[0.5, 0.5, 0.5])
    assert pol.expected_tokens(2) == pytest.approx(1 + 0.5 + 0.25)


def _warm(pol, t_first, t_step, t_v0):
    pol.observe(0, 0, 0.0, 0.0, t_v0)
    pol.observe(2, 1, t_first, t_step, t_v0 * 1.06)


def test_policy_drafts_long_when_acceptance_high_and_draft_cheap():
    pol = AdaptiveK(k_max=8)
    _warm(pol, 0.005, 0.004, 0.064)
    pol.p = [0.98] * 8
    assert pol.choose() >= 6


def test_policy_stops_drafting_when_acceptance_zero_but_probes():
    pol = AdaptiveK(k_max=8, probe_every=4, k_init=3)
    _warm(pol, 0.02, 0.015, 0.064)
    pol.p = [0.0] * 8
    ks = [pol.choose() for _ in range(8)]
    assert ks.count(0) == 6 and ks.count(3) == 2


def test_policy_ignores_long_catch_up_in_first_call_cost():
    pol = AdaptiveK(k_max=8, ema=0.5)
    _warm(pol, 0.010, 0.010, 0.064)
    pol.observe(3, 3, 0.200, 0.020, 0.070, catch_up=20)  # draft lagged 20 tokens
    assert pol.t_first == pytest.approx(0.010)
    pol.observe(3, 3, 0.012, 0.020, 0.070, catch_up=2)
    assert pol.t_first == pytest.approx(0.011)


def test_policy_fixed_and_warmup():
    assert AdaptiveK(fixed=5).choose() == 5
    pol = AdaptiveK(k_init=3)
    assert pol.choose() == 0  # first round measures a plain step
    pol.observe(0, 0, 0, 0, 0.06)
    assert pol.choose() == 3


def test_policy_learns_per_position_acceptance():
    pol = AdaptiveK(k_max=4, ema=1.0)
    pol.observe(4, 2, 0.01, 0.03, 0.07)  # positions 0,1 accepted, 2 rejected, 3 unseen
    assert pol.p[:3] == [1.0, 1.0, 0.0] and pol.p[3] == pol.p_prior


# ---- hardware: speculative == plain greedy (modulo near-ties) ------------------------

@pytest.fixture(scope="module")
def models(igpu):
    from xdna_relay.backends.igpu.llama import LlamaCppExecutor

    tgt = LlamaCppExecutor(require(QWEN3_8B_Q4), n_ctx=2048)
    dft = LlamaCppExecutor(require(QWEN3_06B_Q8), n_ctx=2048)
    yield tgt, dft
    tgt.close()
    dft.close()


def _check_equal_or_near_tie(tgt, prompt, plain, spec):
    p = next((i for i, (a, b) in enumerate(zip(plain, spec)) if a != b), None)
    if p is None:
        n = min(len(plain), len(spec))
        assert plain[:n] == spec[:n]
        # same tokens; lengths may differ only by the budget, never past an EOG
        assert len(plain) == len(spec) or (tgt.is_eog(plain[-1]) == tgt.is_eog(spec[-1]))
        return
    tgt.reset()
    tgt.eval(prompt + plain[:p])
    l = tgt.logits(0)
    gap = abs(float(l[plain[p]] - l[spec[p]]))
    assert gap < NEAR_TIE, f"divergence at {p} is not a near-tie: gap {gap:.3f}"


@pytest.mark.parametrize("text", PROMPTS)
@pytest.mark.parametrize("k", ["adaptive", 1, 4])
def test_speculative_matches_plain_greedy(models, text, k):
    tgt, dft = models
    prompt = tgt.tokenize(qwen3_chat(text), add_special=False)
    plain = generate(tgt, prompt, 96).tokens
    pol = AdaptiveK(fixed=None if k == "adaptive" else k)
    st = generate(tgt, prompt, 96, dft, pol)
    assert len(st.tokens) <= 96
    assert st.accepted <= st.drafted
    if k != "adaptive":
        # fixed k, shortened only when fewer than k tokens are left in the budget
        assert st.ks[0] == k and all(1 <= x <= k for x in st.ks)
    _check_equal_or_near_tie(tgt, prompt, plain, st.tokens)


def test_stops_at_eog_inside_accepted_drafts(models):
    tgt, dft = models
    prompt = tgt.tokenize(qwen3_chat("Reply with exactly the word: yes"), add_special=False)
    st = generate(tgt, prompt, 64, dft, AdaptiveK(fixed=6))
    eog = [i for i, t in enumerate(st.tokens) if tgt.is_eog(t)]
    assert eog, "expected an end-of-generation token within 64 tokens"
    assert eog[0] == len(st.tokens) - 1, "tokens emitted after EOG"


def test_speculative_is_faster(models):
    """Not a correctness property, but the point of the feature: on a code prompt the
    adaptive policy must beat plain decoding by >= 1.3x (measured ~1.9x)."""
    tgt, dft = models
    prompt = tgt.tokenize(qwen3_chat(PROMPTS[0]), add_special=False)
    plain = generate(tgt, prompt, 128)
    spec = generate(tgt, prompt, 128, dft, AdaptiveK())
    assert spec.decode_tps >= 1.3 * plain.decode_tps, (plain.decode_tps, spec.decode_tps)
