"""T-041: speculative decoding in the batched engine (BatchK policy; Engine with a draft model).

Equality rule as T-030: each request's output equals plain greedy decoding of its prompt, or
first differs at a near-tie (the two candidates' target logits differ by < NEAR_TIE).
Hardware pair: Qwen3-8B Q4_K_M target, Qwen3-0.6B Q8_0 draft (the draft is often rejected,
so rollback is exercised).
"""
import threading

import pytest

from xdna_relay.sched.speculative import Acceptance, BatchK, _interp, passes
from tests.reference.models import QWEN3_06B_Q8, QWEN3_8B_Q4, qwen3_chat, require

NEAR_TIE = 0.25
PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Explain how TCP slow start works in one paragraph.",
]


# ---- policy (pure Python) -------------------------------------------------------------

def test_interp_between_and_outside_measured_counts():
    d = {2: 0.060, 10: 0.070}
    assert _interp(d, 6, 0.02) == pytest.approx(0.065)
    assert _interp(d, 2, 0.02) == 0.060
    assert _interp(d, 20, 0.02) == pytest.approx(0.070 * 1.4 / 1.2)
    assert _interp({}, 3, 0.02) is None


def test_acceptance_expected_and_observe():
    a = Acceptance([1.0, 1.0, 0.5])
    assert a.expected(0) == 1.0 and a.expected(2) == 3.0 and a.expected(3) == 3.5
    a.observe(3, 1)  # position 0 accepted, position 1 rejected, position 2 not reached
    assert a.p[0] == 1.0 and a.p[1] < 1.0 and a.p[2] == 0.5


def _flat(pol, n_max=64, base=0.064, slope=0.0005, d0=0.006, dslope=0.0002):
    pol.verify = {t: base + slope * t for t in range(1, n_max)}
    pol.draft = {t: d0 + dslope * t for t in range(1, n_max)}


def test_warmup_then_init():
    pol = BatchK(k_init=3)
    accs = [pol.new_sequence(), pol.new_sequence()]
    assert pol.choose(accs, [1, 1]) == [0, 0]
    pol.verify[2] = 0.065
    assert pol.choose(accs, [1, 1]) == [3, 3]


def test_drafts_long_when_acceptance_high_and_verify_flat():
    pol = BatchK(k_max=8)
    _flat(pol)
    accs = [Acceptance([0.95] * 8)]
    assert pol.choose(accs, [1])[0] >= 6


def test_shorter_k_when_batch_makes_verify_rows_costly():
    """Same acceptance; verify costs a lot per row (compute bound). 1 sequence vs 4: the
    batch of 4 must not pick a longer k, and picks a strictly shorter one here."""
    pol = BatchK(k_max=8)
    pol.verify = {t: 0.060 + 0.004 * t for t in range(1, 64)}
    pol.draft = {t: 0.004 + 0.0002 * t for t in range(1, 64)}
    k1 = pol.choose([Acceptance([0.8] * 8)], [1])[0]
    k4 = max(pol.choose([Acceptance([0.8] * 8) for _ in range(4)], [1] * 4))
    assert k1 > 0 and k4 < k1, (k1, k4)


def test_low_acceptance_sequence_does_not_draft():
    pol = BatchK(k_max=6)
    _flat(pol, slope=0.002)
    ks = pol.choose([Acceptance([0.95] * 6), Acceptance([0.02] * 6)], [1, 1])
    assert ks[0] > 0 and ks[1] == 0, ks


def test_zero_acceptance_stops_but_probes():
    pol = BatchK(k_max=4, probe_every=4, k_init=2)
    _flat(pol, slope=0.002)
    accs = [Acceptance([0.0] * 4)]
    picks = [pol.choose(accs, [1])[0] for _ in range(8)]
    assert picks.count(0) == 6 and picks.count(2) == 2, picks


def test_observe_keys_costs_by_tokens_and_pools_acceptance():
    pol = BatchK(k_max=4, p_prior=0.5, split_ragged=True)
    accs = [pol.new_sequence(), pol.new_sequence()]
    pol.observe([3, 0], [3, 0], accs, t_first=0.010, first_tokens=40, t_rest=0.010,
                t_verify=0.070, verify_shape=[4, 1])
    assert pol.verify == {}  # a split verify (2 passes) is not attributed
    assert pol.draft[40] == 0.010 and pol.draft[1] == pytest.approx(0.005)  # 2 further calls, 1 seq
    assert accs[0].p[0] > 0.5 and accs[1].p == [0.5] * 4  # the non-drafting one unchanged
    assert pol.p[0] > 0.5 and pol.new_sequence().p[0] == pol.p[0]
    pol.observe([1], [1], accs[:1], t_first=0.01, first_tokens=1, t_rest=0.0, t_verify=0.07, verify_shape=[2])
    assert pol.verify == {2: 0.07}


def test_passes():
    assert passes([1, 1, 1, 1]) == [4]
    assert passes([2, 1, 1, 1]) == [4, 1]
    assert passes([3, 3, 1, 1]) == [4, 4]
    assert passes([4, 1, 0]) == [2, 3]
    assert passes([]) == []


def test_split_ragged_prefers_even_shapes():
    """Measured on the 890M (per-sequence KV): [2,1,1,1] costs ~2 passes. With a strong and a
    weak drafter, drafting only the strong one is best in one pass; with split passes the
    policy drafts both or neither."""
    def pol(split):
        b = BatchK(k_max=4, split_ragged=split)
        b.verify = {t: 0.065 + 0.003 * t for t in range(1, 64)}
        b.draft = {t: 0.012 + 0.0003 * t for t in range(1, 64)}
        return b
    accs = [Acceptance([0.9] * 4), Acceptance([0.3] * 4)]
    ks = pol(False).choose(accs, [1, 1])
    assert ks[0] > 0 and ks[1] == 0, ks
    ks = pol(True).choose(accs, [1, 1])
    assert ks[0] == ks[1], ks


def test_split_verify_cost_sums_passes():
    b = BatchK(split_ragged=True)
    b.verify = {1: 0.065, 3: 0.068, 4: 0.072}
    assert b.verify_cost([2, 1, 1, 1]) == pytest.approx(0.072 + 0.065)
    assert b.verify_cost([1, 1, 1, 1]) == pytest.approx(0.072)


def test_outcome_check_overrides_model():
    """The model says draft; measured rounds at this batch size say drafting is slower: plain
    rounds, except every explore_every-th round, which drafts to re-measure."""
    pol = BatchK(k_max=4, explore_every=4)
    _flat(pol)
    accs = [Acceptance([0.9] * 4) for _ in range(4)]
    assert any(pol.choose(accs, [1] * 4))
    for _ in range(3):
        pol.observe_round(4, True, 6, 0.120)  # 50 tok/s
        pol.observe_round(4, False, 4, 0.074)  # 54 tok/s
    picks = [pol.choose(accs, [1] * 4) for _ in range(8)]
    assert sum(1 for p in picks if any(p)) == 2, picks  # rounds 4 and 8 of this size explore
    pol.observe_round(4, True, 8, 0.110)  # drafting got faster
    for _ in range(20):
        pol.observe_round(4, True, 8, 0.110)
    assert any(pol.choose(accs, [1] * 4))
    # other sizes are not affected; rounds with a prefill chunk skip the check
    assert any(pol.choose(accs[:2], [1] * 2))


def test_fixed():
    pol = BatchK(fixed=4)
    assert pol.choose([Acceptance([0.0] * 8)] * 2, [1, 1]) == [4, 4]


# ---- hardware: the engine with a draft model -------------------------------------------

@pytest.fixture(scope="module")
def engine(igpu):
    from xdna_relay.server.engine import Engine
    e = Engine(require(QWEN3_8B_Q4), slots=2, n_ctx=2048, draft_gguf=require(QWEN3_06B_Q8))
    yield e
    e.close()


@pytest.fixture(scope="module")
def ref(igpu):
    from xdna_relay.backends.igpu.llama import LlamaCppExecutor
    from xdna_relay.sched.speculative import generate
    tgt = LlamaCppExecutor(require(QWEN3_8B_Q4), n_ctx=2048)
    memo = {}

    def plain(prompt, n):
        key = (tuple(prompt), n)
        if key not in memo:
            memo[key] = generate(tgt, prompt, n).tokens
        return memo[key]

    yield tgt, plain
    tgt.close()


def _check(ref, prompt, got, n):
    tgt, plain = ref
    want = plain(prompt, n)
    p = next((i for i, (a, b) in enumerate(zip(want, got)) if a != b), None)
    if p is None:
        m = min(len(want), len(got))
        assert want[:m] == got[:m]
        assert len(want) == len(got) or tgt.is_eog(want[-1]) == tgt.is_eog(got[-1])
        return
    tgt.reset()
    tgt.eval(prompt + want[:p])
    l = tgt.logits(0)
    gap = abs(float(l[want[p]] - l[got[p]]))
    assert gap < NEAR_TIE, f"divergence at {p} is not a near-tie: gap {gap:.3f}"


def _prompt(engine, text):
    return engine.tokenize(qwen3_chat(text))


def _concurrent(engine, prompts, n):
    out = [None] * len(prompts)

    def run(i):
        out[i] = engine.complete(prompts[i], n)

    ts = [threading.Thread(target=run, args=(i,)) for i in range(len(prompts))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out


def test_engine_is_batched_with_draft(engine):
    assert engine.batched and engine.dshared is not None and engine.spec is not None


@pytest.mark.parametrize("k", [None, 4])
def test_concurrent_speculative_matches_plain(engine, ref, k):
    engine.spec.fixed = k
    try:
        engine.reset_state()
        ps = [_prompt(engine, t) for t in PROMPTS]
        res = _concurrent(engine, ps, 96)
    finally:
        engine.spec.fixed = None
    for p, r in zip(ps, res):
        st = r.stats
        assert len(r.tokens) <= 96 and st.drafted > 0 and 0 <= st.accepted <= st.drafted
        if k is not None:
            assert all(x <= k for x in st.ks)
        _check(ref, p, r.tokens, 96)


def test_streamed_tokens_equal_result(engine):
    seen = []
    r = engine.complete(_prompt(engine, PROMPTS[1]), 40, on_token=lambda t: seen.append(t) and False)
    assert seen == r.tokens


def test_eog_inside_accepted_drafts(engine):
    engine.spec.fixed = 6
    try:
        r = engine.complete(_prompt(engine, "Reply with exactly the word: yes"), 64)
    finally:
        engine.spec.fixed = None
    eog = [i for i, t in enumerate(r.tokens) if engine.is_eog(t)]
    assert eog and eog[0] == len(r.tokens) - 1, r.tokens


@pytest.mark.parametrize("n", [1, 2, 5])
def test_budget_never_exceeded(engine, n):
    engine.spec.fixed = 6
    try:
        r = engine.complete(_prompt(engine, PROMPTS[0]), n)
    finally:
        engine.spec.fixed = None
    assert len(r.tokens) == n


def test_stop_callback_stops_mid_round(engine):
    engine.spec.fixed = 6
    seen = []
    try:
        r = engine.complete(_prompt(engine, PROMPTS[0]), 64, on_token=lambda t: (seen.append(t), len(seen) >= 3)[1])
    finally:
        engine.spec.fixed = None
    assert len(seen) == 3 and r.tokens == seen and r.stopped


def test_draft_catches_up_when_it_lags(engine, ref):
    """The target reuses its live sequence while the draft holds nothing (as after an NPU
    prefill): the draft catches up in chunks, then drafts; output unchanged."""
    p = _prompt(engine, PROMPTS[0])
    engine.complete(p, 8)
    for s in engine.slots:
        s.draft.reset()
    engine.cache = type(engine.cache)(max_bytes=engine.cache.max_bytes)  # no draft snapshot either
    r = engine.complete(p, 96)
    assert r.reused > 0 and r.stats.drafted > 0
    _check(ref, p, r.tokens, 96)


def test_batched_speculation_is_faster(engine):
    """The point of the feature: two concurrent code requests, adaptive vs no drafting (k=0
    on the same engine): total decode tokens/s >= 1.2x (measured: see bench results)."""
    ps = [_prompt(engine, PROMPTS[0]), _prompt(engine, PROMPTS[0].replace("n-th", "k-th"))]

    def tps(k):
        engine.spec.fixed = k
        try:
            engine.reset_state()
            _concurrent(engine, ps, 8)  # prefill both (cached), measure decode below
            res = _concurrent(engine, ps, 128)
        finally:
            engine.spec.fixed = None
        return sum(len(r.tokens) for r in res) / max(r.total_s - r.ttft_s for r in res)

    plain, spec = tps(0), tps(None)
    assert spec >= 1.2 * plain, (plain, spec)


def test_npu_prefill_then_draft_catches_up(npu, igpu):
    """NPU prefill hands the KV into a unified-KV target context; the draft (holding nothing)
    catches up, then drafts. Qwen3-0.6B as target and draft: nearly every draft is accepted
    (not all: the target's prompt KV is the NPU's bf16 arithmetic, the draft's llama.cpp's;
    measured 30/33). Output vs plain greedy on the iGPU: near-tie rule (this prompt flips a
    0.02-logit tie at token 5 from the NPU KV alone, with no drafting)."""
    from tests.reference.models import QWEN3_06B_HF
    from xdna_relay.backends.igpu.llama import LlamaCppExecutor
    from xdna_relay.sched.speculative import generate
    from xdna_relay.server.engine import Engine
    e = Engine(require(QWEN3_06B_Q8), slots=2, n_ctx=2048, draft_gguf=require(QWEN3_06B_Q8),
               hf_dir=require(QWEN3_06B_HF), npu=True, placement="npu")
    try:
        p = _prompt(e, PROMPTS[1] + " " + "Mention the congestion window. " * 20)
        r = e.complete(p, 64)
    finally:
        e.close()
    assert r.placement == "npu" and r.npu_tokens == len(p) - 1
    assert r.stats.drafted > 0 and r.stats.accepted >= 0.8 * r.stats.drafted
    tgt = LlamaCppExecutor(require(QWEN3_06B_Q8), n_ctx=2048)
    try:
        memo = {}
        _check((tgt, lambda pr, n: memo.setdefault(n, generate(tgt, pr, n).tokens)), p, r.tokens, 64)
    finally:
        tgt.close()
