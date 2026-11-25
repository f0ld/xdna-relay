"""T-043: speculative decoding with the target's own MTP head (Qwen3.6-35B-A3B, batched).

The shim keeps a second, MTP-type context of the same model current after every target
batch (token x_p with the target's hidden state at p-1) and drafts greedily from it; a
rejected draft truncates the target (recurrent rollback, T-042) and the MTP KV, and the
next draft continues from the hidden state of the last accepted token.
Correctness rule (MoE): every emitted token is the target's greedy choice or within MAX_GAP
logits of it, teacher-forced over the whole output, with at most MAX_MISS such misses. The
T-030 rule (equal to plain greedy up to a < 0.25 near-tie) does not hold for MoE models:
plain decoding alone vs batched with another request (no drafting) already diverges with a
0.35 gap (a batch-shape-dependent expert-routing flip), and the single-eval reference adds
its own (up to 0.74 measured). A bug (stale state after a rollback, a misaligned MTP state)
emits tokens several logits off, at many positions.
Measured (2026-11-25, direct loop k=3): code 20.5 -> 31.4 tok/s (94/108 accepted), story
20.3 -> 19.4 (70/171): matches llama-server's draft-mtp.
"""
import threading

import pytest

from tests.reference.models import QWEN36_35B_A3B_MTP_Q4, require

MAX_GAP = 1.0
MAX_MISS = 4
CODE = "Write a Python function that reads a CSV file and returns per-column mean, min and max."
STORY = "Write a short story about a lighthouse keeper who finds a message in a bottle."


def _chat(text):
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


@pytest.fixture(scope="module")
def engine(igpu):
    from xdna_relay.server.engine import Engine
    e = Engine(require(QWEN36_35B_A3B_MTP_Q4), slots=2, n_ctx=2048, mtp=True)
    yield e
    e.close()


def _run(engine, prompts, n, k):
    engine.spec.fixed = k
    out = [None] * len(prompts)

    def one(i):
        out[i] = engine.complete(prompts[i], n)
    try:
        engine.reset_state()
        ts = [threading.Thread(target=one, args=(i,)) for i in range(len(prompts))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    finally:
        engine.spec.fixed = None
    return out


def _check(engine, prompt, got):
    """Teacher-forced: the target's logits at every position of prompt + got."""
    ex = engine.slots[1].target
    ex.reset()
    ex.eval(prompt)
    rows = [ex.logits(0)]
    if len(got) > 1:
        ex.eval(got[:-1], all_logits=True)
        rows += [ex.logits(i) for i in range(len(got) - 1)]
    ex.reset()
    gaps = [float(l.max() - l[t]) for l, t in zip(rows, got)]
    miss = [(i, round(g, 3)) for i, g in enumerate(gaps) if g > 0]
    assert max(gaps) < MAX_GAP and len(miss) <= MAX_MISS, miss


def test_engine_uses_mtp(engine):
    assert engine.mtp and engine.spec is not None and engine.dshared is None
    assert engine.shared.rollback == 8 and engine.spec.k_max <= 8
    assert engine.spec.draft and engine.spec.verify  # swept at start


@pytest.mark.parametrize("k", [None, 3])
def test_concurrent_mtp_matches_plain(engine, k):
    ps = [engine.tokenize(_chat(CODE)), engine.tokenize(_chat(STORY))]
    plain = _run(engine, ps, 96, 0)
    spec = _run(engine, ps, 96, k)
    for p, a, b in zip(ps, plain, spec):
        assert b.stats.drafted > 0 and 0 <= b.stats.accepted <= b.stats.drafted
        if k is not None:
            assert all(x <= k for x in b.stats.ks)
        _check(engine, p, a.tokens)
        _check(engine, p, b.tokens)
    assert spec[0].stats.accepted >= 0.6 * spec[0].stats.drafted  # code: MTP predicts well


@pytest.mark.parametrize("n", [1, 2, 5])
def test_budget_never_exceeded(engine, n):
    r = _run(engine, [engine.tokenize(_chat(CODE))], n, 6)[0]
    assert len(r.tokens) == n


def test_eog_inside_accepted_drafts(engine):
    r = _run(engine, [engine.tokenize(_chat("Reply with exactly the word: yes"))], 64, 6)[0]
    eog = [i for i, t in enumerate(r.tokens) if engine.is_eog(t)]
    assert eog and eog[0] == len(r.tokens) - 1, r.tokens


def test_second_turn_after_restore(engine):
    """A restored prompt snapshot (T-042) carries no MTP state: drafting continues (worse
    acceptance until the MTP context sees new tokens), output stays exact."""
    sys1 = " ".join(f"Fact {i}: node n{i} links to n{i + 1}." for i in range(30))
    t1 = f"<|im_start|>system\n{sys1}<|im_end|>\n" + _chat("What does n3 link to?")
    engine.reset_state()
    engine.complete(engine.tokenize(t1), 8)
    p2 = engine.tokenize(t1 + "It links to n4.<|im_end|>\n" + _chat("And n7? Explain the chain in detail."))
    warm = engine.complete(p2, 48)
    assert warm.reused > 0 and warm.stats.drafted > 0
    _check(engine, p2, warm.tokens)


def test_mtp_is_faster_on_code(engine):
    p = [engine.tokenize(_chat(CODE))]
    plain = _run(engine, p, 128, 0)[0]
    spec = _run(engine, p, 128, None)[0]
    tps = lambda r: (len(r.tokens) - 1) / (r.total_s - r.ttft_s)
    assert tps(spec) >= 1.3 * tps(plain), (tps(plain), tps(spec))
