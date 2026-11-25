"""T-042: hybrid (recurrent + attention) models: bounded rollback and exact-prefix reuse.

Qwen3.6-35B-A3B (`qwen35moe`): its linear-attention layers keep a recurrent state that
llama.cpp rolls back at most ``n_rs_seq`` tokens, and only within the last batch (enough to
reject drafts after a verify); deeper truncation is refused. So the prefix cache reuses a
hybrid sequence only at exact prefixes: the live sequence when it is a prefix of the prompt,
or a snapshot whose tokens are all a prefix of the prompt. The engine snapshots hybrid
sequences at the end of each prompt (where a next chat turn continues, even when the
template rewrites the previous answer, e.g. dropping its thinking).
Measured (2026-11-25): snapshot 63 MiB at 28 tokens, get 40 ms, set 4 ms.
"""
import numpy as np
import pytest

from xdna_relay.sched.executor import Snapshot
from xdna_relay.sched.prefix_cache import PrefixCache, prepare
from tests.reference.models import QWEN36_35B_A3B_Q4, require


def _snap(tokens, model="m", exact=True):
    return Snapshot(model, tuple(tokens), np.zeros(8, np.uint8), exact=exact)


# ---- prefix cache (pure Python) -------------------------------------------------------

def test_best_exact_only_full_prefixes():
    c = PrefixCache(min_tokens=1)
    c.put(_snap(range(10)))  # prefix of the prompt: usable
    c.put(_snap(list(range(15)) + [99]))  # shares 15 tokens, diverges: not usable exactly
    prompt = list(range(20))
    snap, n = c.best("m", prompt, exact=True)
    assert n == 10 and snap.tokens == tuple(range(10))
    snap, n = c.best("m", prompt)  # dense models: longest common prefix
    assert n == 15
    assert c.peek("m", prompt, exact=True) == 10 and c.peek("m", prompt) == 15


def test_put_keeps_exact_prefix_entries():
    """Dense: a snapshot extended by a newer one is dropped (the newer truncates). Exact
    (recurrent): both stay, the prompt-end snapshot is what a next turn can use."""
    c = PrefixCache(min_tokens=1)
    c.put(_snap(range(5), exact=False))
    c.put(_snap(range(9), exact=False))
    assert len(c) == 1
    c = PrefixCache(min_tokens=1)
    c.put(_snap(range(5)))
    c.put(_snap(range(9)))
    assert len(c) == 2


def test_best_exact_respects_limit():
    c = PrefixCache(min_tokens=1)
    c.put(_snap(range(20)))
    c.put(_snap(range(8)))
    snap, n = c.best("m", list(range(20)), exact=True, limit=19)  # one token must be evaluated
    assert n == 8


class FakeHybrid:
    """Executor stand-in: truncation below what it holds is refused (recurrent state)."""
    name = "m"

    def __init__(self, held):
        self.held, self.n_past, self.rollback = list(held), len(held), 8
        self.restored = None

    def truncate(self, n):
        if n != self.n_past and n != 0:
            raise RuntimeError("cannot roll back")
        self.n_past = n
        del self.held[n:]

    def reset(self):
        self.truncate(0)

    def restore(self, snap):
        self.restored = snap
        self.held, self.n_past = list(snap.tokens), len(snap.tokens)


def test_prepare_hybrid_live_prefix_kept():
    ex = FakeHybrid(range(10))
    n, info = prepare(ex, list(range(30)), PrefixCache(min_tokens=1))
    assert n == 10 and info.reused_live == 10


def test_prepare_hybrid_diverged_live_uses_exact_snapshot():
    ex = FakeHybrid(list(range(12)) + [77, 78])  # diverges at 12: cannot truncate to 12
    c = PrefixCache(min_tokens=1)
    c.put(_snap(range(9)))
    n, info = prepare(ex, list(range(30)), c)
    assert n == 9 and info.reused_cache == 9 and ex.restored.tokens == tuple(range(9))


def test_prepare_hybrid_nothing_reusable_resets():
    ex = FakeHybrid(list(range(5)) + [77])
    n, info = prepare(ex, list(range(30)), PrefixCache(min_tokens=1))
    assert n == 0 and ex.held == []


# ---- hardware: the 35B-A3B ------------------------------------------------------------

@pytest.fixture(scope="module")
def ctx(igpu):
    from xdna_relay.backends.igpu.llama_multi import SharedLlamaContext
    c = SharedLlamaContext(require(QWEN36_35B_A3B_Q4), n_seq=2, n_ctx=2 * 4096, n_rs_seq=8)
    yield c
    c.close()


def _prompt(c, text):
    return c.tokenize(f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", False)


def test_model_is_hybrid_with_rollback(ctx):
    assert ctx.kind == 2 and ctx.rollback == 8 and ctx.seqs[0].rollback == 8


def test_verify_then_rollback_matches_greedy(ctx):
    a, b = ctx.seqs
    p = _prompt(ctx, "Write a haiku about the sea.")
    a.reset(); b.reset()
    ref = [int(a.eval(p).tokens[-1])]
    for _ in range(8):
        ref.append(ctx.step([(a, [ref[-1]], True)])[0][0])
    b.eval(p)
    drafts = ref[1:4] + [11, 12, 13, 14]  # 3 right, then wrong
    got = [x[0] for x in ctx.step([(b, [ref[0]] + drafts, "all")])[0]]
    assert got[:4] == ref[1:5]
    b.truncate(len(p) + 4)  # keep [ref0, d1..d3]; roll back 4 tokens
    assert ctx.step([(b, [got[3]], True)])[0][0] == ref[5]


def test_deep_rollback_refused(ctx):
    a = ctx.seqs[0]
    a.reset()
    a.eval(_prompt(ctx, "Name a colour."))
    with pytest.raises(RuntimeError):
        a.truncate(a.n_past - 10)


def test_snapshot_restore_continues_identically(ctx):
    a, b = ctx.seqs
    a.reset(); b.reset()
    p = _prompt(ctx, "Count from one to five.")
    t = int(a.eval(p).tokens[-1])
    b.restore(a.snapshot())
    assert ctx.step([(a, [t], True)])[0][0] == ctx.step([(b, [t], True)])[0][0]


def test_engine_second_turn_reuses_prompt_snapshot(igpu):
    """Chat turn 2 extends turn 1's prompt (not its answer as generated): the prompt-end
    snapshot is reused exactly; output matches a cold run (near-tie rule: first 8 tokens)."""
    from xdna_relay.server.engine import Engine
    e = Engine(require(QWEN36_35B_A3B_Q4), slots=1, n_ctx=4096)
    try:
        sys1 = "You are terse. " + " ".join(f"Fact {i}: node n{i} links to n{i + 1}." for i in range(40))
        t1 = f"<|im_start|>system\n{sys1}<|im_end|>\n<|im_start|>user\nWhat does n3 link to?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        p1 = e.tokenize(t1)
        e.complete(p1, 12)
        # the history carries a different answer text than generated (as when a template
        # drops thinking): the live sequence diverges right after p1
        p2 = e.tokenize(t1 + "It links to node n4, per fact 3.<|im_end|>\n<|im_start|>user\nAnd n7?<|im_end|>\n"
                        "<|im_start|>assistant\n<think>\n\n</think>\n\n")
        warm = e.complete(p2, 16)
        assert warm.reused >= len(p1), (warm.reused, len(p1))
        e.reset_state()
        cold = e.complete(p2, 16)
        assert cold.reused == 0
        assert warm.tokens[:8] == cold.tokens[:8], (warm.tokens, cold.tokens)
    finally:
        e.close()
