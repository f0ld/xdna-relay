"""T-027: prefix cache (live-sequence reuse + host-RAM snapshots)."""
import numpy as np
import pytest

from xdna_relay.sched.executor import Snapshot
from xdna_relay.sched.prefix_cache import PrefixCache, common_prefix, prefill
from xdna_relay.sched.speculative import generate
from tests.reference.models import QWEN3_06B_Q8, qwen3_chat, require

SYSTEM = ("You are a careful assistant for a network operations team. Answer precisely, cite the "
          "relevant RFC when you can, and keep answers under five sentences. ") * 12


def _chat(system, user):
    return (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


# ---- pure Python -------------------------------------------------------------------------

def _snap(model, tokens, nbytes=100):
    return Snapshot(model, tuple(tokens), np.zeros(nbytes, np.uint8))


def test_common_prefix():
    assert common_prefix([1, 2, 3], [1, 2, 4, 5]) == 2
    assert common_prefix([], [1]) == 0
    assert common_prefix([1, 2], [1, 2]) == 2


def test_best_picks_longest_shared_prefix_per_model():
    c = PrefixCache(min_tokens=1)
    c.put(_snap("a", range(10)))
    c.put(_snap("a", list(range(5)) + [99] * 10))
    c.put(_snap("b", range(20)))
    snap, n = c.best("a", list(range(12)))
    assert n == 10 and snap.tokens == tuple(range(10))
    assert c.best("c", list(range(12))) == (None, 0)


def test_lru_eviction_by_bytes_and_prefix_redundancy():
    c = PrefixCache(max_bytes=250, min_tokens=1)
    c.put(_snap("a", [1, 2]))
    c.put(_snap("a", [1, 2, 3]))  # makes [1, 2] redundant
    assert len(c) == 1 and c.bytes == 100
    c.put(_snap("a", [7, 8]))
    c.best("a", [1, 2, 3])  # touch -> most recent
    c.put(_snap("a", [9, 9]))  # 300 bytes > 250: evict LRU = [7, 8]
    assert {s.tokens for s in c._entries.values()} == {(1, 2, 3), (9, 9)}
    assert c.bytes == 200


def test_min_tokens():
    c = PrefixCache(min_tokens=64)
    c.put(_snap("a", range(10)))
    assert len(c) == 0


# ---- hardware -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ex(igpu):
    from xdna_relay.backends.igpu.llama import LlamaCppExecutor

    e = LlamaCppExecutor(require(QWEN3_06B_Q8), n_ctx=4096)
    yield e
    e.close()


def test_snapshot_restore_roundtrip(ex):
    p = ex.tokenize(_chat(SYSTEM, "What is BGP?"), add_special=False)
    ex.reset()
    ex.eval(p)
    snap = ex.snapshot()
    ex.eval([1, 2, 3])
    ex.eval([4])
    l_after_more = ex.logits(0)
    ex.restore(snap)
    assert ex.n_past == len(p) and ex.held == p
    ex.eval([1, 2, 3])
    ex.eval([4])
    assert np.array_equal(ex.logits(0), l_after_more)


def test_cross_conversation_hit_restores_shared_prefix(ex):
    cache = PrefixCache(min_tokens=64)
    a = ex.tokenize(_chat(SYSTEM, "What is BGP?"), add_special=False)
    b = ex.tokenize(_chat(SYSTEM, "What does OSPF stand for?"), add_special=False)
    ex.reset()
    prefill(ex, a, cache)
    ex.reset()  # another conversation used the executor in between
    rows, info = prefill(ex, b, cache)
    shared = common_prefix(a, b)
    assert shared > 100
    assert info.reused_cache == shared and info.evaluated == len(b) - shared
    assert ex.held == b
    # same first token as a cold prefill (or a near-tie)
    tok_cached = int(rows.tokens[-1])
    ex.reset()
    cold = ex.eval(b)
    assert tok_cached == int(cold.tokens[-1]) or float(cold.gaps[-1]) < 0.25


def test_live_reuse_for_multi_turn(ex):
    p1 = ex.tokenize(_chat(SYSTEM, "What is BGP?"), add_special=False)
    st = generate(ex, p1, 24)
    p2 = p1 + st.tokens + ex.tokenize("<|im_end|>\n<|im_start|>user\nAnd OSPF?<|im_end|>\n"
                                     "<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special=False)
    st2 = generate(ex, p2, 8)
    # the executor still held p1 + generated tokens (minus the last, never evaluated)
    assert st2.prefill.reused_live >= len(p1) + len(st.tokens) - 1


def test_generation_identical_with_and_without_cache(ex):
    cache = PrefixCache(min_tokens=64)
    qs = ["What is BGP?", "What does OSPF stand for?", "Explain MTU in one sentence."]
    cold, warm = [], []
    for q in qs:
        p = ex.tokenize(_chat(SYSTEM, q), add_special=False)
        ex.reset()
        cold.append(generate(ex, p, 32).tokens)
    for q in qs:
        p = ex.tokenize(_chat(SYSTEM, q), add_special=False)
        ex.reset()
        warm.append(generate(ex, p, 32, cache=cache).tokens)
    assert cache.hits >= 2
    # Cached prefill splits the prompt into different batches (and KV-cell placement depends
    # on the executor's history), so float rounding differs: a divergence is allowed only at
    # a near-tie, the same rule as test_t030 (NEAR_TIE 0.25 logits).
    for q, c, w in zip(qs, cold, warm):
        n = common_prefix(c, w)
        if n == min(len(c), len(w)):
            continue
        p = ex.tokenize(_chat(SYSTEM, q), add_special=False)
        ex.reset()
        ex.eval(p + c[:n])
        l = ex.logits(0)
        gap = abs(float(l[c[n]] - l[w[n]]))
        assert gap < 0.25, f"{q!r}: divergence at {n} is not a near-tie (gap {gap:.3f})"
