"""T-035: one llama.cpp context, several sequences, batched decode (llama_multi.py).

Measured 2026-11-25 (Qwen3-8B Q4_K_M): 15.6 / 30.3 / 56.6 tok/s total at 1 / 2 / 4
sequences (separate contexts: 8.0 each at 2); outputs identical to a single-sequence context
except near-ties (one of 4 sequences diverged at token 42 of 48). Here Qwen3-0.6B Q8_0,
whose stories hit near-ties early: every divergence measured sat at a top-2 gap of
0.002–0.07 logits (batched vs single context; batched 2 vs 4; shifted KV cells). Rule (as
T-030): two runs agree, or first differ where the gap is < 0.25 logits.
"""
import pytest

from tests.reference.models import QWEN3_06B_Q8, require

WORDS = ("cat", "lighthouse", "robot", "river")


@pytest.fixture(scope="module")
def ctx(igpu):
    from xdna_relay.backends.igpu.llama_multi import SharedLlamaContext
    c = SharedLlamaContext(require(QWEN3_06B_Q8), n_seq=4, n_ctx=8192)
    yield c
    c.close()


def _prompts(ctx):
    return [ctx.tokenize(f"<|im_start|>user\nWrite a short story about a {w}.<|im_end|>\n"
                         "<|im_start|>assistant\n<think>\n\n</think>\n\n", False) for w in WORDS]


NEAR_TIE = 0.25


def _run(ctx, seqs, ps, steps):
    outs, gaps = [], []
    for s, p in zip(seqs, ps):
        s.reset()
        r = s.eval(p)
        outs.append([int(r.tokens[-1])])
        gaps.append([float(r.gaps[-1])])
    for _ in range(steps):
        r = ctx.decode_step(seqs, [o[-1] for o in outs])
        for o, g, t, gg in zip(outs, gaps, r.tokens, r.gaps):
            o.append(int(t))
            g.append(float(gg))
    return outs, gaps


def _diverge(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


def _agree_or_near_tie(a, ga, b, gb):
    d = _diverge(a, b)
    assert d is None or min(ga[d], gb[d]) < NEAR_TIE, (d, ga[d], gb[d], a, b)


def test_batched_matches_single_sequence(ctx):
    from xdna_relay.backends.igpu.llama import LlamaCppExecutor

    ps = _prompts(ctx)
    ref = LlamaCppExecutor(require(QWEN3_06B_Q8), n_ctx=2048)
    try:
        refs = []
        for p in ps:
            ref.reset()
            r = ref.eval(p)
            out, gap = [int(r.tokens[-1])], [float(r.gaps[-1])]
            for _ in range(31):
                r = ref.eval([out[-1]])
                out.append(int(r.tokens[-1]))
                gap.append(float(r.gaps[-1]))
            refs.append((out, gap))
    finally:
        ref.close()
    outs, gaps = _run(ctx, ctx.seqs, ps, 31)
    for o, g, (r, rg) in zip(outs, gaps, refs):
        _agree_or_near_tie(o, g, r, rg)
    for s, p, o in zip(ctx.seqs, ps, outs):
        assert s.held == p + o[:-1] and s.n_past == len(p) + 31


def test_batch_size_and_cell_placement_only_flip_near_ties(ctx):
    ps = _prompts(ctx)
    a, ga = _run(ctx, ctx.seqs, ps, 24)
    for s in ctx.seqs:  # move where the sequences' cells start, and batch 2 instead of 4
        s.reset()
        s.eval(ps[0][:13])
    b, gb = _run(ctx, ctx.seqs[:2], ps[:2], 24)
    for i in range(2):
        _agree_or_near_tie(a[i], ga[i], b[i], gb[i])


def test_snapshot_restores_into_another_sequence(ctx):
    ps = _prompts(ctx)
    s0, s1 = ctx.seqs[0], ctx.seqs[1]
    s0.reset()
    s0.eval(ps[0])
    snap = s0.snapshot()
    s1.reset()
    s1.restore(snap)
    assert s1.held == ps[0] and s1.n_past == len(ps[0])
    r = ctx.decode_step([s0, s1], [100, 100])
    assert int(r.tokens[0]) == int(r.tokens[1])  # same state, same input -> same output


def test_sequence_context_limit_and_stale_logits(ctx):
    s = ctx.seqs[2]
    s.reset()
    with pytest.raises(ValueError):
        s.eval([1] * (s.n_ctx + 1))
    s.eval(_prompts(ctx)[2])
    s.logits(0)  # valid right after its own eval
    ctx.seqs[3].reset()
    ctx.seqs[3].eval([1, 2, 3])
    with pytest.raises(RuntimeError):
        s.logits(0)  # another decode ran since


def test_multi_stream_state_roundtrip(ctx):
    """A sequence's state from a per-sequence-KV context has one stream per sequence, only its
    own holding cells (llama_state parses and rebuilds it byte for byte)."""
    from xdna_relay.kv import llama_state as ls

    s = ctx.seqs[1]
    s.reset()
    s.eval(_prompts(ctx)[1])
    blob = s.snapshot().data.tobytes()
    st = ls.parse(blob)
    assert st.n_stream == ctx.n_seq and st.n_cells == s.n_past
    assert ls.build(st) == blob
    k, v = ls.to_kv(st)
    assert ls.build(ls.from_kv(st, k, v)) == blob
