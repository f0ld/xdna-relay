"""T-024: LlamaCppExecutor (llama.cpp Vulkan via native/llama_shim) honours the Executor contract."""
import numpy as np
import pytest

from tests.reference.models import QWEN3_06B_Q8, qwen3_chat, require

TEXT = "Explain in two sentences why the sky is blue."


def _kl(a, b):
    pa = np.exp(a - a.max()); pa /= pa.sum()
    pb = np.exp(b - b.max()); pb /= pb.sum()
    return float(np.sum(pa * (np.log(pa + 1e-30) - np.log(pb + 1e-30))))


@pytest.fixture(scope="module")
def ex(igpu):
    from xdna_relay.backends.igpu.llama import LlamaCppExecutor

    e = LlamaCppExecutor(require(QWEN3_06B_Q8), n_ctx=2048, n_batch=256)
    yield e
    e.close()


def test_tokenize_roundtrip(ex):
    toks = ex.tokenize("Hello, NPU and iGPU!", add_special=False)
    assert len(toks) > 3
    assert ex.detokenize(toks) == "Hello, NPU and iGPU!"


def test_eval_advances_and_reads_last_row(ex):
    ex.reset()
    p = ex.tokenize(qwen3_chat(TEXT), add_special=False)
    r = ex.eval(p)
    assert ex.n_past == len(p)
    assert r.tokens.shape == (1,) and 0 <= r.tokens[0] < ex.n_vocab
    l = ex.logits(0)
    assert l.shape == (ex.n_vocab,) and int(np.argmax(l)) == int(r.tokens[0])


def test_chunked_eval_matches_single_call(ex):
    p = ex.tokenize(qwen3_chat(TEXT), add_special=False)
    ex.reset()
    one = ex.eval(p)
    l_one = ex.logits(0)
    ex.reset()
    ex.eval(p[:10])
    two = ex.eval(p[10:])
    l_two = ex.logits(0)
    assert int(one.tokens[-1]) == int(two.tokens[-1])
    # Different batch splits round differently: single logits move by up to ~0.65, the
    # distribution barely (KL measured <= 4.7e-3 on Qwen3-0.6B/8B at several splits).
    assert _kl(l_one, l_two) < 1e-2


def test_truncate_then_reeval_reproduces(ex):
    p = ex.tokenize(qwen3_chat(TEXT), add_special=False)
    ex.reset()
    ex.eval(p)
    for t in [11, 22, 33]:
        ex.eval([t])
    ex.truncate(len(p) + 1)
    assert ex.n_past == len(p) + 1
    ex.eval([22])
    l_a = ex.logits(0)
    ex.truncate(len(p) + 1)
    ex.eval([22])
    l_b = ex.logits(0)
    assert np.array_equal(l_a, l_b)  # same shapes, same state: bitwise


def test_all_logits_rows_match_greedy_steps(ex):
    """Verify path: one batch over [t0..t3] gives a row per token; argmax per row equals the
    single-step greedy readout, except at near-ties (gap < 0.25, see test_t030)."""
    p = ex.tokenize(qwen3_chat(TEXT), add_special=False)
    ex.reset()
    seq = [int(ex.eval(p).tokens[-1])]
    steps = []
    for _ in range(4):
        r = ex.eval([seq[-1]])
        steps.append((int(r.tokens[-1]), float(r.gaps[-1])))
        seq.append(steps[-1][0])
    ex.truncate(len(p))
    rows = ex.eval(seq[:4], all_logits=True)
    assert rows.tokens.shape == (4,)
    for i, (tok, gap) in enumerate(steps):
        assert int(rows.tokens[i]) == tok or gap < 0.25, (i, tok, gap)


def test_errors(ex):
    ex.reset()
    with pytest.raises(ValueError):
        ex.eval([])
    with pytest.raises(ValueError):
        ex.truncate(5)
