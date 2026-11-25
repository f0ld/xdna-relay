"""T-031: Qwen3 on IRON (src/xdna_relay/backends/npu/models/qwen3.py) and its KV handoff.

Reference: llama.cpp on the Qwen3-0.6B Q8_0 GGUF prefilling the same prompt. Measured
2026-11-25 (345 tokens, medium precision): K rel.err 0.005–0.11, V 0.016–0.27 (the middle
layers' outliers; same at high precision, so the Q8 reference's), first-token KL 4.5e-3
(high: 1.7e-3), 64/64 greedy tokens identical.

llama.cpp caches Qwen's K in Hugging Face's two-halves RoPE layout (NEOX), unlike Llama's:
the handoff must not permute it (docs/knowledge/pitfalls.md).
"""
import numpy as np
import pytest

from xdna_relay.kv import llama_state as ls
from tests.reference.models import QWEN3_06B_HF, QWEN3_06B_Q8, require

TEXT = ("The lighthouse keeper kept a careful log of every ship that passed the rocky point. "
        "Some nights the fog was so thick that he listened for bells and engines instead. ") * 8


def _kl(a, b):
    pa = np.exp(a - a.max()); pa /= pa.sum()
    pb = np.exp(b - b.max()); pb /= pb.sum()
    return float(np.sum(pa * (np.log(pa + 1e-30) - np.log(pb + 1e-30))))


@pytest.fixture(scope="module")
def gpu(igpu):
    from xdna_relay.backends.igpu.llama import LlamaCppExecutor

    e = LlamaCppExecutor(require(QWEN3_06B_Q8), n_ctx=2048)
    yield e
    e.close()


@pytest.fixture(scope="module")
def npu_ex(npu):
    from xdna_relay.backends.npu.iron_lm import IronLmExecutor

    return IronLmExecutor(require(QWEN3_06B_HF), max_seq_len=1024)


def _greedy(ex, last, n):
    out = [int(ex.eval([last]).tokens[-1])]
    for _ in range(n - 1):
        out.append(int(ex.eval([out[-1]]).tokens[-1]))
    return out


def test_model_selected_and_eog(npu_ex):
    from xdna_relay.backends.npu.models.qwen3 import Qwen3

    assert isinstance(npu_ex.model, Qwen3)
    assert not npu_ex.llama_cpp_k_interleaved
    im_end = npu_ex.tok.encoding.encode("<|im_end|>")
    assert len(im_end) == 1 and npu_ex.is_eog(im_end[0])
    assert not npu_ex.is_eog(npu_ex.tok.encoding.encode("<")[0])


def test_qwen3_handoff_matches_llama_prefill(gpu, npu_ex):
    from xdna_relay.kv.handoff import npu_to_llama

    prompt = gpu.tokenize(TEXT + "\nWhat did the keeper listen for?\n")
    assert npu_ex.tokenize(TEXT)[:8] == gpu.tokenize(TEXT)[:8]
    n = len(prompt) - 1
    gpu.reset()
    gpu.eval(prompt[:-1])
    rk, rv = ls.to_kv(ls.parse(gpu.snapshot().data.tobytes()))
    gpu.eval([prompt[-1]])
    ref_logits = gpu.logits(0)
    gpu.truncate(n)
    ref_tokens = _greedy(gpu, prompt[-1], 32)

    npu_ex.reset()
    npu_ex.eval(prompt[:-1])
    ks, vs = npu_ex.export_kv(n)
    for i in range(len(ks)):
        rel_k = np.linalg.norm(ks[i].reshape(n, -1) - rk[i]) / np.linalg.norm(rk[i])
        rel_v = np.linalg.norm(vs[i].reshape(n, -1) - rv[i]) / np.linalg.norm(rv[i])
        assert rel_k < 0.2 and rel_v < 0.4, (i, rel_k, rel_v)
    # the interleaved (Llama) layout is wrong for Qwen
    k0 = ls.halves_to_interleaved(ks[0]).reshape(n, -1)
    assert np.linalg.norm(k0 - rk[0]) / np.linalg.norm(rk[0]) > 0.5

    gpu.reset()
    npu_to_llama(npu_ex, gpu, n)
    gpu.eval([prompt[-1]])
    assert _kl(ref_logits, gpu.logits(0)) < 2e-2
    gpu.truncate(n)
    ho_tokens = _greedy(gpu, prompt[-1], 32)
    p = next((i for i, (a, b) in enumerate(zip(ref_tokens, ho_tokens)) if a != b), None)
    assert p is None or p >= 8, (p, ref_tokens, ho_tokens)  # measured: identical (64/64)


def test_worker_handoff_qwen3(gpu, npu):
    """The worker-process path packs Qwen's K without permuting it."""
    from xdna_relay.backends.npu.worker import NpuWorker
    from xdna_relay.kv.handoff import worker_to_llama

    prompt = gpu.tokenize(TEXT + "\nWhat did the keeper listen for?\n")
    n = len(prompt) - 1
    gpu.reset()
    gpu.eval(prompt[:-1])
    gpu.eval([prompt[-1]])
    ref_logits = gpu.logits(0)
    w = NpuWorker(require(QWEN3_06B_HF), max_seq_len=1024)
    try:
        gpu.reset()
        info = worker_to_llama(w, gpu, prompt[:-1])
    finally:
        w.close()
    assert gpu.n_past == n and info.tokens == n
    gpu.eval([prompt[-1]])
    assert _kl(ref_logits, gpu.logits(0)) < 2e-2
