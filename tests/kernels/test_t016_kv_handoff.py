"""T-016: KV handoff NPU (IRON) prefill -> llama.cpp (iGPU) decode.

Reference: llama.cpp prefilling the same prompt itself, same weights (BF16 GGUF of the HF
checkpoint IRON loads). Measured 2026-11-24 (Llama 3.2 1B, 346 tokens): K rel.err
0.004–0.014, V 0.011–0.036, first-token KL 4.9e-4, 64/64 greedy tokens identical.
"""
import numpy as np
import pytest

from xdna_relay.kv import llama_state as ls
from tests.reference.models import LLAMA32_1B_BF16, LLAMA32_1B_HF, require

TEXT = ("The lighthouse keeper kept a careful log of every ship that passed the rocky point. "
        "Some nights the fog was so thick that he listened for bells and engines instead. ") * 8


def _kl(a, b):
    pa = np.exp(a - a.max()); pa /= pa.sum()
    pb = np.exp(b - b.max()); pb /= pb.sum()
    return float(np.sum(pa * (np.log(pa + 1e-30) - np.log(pb + 1e-30))))


def test_halves_to_interleaved():
    x = np.arange(2 * 8, dtype=np.float32).reshape(2, 8)  # D = 8: halves [0..3] [4..7]
    y = ls.halves_to_interleaved(x)
    assert y[0].tolist() == [0, 4, 1, 5, 2, 6, 3, 7]
    # a rotation by RoPE in the halves layout == the same rotation in the interleaved one
    D, th = 8, np.linspace(0.1, 1.0, 4)
    c, s = np.cos(th), np.sin(th)
    v = np.random.default_rng(0).standard_normal(D).astype(np.float32)
    h1, h2 = v[:4], v[4:]
    rot_halves = np.concatenate([h1 * c - h2 * s, h2 * c + h1 * s])
    iv = ls.halves_to_interleaved(v)
    rot_inter = np.empty(D, np.float32)
    rot_inter[0::2] = iv[0::2] * c - iv[1::2] * s
    rot_inter[1::2] = iv[1::2] * c + iv[0::2] * s
    assert np.allclose(ls.halves_to_interleaved(rot_halves), rot_inter, atol=1e-6)


@pytest.fixture(scope="module")
def gpu(igpu):
    from xdna_relay.backends.igpu.llama import LlamaCppExecutor

    e = LlamaCppExecutor(require(LLAMA32_1B_BF16), n_ctx=2048)
    yield e
    e.close()


@pytest.fixture(scope="module")
def npu_ex(npu):
    from xdna_relay.backends.npu.iron_lm import IronLmExecutor

    return IronLmExecutor(require(LLAMA32_1B_HF), max_seq_len=1024)


def test_state_parse_build_roundtrip(gpu):
    gpu.reset()
    gpu.eval(gpu.tokenize(TEXT))
    blob = gpu.snapshot().data.tobytes()
    st = ls.parse(blob)
    assert st.n_cells == gpu.n_past and len(st.k) == len(st.v) == 16
    assert ls.build(st) == blob
    k, v = ls.to_kv(st)
    assert ls.build(ls.from_kv(st, k, v)) == blob  # f16 -> f32 -> f16 is exact


def test_handoff_matches_llama_prefill(gpu, npu_ex):
    from xdna_relay.kv.handoff import npu_to_llama

    prompt = gpu.tokenize(TEXT + "\nWhat did the keeper listen for?\n")
    n = len(prompt) - 1
    gpu.reset()
    gpu.eval(prompt[:-1])
    rk, rv = ls.to_kv(ls.parse(gpu.snapshot().data.tobytes()))
    ref = gpu.eval([prompt[-1]])
    ref_logits = gpu.logits(0)
    ref_tokens = [int(ref.tokens[-1])]
    for _ in range(31):
        ref_tokens.append(int(gpu.eval([ref_tokens[-1]]).tokens[-1]))

    npu_ex.reset()
    npu_ex.eval(prompt[:-1])
    ks, vs = npu_ex.export_kv(n)
    for i in range(len(ks)):
        kp = ls.halves_to_interleaved(ks[i]).reshape(n, -1)
        rel_k = np.linalg.norm(kp - rk[i]) / np.linalg.norm(rk[i])
        rel_v = np.linalg.norm(vs[i].reshape(n, -1) - rv[i]) / np.linalg.norm(rv[i])
        assert rel_k < 0.05 and rel_v < 0.08, (i, rel_k, rel_v)
    # the layout matters: without the permutation K is far off
    assert np.linalg.norm(ks[0].reshape(n, -1) - rk[0]) / np.linalg.norm(rk[0]) > 0.5

    gpu.reset()
    info = npu_to_llama(npu_ex, gpu, n)
    assert gpu.n_past == n and gpu.held == prompt[:-1] and info.tokens == n
    ho = gpu.eval([prompt[-1]])
    assert _kl(ref_logits, gpu.logits(0)) < 1e-2
    ho_tokens = [int(ho.tokens[-1])]
    for _ in range(31):
        ho_tokens.append(int(gpu.eval([ho_tokens[-1]]).tokens[-1]))
    p = next((i for i, (a, b) in enumerate(zip(ref_tokens, ho_tokens)) if a != b), None)
    assert p is None or p >= 8, (p, ref_tokens, ho_tokens)  # measured: identical (64/64)
