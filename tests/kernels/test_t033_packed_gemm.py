"""T-033: prompt-chunk projections on pre-packed bfp16 weights (models/packed.py).

Reference: llama.cpp on the Qwen3-0.6B Q8_0 GGUF prefilling the same prompt, as T-031.
Measured 2026-11-25 (345 tokens): first-token KL 3.75e-3 (unpacked 4.5e-3), 64/64 greedy
tokens identical; Qwen3-8B into Q4_K_M: KL 2.79e-2, 50/64 (unpacked the same).
"""
import numpy as np
import pytest

from tests.reference.models import QWEN3_06B_HF, QWEN3_06B_Q8, require

TEXT = ("The lighthouse keeper kept a careful log of every ship that passed the rocky point. "
        "Some nights the fog was so thick that he listened for bells and engines instead. ") * 8


def _kl(a, b):
    pa = np.exp(a - a.max()); pa /= pa.sum()
    pb = np.exp(b - b.max()); pb /= pb.sum()
    return float(np.sum(pa * (np.log(pa + 1e-30) - np.log(pb + 1e-30))))


def test_packed_bd_fits_patch():
    """The block-float DMA check no longer raises (mlir-aie's np.dtype() on the marker)."""
    from aie.helpers.util import v8bfp16ebs8
    from aie.helpers.taplib import TensorAccessPattern
    from aie.iron.device import NPU2

    import xdna_relay.backends.npu.models.packed  # noqa: F401  (applies the patch)
    tap = TensorAccessPattern.full((4096,))
    assert NPU2().dma_bd_fits(0, 0, tap, v8bfp16ebs8) in (True, False)


def test_packed_handoff_matches_llama_prefill(npu, igpu):
    from xdna_relay.backends.igpu.llama import LlamaCppExecutor
    from xdna_relay.backends.npu.iron_lm import IronLmExecutor
    from xdna_relay.kv.handoff import npu_to_llama

    gpu = LlamaCppExecutor(require(QWEN3_06B_Q8), n_ctx=2048)
    try:
        ex = IronLmExecutor(require(QWEN3_06B_HF), max_seq_len=1024, packed=True)
        assert ex.packed_bytes > 0
        prompt = gpu.tokenize(TEXT + "\nWhat did the keeper listen for?\n")
        n = len(prompt) - 1
        gpu.reset()
        gpu.eval(prompt[:-1])
        gpu.eval([prompt[-1]])
        ref = gpu.logits(0)
        ref_tokens = [int(np.argmax(ref))]
        for _ in range(15):
            ref_tokens.append(int(gpu.eval([ref_tokens[-1]]).tokens[-1]))
        ex.eval(prompt[:-1])
        gpu.reset()
        npu_to_llama(ex, gpu, n)
        gpu.eval([prompt[-1]])
        assert _kl(ref, gpu.logits(0)) < 2e-2
        out = [int(np.argmax(gpu.logits(0)))]
        for _ in range(15):
            out.append(int(gpu.eval([out[-1]]).tokens[-1]))
        p = next((i for i, (a, b) in enumerate(zip(ref_tokens, out)) if a != b), None)
        assert p is None or p >= 8, (p, ref_tokens, out)
    finally:
        gpu.close()
