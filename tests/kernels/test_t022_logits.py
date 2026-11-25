"""T-022: IRON NPU logits vs the Hugging Face float32 reference (Llama 3.2 1B, high).

Measured 2026-11-25 (bench/results/2026-11-25-logits-check.md): 286-token prompt + 16
teacher-forced steps, KL mean 9.97e-4, max 8.99e-3, top-1 17/17. Needs the reference venv
(torch + transformers): ~/.cache/xdna-relay/ref-venv (fails, never skips, without it).
"""
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tests.reference.models import LLAMA32_1B_HF, require

REF_PY = Path.home() / ".cache/xdna-relay/ref-venv/bin/python"
ROOT = Path(__file__).resolve().parents[2]
TEXT = ("The lighthouse keeper kept a careful log of every ship that passed the rocky point. "
        "Some nights the fog was so thick that he could not see the water below the tower. ") * 4


def _kl(a, b):
    pa = np.exp(a - a.max()); pa /= pa.sum()
    pb = np.exp(b - b.max()); pb /= pb.sum()
    return float(np.sum(pa * (np.log(pa + 1e-30) - np.log(pb + 1e-30))))


def test_iron_logits_match_hf_reference(npu, tmp_path):
    if not REF_PY.exists():
        pytest.fail(f"reference venv missing: {REF_PY} (see bench/results/2026-11-25-logits-check.md)")
    from xdna_relay.backends.npu.iron_lm import IronLmExecutor

    hf = require(LLAMA32_1B_HF)
    ex = IronLmExecutor(hf, max_seq_len=1024, precision="high")
    ids = ex.tokenize(TEXT + "\nWhat did the keeper do?\n")
    np.savez(tmp_path / "in.npz", ids=np.array(ids, np.int32), steps=8)
    subprocess.run([str(REF_PY), str(ROOT / "tests/reference/hf_golden.py"), hf,
                    str(tmp_path / "in.npz"), str(tmp_path / "out.npz")], check=True,
                   env={**os.environ, "PYTHONPATH": ""})
    g = np.load(tmp_path / "out.npz")
    ref, toks = g["logits"], [int(t) for t in g["tokens"]]
    ex.eval(ids)
    got = [ex.logits(0).copy()]
    for t in toks:
        ex.eval([t])
        got.append(ex.logits(0).copy())
    kls = [_kl(r, x) for r, x in zip(ref, got)]
    top1 = sum(int(np.argmax(r) == np.argmax(x)) for r, x in zip(ref, got))
    assert np.mean(kls) < 5e-3 and max(kls) < 3e-2, kls  # measured 1.0e-3 / 9.0e-3
    assert top1 >= len(got) - 1, top1  # measured 17/17 (one near-tie allowed)
