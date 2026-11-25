"""T-022: IRON (NPU) logits vs the Hugging Face reference, and llama.cpp (iGPU) as a second
implementation, teacher-forced on the reference's greedy tokens.

Per step (after the prompt, then after each token): KL(ref || x), top-1 agreement, max |diff|.

Usage: python bench/logits_check.py --hf HF_DIR [--gguf BF16.gguf] [--words 250] [--steps 16]
       [--precision medium] [--packed] [--ref-dtype float32]
"""
import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

REF_PY = Path.home() / ".cache/xdna-relay/ref-venv/bin/python"
ROOT = Path(__file__).resolve().parents[1]
TEXT = ("The lighthouse keeper kept a careful log of every ship that passed the rocky point. "
        "Some nights the fog was so thick that he could not see the water below the tower, and "
        "he listened instead for bells and engines. ")


def kl(p_logits, q_logits):
    p = np.exp(p_logits - p_logits.max()); p /= p.sum()
    q = np.exp(q_logits - q_logits.max()); q /= q.sum()
    return float(np.sum(p * (np.log(p + 1e-30) - np.log(q + 1e-30))))


def compare(name, ref, got):
    kls = [kl(r, g) for r, g in zip(ref, got)]
    top1 = [int(np.argmax(r) == np.argmax(g)) for r, g in zip(ref, got)]
    mad = [float(np.max(np.abs(r - g))) for r, g in zip(ref, got)]
    print(f"{name:28s} KL mean {np.mean(kls):.2e} max {np.max(kls):.2e} | top-1 {sum(top1)}/{len(top1)} "
          f"| max|diff| median {np.median(mad):.3f} max {np.max(mad):.3f}", flush=True)
    return dict(kl_mean=float(np.mean(kls)), kl_max=float(np.max(kls)), top1=sum(top1), n=len(top1))


def main():
    from tdr_guard import guard
    guard()
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--gguf")
    ap.add_argument("--words", type=int, default=250)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--precision")
    ap.add_argument("--packed", action="store_true")
    ap.add_argument("--ref-dtype", default="float32")
    a = ap.parse_args()
    from xdna_relay.backends.npu.iron_lm import IronLmExecutor

    npu = IronLmExecutor(a.hf, max_seq_len=2048, precision=a.precision, packed=a.packed)
    words = (TEXT * 60).split()[: a.words]
    ids = npu.tokenize(" ".join(words) + "\nWhat did the keeper do on foggy nights?\n")
    with tempfile.TemporaryDirectory() as d:
        np.savez(f"{d}/in.npz", ids=np.array(ids, np.int32), steps=a.steps)
        subprocess.run([str(REF_PY), str(ROOT / "tests/reference/hf_golden.py"), a.hf, f"{d}/in.npz",
                        f"{d}/out.npz", "--dtype", a.ref_dtype], check=True,
                       env={**os.environ, "PYTHONPATH": ""})
        g = np.load(f"{d}/out.npz")
        ref, toks = g["logits"], [int(t) for t in g["tokens"]]
    print(f"{npu.name}: prompt {len(ids)} tokens, {a.steps} teacher-forced steps; reference HF {a.ref_dtype}")

    got = [np.asarray(npu.eval(ids) and npu.logits(0), np.float32)]
    for t in toks:
        npu.eval([t]); got.append(np.asarray(npu.logits(0), np.float32))
    compare("IRON NPU", ref, got)

    if a.gguf:
        from xdna_relay.backends.igpu.llama import LlamaCppExecutor
        gpu = LlamaCppExecutor(a.gguf, n_ctx=4096)
        assert gpu.tokenize(" ".join(words))[:8] == npu.tokenize(" ".join(words))[:8], "tokenizers differ"
        gpu.eval(ids); got = [np.asarray(gpu.logits(0), np.float32).copy()]
        for t in toks:
            gpu.eval([t]); got.append(np.asarray(gpu.logits(0), np.float32).copy())
        compare(f"llama.cpp {Path(a.gguf).stem}", ref, got)


if __name__ == "__main__":
    main()
