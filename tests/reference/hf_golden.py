"""Hugging Face reference logits (T-022). Runs in the reference venv (torch + transformers,
~/.cache/xdna-relay/ref-venv), not the project venv.

Input npz: ``ids`` (prompt token ids), ``steps`` (greedy steps). Output npz: ``logits``
[steps + 1, vocab] float32 — after the prompt, then after each greedy token — and ``tokens``
(the greedy tokens fed, teacher-forcing every other implementation).

Usage: ref-venv/bin/python tests/reference/hf_golden.py HF_DIR IN.npz OUT.npz [--dtype float32]
"""
import argparse

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf"); ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    a = ap.parse_args()
    d = np.load(a.inp)
    ids, steps = d["ids"].astype(np.int64), int(d["steps"])
    torch.set_num_threads(max(1, torch.get_num_threads()))
    model = AutoModelForCausalLM.from_pretrained(a.hf, dtype=getattr(torch, a.dtype))
    model.eval()
    out, toks = [], []
    with torch.no_grad():
        r = model(torch.tensor(ids[None]), use_cache=True)
        past = r.past_key_values
        logits = r.logits[0, -1].float().numpy()
        for _ in range(steps):
            out.append(logits)
            t = int(np.argmax(logits))
            toks.append(t)
            r = model(torch.tensor([[t]]), past_key_values=past, use_cache=True)
            past = r.past_key_values
            logits = r.logits[0, -1].float().numpy()
        out.append(logits)
    np.savez(a.out, logits=np.stack(out).astype(np.float32), tokens=np.array(toks, np.int32))


if __name__ == "__main__":
    main()
