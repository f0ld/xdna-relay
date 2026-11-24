"""T-022: IRON Llama on the NPU, driven step by step from the host (Session.logits).

The device-side sampling loop (Session.draw with a full-ELF decode step) needs an amdxdna
that can pin user-pointer BOs (>= 2.27, xdna-driver 72e64a6); the in-kernel driver here
fails with EFAULT. Host-driven steps are what our executor needs anyway.

Usage: python bench/iron_llama_probe.py <hf_dir> [--prompt-words 400] [--gen 64] [--max-seq-len 2048]
Prints greedy text, prefill time, per-step decode times, and a rollback timing.
"""
import argparse
import time

import numpy as np

from iron.applications.llama.model import Llama


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--prompt-words", type=int, default=400)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--full-elf", action=argparse.BooleanOptionalAction, default=None)
    a = ap.parse_args()

    t0 = time.perf_counter()
    model = Llama.from_checkpoint(a.path, max_seq_len=a.max_seq_len)
    model.load_versions(release=model.checkpoint.release, full_elf=a.full_elf)
    print(f"[load] {time.perf_counter() - t0:.1f} s (compile or cache + weights)")

    text = a.prompt or ("The history of the lighthouse begins with the ancient world. " * 200)
    words = text.split()[: a.prompt_words] if not a.prompt else text.split()
    tokens = list(model.tokenizer(" ".join(words)))
    s = model.session

    t = time.perf_counter()
    logits = s.logits(tokens)
    t_pre = time.perf_counter() - t
    print(f"[prefill] {len(tokens)} tokens in {t_pre:.3f} s = {len(tokens) / t_pre:.1f} tok/s")

    steps = []
    out = []
    for _ in range(a.gen):
        nxt = int(np.argmax(logits))
        out.append(nxt)
        tokens.append(nxt)
        t = time.perf_counter()
        logits = s.logits(tokens)
        steps.append(time.perf_counter() - t)
    st = np.array(steps[2:]) if len(steps) > 4 else np.array(steps)
    print(f"[decode] {a.gen} steps: median {1e3 * np.median(st):.1f} ms, p90 {1e3 * np.percentile(st, 90):.1f} ms"
          f" -> {1 / np.median(st):.2f} tok/s")
    print("[text]", repr(model.tokenizer.decode(out)[:300]))

    # Rollback cost: forget the last 4 tokens, then re-feed them one by one vs as a chunk.
    n = len(tokens)
    s.forget(n - 4)
    t = time.perf_counter()
    s.logits(tokens[: n - 3])  # resumes from the held prefix: 1 decode-step-sized call?
    t_r1 = time.perf_counter() - t
    t = time.perf_counter()
    s.logits(tokens)  # 3 more tokens: not a single decode step -> prompt chunk path
    t_r3 = time.perf_counter() - t
    print(f"[rollback] forget 4 then 1 token: {1e3 * t_r1:.1f} ms; then 3 tokens at once: {1e3 * t_r3:.1f} ms")


if __name__ == "__main__":
    main()
