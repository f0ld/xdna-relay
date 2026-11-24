"""T-016: NPU (IRON) prefill -> llama.cpp (iGPU) decode, vs llama.cpp prefilling itself.

Usage: python bench/handoff_probe.py --hf HF_DIR --gguf MODEL.gguf [--words 300] [--gen 64]
"""
import argparse
import time

import numpy as np

from xdna_relay.backends.igpu.llama import LlamaCppExecutor
from xdna_relay.backends.npu.iron_lm import IronLmExecutor
from xdna_relay.kv import llama_state as ls
from xdna_relay.kv.handoff import npu_to_llama
from xdna_relay.sched.speculative import generate

TEXT = ("The lighthouse keeper kept a careful log of every ship that passed the rocky point. "
        "Some nights the fog was so thick that he could not see the water below the tower, and "
        "he listened instead for bells and engines. ")


def kl(a, b):
    pa = np.exp(a - a.max()); pa /= pa.sum()
    pb = np.exp(b - b.max()); pb /= pb.sum()
    return float(np.sum(pa * (np.log(pa + 1e-30) - np.log(pb + 1e-30))))


def greedy_from_held(ex, last, n):
    out = []
    r = ex.eval([last])
    for _ in range(n):
        t = int(r.tokens[-1])
        out.append(t)
        r = ex.eval([t])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--words", type=int, default=300)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--precision", help="IRON matmul precision override (high, medium)")
    ap.add_argument("--profile-dir", help="IRON tuned profile directory (bench/iron_tune.py)")
    a = ap.parse_args()
    gpu = LlamaCppExecutor(a.gguf, n_ctx=4096)
    npu = IronLmExecutor(a.hf, max_seq_len=2048, precision=a.precision, profile_dir=a.profile_dir)
    words = (TEXT * 50).split()[: a.words]
    prompt = gpu.tokenize(" ".join(words) + "\nIn one sentence, what did the keeper do on foggy nights?\n")
    assert npu.tokenize(" ".join(words)) [:5] == prompt[:5], "tokenizers disagree"
    n = len(prompt) - 1
    print(f"prompt {len(prompt)} tokens; model {gpu.name}")

    # reference: llama.cpp prefills itself
    gpu.reset()
    t = time.perf_counter(); gpu.eval(prompt[:-1]); t_gpu_prefill = time.perf_counter() - t
    ref_state = ls.parse(gpu.snapshot().data.tobytes())
    r = gpu.eval([prompt[-1]]); ref_logits = gpu.logits(0)
    gpu.truncate(n)
    ref_tokens = greedy_from_held(gpu, prompt[-1], a.gen)

    # NPU prefill + handoff
    npu.reset()
    t = time.perf_counter(); npu.eval(prompt[:-1]); t_npu_prefill = time.perf_counter() - t
    t = time.perf_counter(); npu.eval(prompt[:-1]); t_npu_warm = time.perf_counter() - t  # held: no-op
    ks, vs = npu.export_kv(n)
    rk, rv = ls.to_kv(ref_state)
    def rel(x, y):
        return float(np.linalg.norm(x - y) / np.linalg.norm(y))
    for i in (0, len(ks) // 2, len(ks) - 1):
        k_perm = ls.halves_to_interleaved(ks[i]).reshape(n, -1)
        print(f"layer {i:2d}: K rel.err {rel(k_perm, rk[i]):.4f} (unpermuted {rel(ks[i].reshape(n, -1), rk[i]):.3f}), "
              f"V rel.err {rel(vs[i].reshape(n, -1), rv[i]):.4f}")
    gpu.reset()
    info = npu_to_llama(npu, gpu, n)
    r = gpu.eval([prompt[-1]]); ho_logits = gpu.logits(0)
    gpu.truncate(n)
    ho_tokens = greedy_from_held(gpu, prompt[-1], a.gen)
    same = next((i for i, (x, y) in enumerate(zip(ref_tokens, ho_tokens)) if x != y), len(ref_tokens))
    print(f"first token: ref {int(np.argmax(ref_logits))} handoff {int(np.argmax(ho_logits))}; KL {kl(ref_logits, ho_logits):.2e}")
    print(f"greedy {a.gen} tokens identical up to {same}")
    print("ref:", repr(gpu.detokenize(ref_tokens)[:160]))
    print("ho :", repr(gpu.detokenize(ho_tokens)[:160]))
    print(f"timing: iGPU prefill {1e3 * t_gpu_prefill:.0f} ms | NPU prefill {1e3 * t_npu_prefill:.0f} ms, "
          f"export {1e3 * info.t_export:.0f} ms, convert {1e3 * info.t_convert:.0f} ms, import {1e3 * info.t_import:.0f} ms "
          f"({info.nbytes / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
