"""T-017: two overlapping requests; B's prefill on the iGPU vs on the NPU (+ KV handoff).

Request A (short prompt) decodes --gen-a tokens on iGPU context 1. One second after A's first
token, request B arrives with a long prompt; B decodes --gen-b tokens on iGPU context 2.
  igpu  : B's prefill runs on iGPU context 2, concurrently with A's decode.
  hybrid: B's prefill runs on the NPU (IRON), then the KV goes to iGPU context 2.
Reported: A's tok/s while B is in prefill and overall, B's TTFT, makespan.

Usage: python bench/hybrid_bench.py --hf HF_DIR --gguf MODEL.gguf [--b-words 900]
"""
import argparse
import threading
import time

from xdna_relay.backends.igpu.llama import LlamaCppExecutor
from xdna_relay.backends.npu.iron_lm import IronLmExecutor
from xdna_relay.backends.npu.worker import NpuWorker
from xdna_relay.kv.handoff import npu_to_llama, worker_to_llama

A_TEXT = "Write a long, detailed essay about the history of lighthouses and their keepers."
B_TEXT = ("Router r{i} peers with r{j} over BGP, uses MTU 9000 and sits in OSPF area {a}. ")


def b_prompt(words):
    s, i = "", 0
    while len(s.split()) < words:
        s += B_TEXT.format(i=i, j=i + 1, a=i % 4)
        i += 1
    return s + "\nWhich OSPF area is router r7 in?\n"


def decode(ex, first_token, n, stamps):
    t = first_token
    for _ in range(n):
        t = int(ex.eval([t]).tokens[-1])
        stamps.append(time.perf_counter())
        if ex.is_eog(t):
            break


def run(mode, gpu_a, gpu_b, npu, a_prompt, b_prompt_toks, gen_a, gen_b, policy=None):
    gpu_a.reset(); gpu_b.reset()
    if npu is not None and hasattr(npu, "reset"):
        npu.reset()
    a_stamps, res = [], {}
    t0 = time.perf_counter()
    a_first = int(gpu_a.eval(a_prompt).tokens[-1])
    a_stamps.append(time.perf_counter())
    ta = threading.Thread(target=decode, args=(gpu_a, a_first, gen_a - 1, a_stamps))
    ta.start()
    time.sleep(1.0)
    tb0 = time.perf_counter()
    if mode == "auto":
        c = policy.choose(len(b_prompt_toks) - 1, active=1)
        res["choice"] = c
        mode = {"npu": "hybrid-proc", "split": "split", "igpu": "igpu"}[c.device]
        split_k = c.npu_tokens
    elif mode == "split":
        split_k = (len(b_prompt_toks) - 1) // 256 * 256
        if split_k == 0:  # shorter than one NPU block: nothing to split
            mode = "igpu"
    res["ran"] = mode
    if mode == "igpu":
        b_first = int(gpu_b.eval(b_prompt_toks).tokens[-1])
    elif mode in ("hybrid-proc", "split"):
        k = len(b_prompt_toks) - 1 if mode == "hybrid-proc" else split_k
        res["handoff"] = h = worker_to_llama(npu, gpu_b, b_prompt_toks[:k])
        print("  npu " + " ".join(f"{k} {1e3 * v:.0f}" for k, v in npu.last_timing.items())
              + f" | ipc {1e3 * h.t_convert:.0f} import {1e3 * h.t_import:.0f} ms"
              + (f" | iGPU tail {len(b_prompt_toks) - k} tokens" if mode == "split" else ""), flush=True)
        b_first = int(gpu_b.eval(b_prompt_toks[k:]).tokens[-1])
    else:
        npu.eval(b_prompt_toks[:-1])
        res["handoff"] = npu_to_llama(npu, gpu_b)
        b_first = int(gpu_b.eval([b_prompt_toks[-1]]).tokens[-1])
    tb1 = time.perf_counter()
    b_stamps = []
    decode(gpu_b, b_first, gen_b - 1, b_stamps)
    ta.join()
    t_end = time.perf_counter()
    during = [s for s in a_stamps if tb0 <= s <= tb1]
    res.update(
        a_tps_during_b_prefill=len(during) / (tb1 - tb0),
        a_tps_overall=len(a_stamps) / (a_stamps[-1] - a_stamps[0]),
        b_ttft=tb1 - tb0,
        b_tps=len(b_stamps) / (b_stamps[-1] - tb1) if b_stamps else 0.0,
        makespan=t_end - t0,
    )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--b-words", type=int, default=900)
    ap.add_argument("--gen-a", type=int, default=200)
    ap.add_argument("--gen-b", type=int, default=32)
    ap.add_argument("--modes", default="igpu,hybrid")
    ap.add_argument("--preset", help="placement preset for mode auto (sched.placement.presets())")
    ap.add_argument("--precision", help="IRON matmul precision override (high, medium)")
    ap.add_argument("--profile-dir", help="IRON tuned profile directory (bench/iron_tune.py)")
    a = ap.parse_args()
    gpu_a = LlamaCppExecutor(a.gguf, n_ctx=4096, name="A")
    gpu_b = LlamaCppExecutor(a.gguf, n_ctx=4096)
    modes = a.modes.split(",")
    policy = None
    if "auto" in modes:
        from xdna_relay.sched.placement import PlacementPolicy, costs_for
        policy = PlacementPolicy(costs_for(a.preset))
    if {"hybrid-proc", "split", "auto"} & set(modes):
        npu = NpuWorker(a.hf, max_seq_len=2048, precision=a.precision, profile_dir=a.profile_dir)
    elif "hybrid" in modes:
        npu = IronLmExecutor(a.hf, max_seq_len=2048, precision=a.precision, profile_dir=a.profile_dir)
    else:
        npu = None
    a_prompt = gpu_a.tokenize(A_TEXT)
    bp = gpu_b.tokenize(b_prompt(a.b_words))
    # alone: A's decode speed with nothing else running
    gpu_a.reset(); st = []
    decode(gpu_a, int(gpu_a.eval(a_prompt).tokens[-1]), 64, st)
    print(f"model {gpu_b.name}; B prompt {len(bp)} tokens; A alone {len(st) / (st[-1] - st[0] + 1e-9):.1f} tok/s")
    if isinstance(npu, NpuWorker):  # warm once; vary the prompt so no prefix is reused
        worker_to_llama(npu, gpu_b, bp[:8] + [13])
    elif npu is not None:
        npu.eval(bp[:8]); npu.reset()
    print("mode,a_tps_during_b_prefill,a_tps_overall,b_ttft_s,b_tps,makespan_s,handoff_ms")
    for i, mode in enumerate(modes):
        # a fresh B prompt per run (salted), so no executor reuses a held prefix
        bpi = gpu_b.tokenize(f"[{i}] " + b_prompt(a.b_words))
        r = run(mode, gpu_a, gpu_b, npu, a_prompt, bpi, a.gen_a, a.gen_b, policy)
        h = r.get("handoff")
        hms = 1e3 * (h.t_convert + h.t_import + (0 if r["ran"] in ("hybrid-proc", "split") else h.t_export)) if h else 0
        if "choice" in r:
            c = r["choice"]
            print(f"  auto chose {c.device}: est. B TTFT {c.ttft_s:.2f} s, A lost {c.lost_decode_s:.2f} s "
                  f"(other: {c.other.device} {c.other.ttft_s:.2f} s / {c.other.lost_decode_s:.2f} s)" if c.other
                  else f"  auto chose {c.device}", flush=True)
        print(f"{mode},{r['a_tps_during_b_prefill']:.1f},{r['a_tps_overall']:.1f},{r['b_ttft']:.2f},"
              f"{r['b_tps']:.1f},{r['makespan']:.2f},{hms:.0f}", flush=True)
    if isinstance(npu, NpuWorker):
        npu.close()


if __name__ == "__main__":
    main()
