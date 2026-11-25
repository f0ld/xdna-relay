"""T-032: warm NPU (IRON) prefill time vs prompt length, nothing else running.

Usage: python bench/npu_prefill_curve.py --hf HF_DIR [--lengths 64,128,256,512,1024,1493]
"""
import argparse
import time

from xdna_relay.backends.npu.iron_lm import IronLmExecutor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--lengths", default="64,128,256,384,512,768,1024,1493,2000")
    ap.add_argument("--packed", action="store_true", help="T-033 packed bfp16 projections (Qwen3)")
    ap.add_argument("--precision")
    a = ap.parse_args()
    ex = IronLmExecutor(a.hf, max_seq_len=2048, precision=a.precision, packed=a.packed)
    toks = ex.tokenize("Router r1 peers with r2 over BGP and sits in OSPF area 3. " * 300)
    ex.eval(toks[:32]); ex.truncate(0)  # warm
    print(f"{ex.name}\nn,seconds,tok_per_s")
    for n in map(int, a.lengths.split(",")):
        best = 1e9
        for _ in range(2):
            ex.truncate(0)
            t = time.perf_counter(); ex.eval(toks[:n]); best = min(best, time.perf_counter() - t)
        print(f"{n},{best:.3f},{n / best:.0f}", flush=True)


if __name__ == "__main__":
    main()
