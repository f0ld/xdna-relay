"""T-041: speculative decoding in the batched engine: total decode tok/s vs concurrency.

Engine(Qwen3-8B Q4_K_M, draft Qwen3-0.6B Q8_0, batched). Per concurrency n (requests started
together, prompts cycled from spec_probe.PROMPTS) and mode: k=0 (no drafting; same engine),
adaptive (BatchK), fixed k. Prompts are prefilled once first (prefix cache), so the timed run
is decode. Columns: all_active_tps = tokens of all requests while every one is still
decoding (last first token .. first finish) / that time: the server's throughput at
concurrency n; min_req_tps = the slowest request's decode tok/s (speculation can slow a
low-acceptance request that shares rounds with high-acceptance ones); mean k, acceptance.
Usage: python bench/batched_spec.py [--n 1,2,4] [--tokens 128] [--modes 0,adaptive,3]
"""
import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spec_probe import PROMPTS  # noqa: E402
from tests.reference.models import QWEN3_06B_Q8, QWEN3_8B_Q4, qwen3_chat  # noqa: E402


def run(engine, prompts, n_tok, stamps=None):
    out = [None] * len(prompts)
    stamps = stamps if stamps is not None else [[] for _ in prompts]

    def one(i):
        out[i] = engine.complete(prompts[i], n_tok, on_token=lambda t: stamps[i].append(time.perf_counter()) and False)

    ts = [threading.Thread(target=one, args=(i,)) for i in range(len(prompts))]
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.perf_counter() - t0
    return out, wall


def main():
    from tdr_guard import guard
    guard()
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="1,2,4")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--modes", default="0,adaptive,3")
    a = ap.parse_args()
    from xdna_relay.server.engine import Engine
    from xdna_relay.sched.calibration_store import power_source
    ns = [int(x) for x in a.n.split(",")]
    prof = Path("/sys/firmware/acpi/platform_profile").read_text().strip()
    print(f"# power {power_source()}, platform profile {prof}")
    e = Engine(str(QWEN3_8B_Q4), slots=max(ns), n_ctx=2048, draft_gguf=str(QWEN3_06B_Q8))
    names = list(PROMPTS)
    print("n,mode,all_active_tps,min_req_tps,mean_k,accept_rate,prompts")
    for n in ns:
        ps = [e.tokenize(qwen3_chat(PROMPTS[names[i % len(names)]])) for i in range(n)]
        for mode in a.modes.split(","):
            e.spec.fixed = None if mode == "adaptive" else int(mode)
            e.reset_state()
            run(e, ps, 4)  # prefill, cached
            stamps = [[] for _ in ps]
            res, wall = run(e, ps, a.tokens, stamps)
            beg, end = max(s[0] for s in stamps), min(s[-1] for s in stamps)
            toks = sum(sum(1 for t in s if beg < t <= end) for s in stamps)
            all_active = toks / (end - beg) if end > beg else 0.0
            min_req = min((len(s) - 1) / (s[-1] - s[0]) for s in stamps if len(s) > 1)
            ks = [k for r in res for k in r.stats.ks]
            dr = sum(r.stats.drafted for r in res)
            acc = sum(r.stats.accepted for r in res)
            print(f"{n},{mode},{all_active:.1f},{min_req:.1f},{sum(ks) / max(len(ks), 1):.2f},"
                  f"{acc / dr if dr else 0:.2f},{'+'.join(names[i % len(names)] for i in range(n))}", flush=True)
    e.close()


if __name__ == "__main__":
    main()
