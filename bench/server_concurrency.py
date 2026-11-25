"""T-035: N clients stream at once (short prompts, --gen tokens each); per-client and total
decode tok/s, measured client-side from the streamed tokens.

Usage: python bench/server_concurrency.py --url http://127.0.0.1:8090 --clients 1,2,4
"""
import argparse
import threading
import time

from server_load import stream

TOPICS = ["a lighthouse keeper", "a river delta", "a robot gardener", "a night train",
          "a mountain village", "a glass city", "an old library", "a desert well"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--clients", default="1,2,4")
    ap.add_argument("--gen", type=int, default=128)
    a = ap.parse_args()
    print("clients,per_client_tps_min,per_client_tps_mean,total_tps,wall_s", flush=True)
    for n in map(int, a.clients.split(",")):
        stamps = [[] for _ in range(n)]
        msgs = [[{"role": "user", "content": f"[{n}] Write a long story about {TOPICS[i % len(TOPICS)]}."}]
                for i in range(n)]
        ts = [threading.Thread(target=stream, args=(a.url, msgs[i], a.gen, stamps[i])) for i in range(n)]
        t0 = time.perf_counter()
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        wall = time.perf_counter() - t0
        rates = [(len(s) - 1) / (s[-1] - s[0]) for s in stamps if len(s) > 1]
        start, end = max(s[0] for s in stamps), min(s[-1] for s in stamps)
        overlap = sum(sum(1 for x in s if start <= x <= end) for s in stamps) / max(end - start, 1e-9)
        print(f"{n},{min(rates):.1f},{sum(rates) / len(rates):.1f},{overlap:.1f},{wall:.1f}", flush=True)


if __name__ == "__main__":
    main()
