"""T-034: the two-request scenario through the HTTP server (client side, streaming).

A (short prompt) streams --gen-a tokens; 1 s after A's first token, B (long prompt) arrives.
Reported: A's tok/s while B waits for its first token, B's TTFT, both totals, and the
server's x_relay (where B's prefill ran).

Usage: python bench/server_load.py --url http://127.0.0.1:8090 [--b-words 900]
"""
import argparse
import json
import threading
import time
import urllib.request

from hybrid_bench import A_TEXT, b_prompt


def stream(url, messages, max_tokens, stamps):
    body = json.dumps({"messages": messages, "max_tokens": max_tokens, "stream": True}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    last = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev["choices"][0]["delta"].get("content"):
                stamps.append(time.perf_counter())
            last = ev
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--b-words", type=int, default=900)
    ap.add_argument("--gen-a", type=int, default=200)
    ap.add_argument("--gen-b", type=int, default=32)
    ap.add_argument("--runs", type=int, default=2)
    a = ap.parse_args()
    print("run,a_tps_during_b_wait,a_tps_overall,b_ttft_s,b_placement,b_reused,a_total_s,b_total_s", flush=True)
    for run in range(a.runs):
        a_st, b_st, res = [], [], {}
        t0 = time.perf_counter()
        ta = threading.Thread(target=lambda: res.__setitem__("a", stream(
            a.url, [{"role": "user", "content": f"[{run}] " + A_TEXT}], a.gen_a, a_st)))
        ta.start()
        while not a_st:
            time.sleep(0.01)
        time.sleep(1.0)
        tb0 = time.perf_counter()
        last_b = stream(a.url, [{"role": "user", "content": f"[{run}] " + b_prompt(a.b_words)}], a.gen_b, b_st)
        ta.join()
        tb1 = b_st[0] if b_st else time.perf_counter()
        during = [s for s in a_st if tb0 <= s <= tb1]
        xr = last_b.get("x_relay", {})
        print(f"{run},{len(during) / (tb1 - tb0):.1f},{len(a_st) / (a_st[-1] - a_st[0]):.1f},{tb1 - tb0:.2f},"
              f"{xr.get('placement')},{xr.get('reused_tokens')},{a_st[-1] - t0:.2f},{b_st[-1] - tb0:.2f}", flush=True)


if __name__ == "__main__":
    main()
