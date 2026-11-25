"""T-038: does online calibration converge on real traffic?

Rounds of the two-request scenario: A streams --gen-a tokens; 0.5 s after A's first token, B
arrives with a prompt from --b-words (cycled). Per round: where B's prefill ran, the server's
estimated vs measured TTFT (x_relay), A's tok/s while B waited. Every --every rounds the
server's calibration state (GET /x_relay/calibration).

Start the server with a deliberately wrong preset to see the fits move.
Usage: python bench/calibration_run.py --url http://127.0.0.1:8090 --rounds 24
"""
import argparse
import json
import threading
import time
import urllib.request

from hybrid_bench import A_TEXT, b_prompt
from server_load import stream


def calib(url):
    with urllib.request.urlopen(url + "/x_relay/calibration", timeout=30) as r:
        return json.load(r)


def main():
    from tdr_guard import guard
    guard()
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--rounds", type=int, default=24)
    ap.add_argument("--b-words", default="30,140,900,200,500,60,700,140")
    ap.add_argument("--gen-a", type=int, default=120)
    ap.add_argument("--gen-b", type=int, default=8)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--a-alone", type=float, default=15.5, help="A's tok/s alone (for client keep)")
    a = ap.parse_args()
    sizes = [int(w) for w in a.b_words.split(",")]
    print("round,b_tokens,placement,npu_tokens,est_ttft_ms,ttft_ms,err_pct,a_tps_during,keep_client,keep_server",
          flush=True)
    for i in range(a.rounds):
        words = sizes[i % len(sizes)]
        a_st, b_st = [], []
        ta = threading.Thread(target=stream, args=(a.url, [{"role": "user", "content": f"[r{i}] " + A_TEXT}],
                                                     a.gen_a, a_st))
        ta.start()
        deadline = time.perf_counter() + 300
        while not a_st:
            if not ta.is_alive() or time.perf_counter() > deadline:
                raise SystemExit("A produced no token")
            time.sleep(0.01)
        time.sleep(0.5)
        tb0 = time.perf_counter()
        last = stream(a.url, [{"role": "user", "content": f"[r{i}] " + b_prompt(words)}], a.gen_b, b_st)
        ta.join()
        tb1 = b_st[0] if b_st else time.perf_counter()
        during = [s for s in a_st if tb0 <= s <= tb1]
        x = last["x_relay"]
        est, got = x["estimated_ttft_ms"], x["ttft_ms"]
        print(f"{i},{last['usage']['prompt_tokens']},{x['placement']},{x['npu_tokens']},{est:.0f},{got:.0f},"
              f"{100 * (est - got) / got:+.0f},{len(during) / max(tb1 - tb0, 1e-9):.1f},"
              f"{len(during) / max(tb1 - tb0, 1e-9) / a.a_alone:.2f},{x.get('others_keep')}", flush=True)
        if (i + 1) % a.every == 0:
            print("# calibration " + json.dumps(calib(a.url)["fits"]), flush=True)


if __name__ == "__main__":
    main()
