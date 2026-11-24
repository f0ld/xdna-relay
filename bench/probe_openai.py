"""Probe an OpenAI-compatible server (FLM or llama-server) for prefill/decode speed.

Usage: python bench/probe_openai.py --url http://127.0.0.1:52700 --model qwen3.5:4b \
           --prompt-words 100,400,1600 --gen 64

Prints one CSV row per request. Speeds come from the server's own counters:
FLM: usage.prefill_speed_tps / decoding_speed_tps; llama-server: timings.prompt_per_second /
predicted_per_second. Wall time is measured here.
"""
import argparse
import json
import sys
import time
import urllib.request

FILLER = ("The quick brown fox jumps over the lazy dog while the river keeps flowing past "
          "old stone bridges and quiet green fields under a pale morning sky. ")


def make_prompt(words: int, salt: int) -> str:
    base = FILLER.split()
    body = " ".join(base[i % len(base)] for i in range(words))
    # salt at the start defeats prompt caches between runs
    return f"[{salt}] {body}\nSummarize the text above in one sentence."


def post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-words", default="100,400,1600")
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    print("label,prompt_tokens,gen_tokens,prefill_tps,decode_tps,ttft_s,wall_s")
    for i, w in enumerate(int(x) for x in a.prompt_words.split(",")):
        p = {"model": a.model, "messages": [{"role": "user", "content": make_prompt(w, int(time.time()) + i)}],
             "max_tokens": a.gen, "temperature": 0, "ignore_eos": True}
        t0 = time.perf_counter()
        r = post(a.url, p)
        wall = time.perf_counter() - t0
        u, t = r.get("usage", {}), r.get("timings")
        if t:  # llama-server
            pt, gt = t["prompt_n"], t["predicted_n"]
            pre, dec, ttft = t["prompt_per_second"], t["predicted_per_second"], t["prompt_ms"] / 1e3
        else:  # FLM
            pt, gt = u["prompt_tokens"], u["completion_tokens"]
            pre, dec, ttft = u["prefill_speed_tps"], u["decoding_speed_tps"], u["prefill_duration_ttft"]
        print(f"{a.label},{pt},{gt},{pre:.1f},{dec:.2f},{ttft:.3f},{wall:.3f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
