"""T-021: llama-server's own prompt caching on the T-027 scenarios (baseline for our prefix
cache, bench/prefix_cache_bench.py; same prompts, Qwen3-8B, 48 generated tokens).

llama-server reuses a slot's KV for a common prefix (`cache_prompt`) and keeps prompt states
in host RAM across slots/conversations (`--cache-ram`, default 8192 MiB). TTFT = wall clock
from request to the first streamed token (HTTP + restore + prefill); `cache_n` / `prompt_n`
from the server's timings: tokens reused / evaluated.

Usage: python bench/server_prompt_cache.py --model GGUF [--configs cold,default,np4]
"""
import argparse
import json
import subprocess
import time
import urllib.request

from prefix_cache_bench import CONTEXT, QS, SYSTEM, chat

SERVER = "/home/f0ld/.cache/xdna-relay/llama.cpp/build-vulkan/bin/llama-server"
PORT = 8091
CONFIGS = {
    "cold": (["--cache-ram", "0", "-np", "1"], False),
    "default": (["-np", "1"], True),  # cache_prompt + --cache-ram 8192 (defaults)
    "np4": (["-np", "4", "-c", "32768"], True),
}


def stream(prompt: str, n: int, cache: bool):
    body = json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0, "stream": True,
                       "cache_prompt": cache}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", body,
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; text = ""; timings = {}
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            if ev.get("content") and ttft is None:
                ttft = time.perf_counter() - t0
            text += ev.get("content", "")
            if ev.get("stop"):
                timings = ev.get("timings", {})
    return ttft, text, timings


def wait_ready():
    for _ in range(300):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(1)
    raise RuntimeError("server did not start")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gen", type=int, default=48)
    ap.add_argument("--configs", default="cold,default,np4")
    a = ap.parse_args()
    print("config,scenario,step,prompt_n,cache_n,ttft_ms,prompt_ms", flush=True)
    for name in a.configs.split(","):
        extra, cache = CONFIGS[name]
        args = [SERVER, "-m", a.model, "-ngl", "99", "-fa", "on", "--port", str(PORT), "-c", "8192",
                "--no-webui"] + extra
        if "-c" in extra:  # config overrides the context size
            i = args.index("-c"); del args[i:i + 2]
        srv = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            wait_ready()
            stream("Hello", 4, False)  # warm up
            turns = []
            for i, q in enumerate(QS):  # A: multi-turn
                turns.append(("user", q))
                ttft, text, tm = stream(chat(SYSTEM + CONTEXT, turns), a.gen, cache)
                turns.append(("assistant", text))
                print(f"{name},A,{i},{tm.get('prompt_n')},{tm.get('cache_n')},{1e3 * ttft:.0f},"
                      f"{tm.get('prompt_ms', 0):.0f}", flush=True)
            for i, q in enumerate(QS):  # B: shared system prompt, another conversation between
                ttft, _, tm = stream(chat(SYSTEM + CONTEXT, [("user", q)]), a.gen, cache)
                print(f"{name},B,{i},{tm.get('prompt_n')},{tm.get('cache_n')},{1e3 * ttft:.0f},"
                      f"{tm.get('prompt_ms', 0):.0f}", flush=True)
                stream(chat("You write haiku.", [("user", f"A haiku about rain #{i}")]), 16, cache)
        finally:
            srv.terminate(); srv.wait(timeout=30)


if __name__ == "__main__":
    main()
