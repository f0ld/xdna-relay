"""T-019 contention matrix: how much does a phase on one device slow a phase on the other?

Needs two servers already running:
  NPU  : FLM  `flm serve <tag> -p 52700`                      (--npu-url)
  iGPU : llama-server on Vulkan, `--port 52701 -np 1`          (--igpu-url)
and llama-bench for the iGPU victim measurements (--llama-bench, --gguf).

For each aggressor in {none, npu-prefill, npu-decode, igpu-prefill, igpu-decode} a thread
keeps that phase busy; the victim phase on the *other* device is measured meanwhile.
Output: CSV on stdout, one row per (aggressor, victim).
"""
import argparse
import json
import re
import subprocess
import threading
import time
import urllib.request

FILLER = ("The quick brown fox jumps over the lazy dog while the river keeps flowing past "
          "old stone bridges and quiet green fields under a pale morning sky. ").split()
STORY = "Write a very long, detailed story about a lighthouse keeper. Do not stop early."


def long_prompt(words: int, salt: int) -> str:
    return f"[{salt}] " + " ".join(FILLER[i % len(FILLER)] for i in range(words)) + "\nSummarize."


def chat(url: str, model: str, content: str, max_tokens: int) -> dict:
    p = {"model": model, "messages": [{"role": "user", "content": content}],
         "max_tokens": max_tokens, "temperature": 0, "cache_prompt": False}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(p).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def speeds(r: dict) -> tuple[float, float]:
    """(prefill tok/s, decode tok/s) from FLM usage or llama-server timings."""
    t = r.get("timings")
    if t:
        return t["prompt_per_second"], t["predicted_per_second"]
    u = r["usage"]
    return u["prefill_speed_tps"], u["decoding_speed_tps"]


class Aggressor(threading.Thread):
    def __init__(self, url: str, model: str, phase: str):
        super().__init__(daemon=True)
        self.url, self.model, self.phase = url, model, phase
        self.stop = threading.Event()
        self.samples: list[float] = []

    def run(self) -> None:
        i = 0
        while not self.stop.is_set():
            i += 1
            if self.phase == "prefill":
                r = chat(self.url, self.model, long_prompt(2400, int(time.time() * 1000) + i), 1)
                self.samples.append(speeds(r)[0])
            else:
                r = chat(self.url, self.model, f"[{i}] {STORY}", 256)
                self.samples.append(speeds(r)[1])


def npu_victim(a, phase: str) -> float:
    salt = int(time.time() * 1000)
    if phase == "prefill":
        return speeds(chat(a.npu_url, a.npu_model, long_prompt(1600, salt), 1))[0]
    return speeds(chat(a.npu_url, a.npu_model, f"[{salt}] {STORY}", 128))[1]


def igpu_victim(a, phase: str) -> float:
    args = ["-p", "1024", "-n", "0"] if phase == "prefill" else ["-p", "0", "-n", "96"]
    out = subprocess.run([a.llama_bench, "-m", a.gguf, "-ngl", "99", "-fa", "1", "-r", "2", "-o", "json", *args],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out[out.index("["):])[0]["avg_ts"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npu-url", default="http://127.0.0.1:52700")
    ap.add_argument("--npu-model", required=True)
    ap.add_argument("--igpu-url", default="http://127.0.0.1:52701")
    ap.add_argument("--llama-bench", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    print("label,aggressor,aggressor_tps,victim,victim_tps")
    cases = [(None, "npu"), (None, "igpu")]
    cases += [(("npu", p), "igpu") for p in ("prefill", "decode")]
    cases += [(("igpu", p), "npu") for p in ("prefill", "decode")]
    for agg, victim_dev in cases:
        for vphase in ("prefill", "decode"):
            g = None
            if agg:
                url, model = (a.npu_url, a.npu_model) if agg[0] == "npu" else (a.igpu_url, "x")
                g = Aggressor(url, model, agg[1])
                g.start()
                time.sleep(4)  # let the aggressor reach steady state
            v = (npu_victim if victim_dev == "npu" else igpu_victim)(a, vphase)
            if g:
                g.stop.set()
                g.join()
            atps = sum(g.samples) / len(g.samples) if g and g.samples else 0.0
            name = f"{agg[0]}-{agg[1]}" if agg else "none"
            print(f"{a.label},{name},{atps:.1f},{victim_dev}-{vphase},{v:.2f}", flush=True)


if __name__ == "__main__":
    main()
