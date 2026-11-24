"""T-021: decode speed + draft acceptance on a fixed, realistic prompt set.

Usage: python bench/spec_probe.py --url http://127.0.0.1:52701 --label <config> [--gen 256]
Works against llama-server (timings.draft_n / draft_n_accepted) and FLM (no draft fields).
CSV: label,prompt,gen_tokens,decode_tps,draft_n,draft_accepted,accept_rate
"""
import argparse
import json
import urllib.request

CODE_SNIPPET = '''def merge(a, b):
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    out.extend(a[i:]); out.extend(b[j:])
    return out

def merge_sort(xs):
    if len(xs) <= 1:
        return xs
    mid = len(xs) // 2
    return merge(merge_sort(xs[:mid]), merge_sort(xs[mid:]))
'''

PROMPTS = {
    "code": "Write a Python function that reads a CSV file and returns per-column mean, min and max "
            "for numeric columns. Use type hints and a docstring. Only output code.",
    "explain": "Explain how TCP congestion control works: slow start, congestion avoidance, fast "
               "retransmit and fast recovery. Use plain prose.",
    "rewrite": "Add a short comment above every line of this code and return the full code:\n\n" + CODE_SNIPPET,
    "story": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="x")
    ap.add_argument("--label", required=True)
    ap.add_argument("--gen", type=int, default=256)
    a = ap.parse_args()
    print("label,prompt,gen_tokens,decode_tps,draft_n,draft_accepted,accept_rate")
    for name, text in PROMPTS.items():
        p = {"model": a.model, "messages": [{"role": "user", "content": text}], "max_tokens": a.gen,
             "temperature": 0, "cache_prompt": False,
             "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(p).encode(),
                                     headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            res = json.loads(r.read())
        t = res.get("timings")
        if t:
            gen, tps = t["predicted_n"], t["predicted_per_second"]
            dn, da = t.get("draft_n", 0), t.get("draft_n_accepted", 0)
        else:
            gen, tps, dn, da = res["usage"]["completion_tokens"], res["usage"]["decoding_speed_tps"], 0, 0
        rate = da / dn if dn else 0.0
        print(f"{a.label},{name},{gen},{tps:.2f},{dn},{da},{rate:.2f}", flush=True)


if __name__ == "__main__":
    main()
