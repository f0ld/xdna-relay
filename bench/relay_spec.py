"""Our speculative loop (xdna_relay.sched.speculative) on the iGPU, same prompts as spec_probe.py.

Usage: python bench/relay_spec.py --target T.gguf [--draft D.gguf] [--k 3|adaptive|0] [--gen 256] [--dump DIR]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from spec_probe import PROMPTS  # noqa: E402

from xdna_relay.backends.igpu.llama import LlamaCppExecutor
from xdna_relay.sched.speculative import AdaptiveK, generate


def qwen3_chat(text: str) -> str:
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft")
    ap.add_argument("--k", default="adaptive")
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--label", default="")
    ap.add_argument("--dump")
    a = ap.parse_args()
    tgt = LlamaCppExecutor(a.target, n_ctx=4096)
    dft = LlamaCppExecutor(a.draft, n_ctx=4096) if a.draft else None
    print("label,prompt,gen_tokens,decode_tps,rounds,drafted,accepted,accept_rate,mean_k,t_draft_s,t_verify_s")
    for name, text in PROMPTS.items():
        prompt = tgt.tokenize(qwen3_chat(text), add_special=False)
        pol = AdaptiveK(fixed=None if a.k == "adaptive" else int(a.k))
        st = generate(tgt, prompt, a.gen, dft, pol)
        rate = st.accepted / st.drafted if st.drafted else 0
        mk = sum(st.ks) / len(st.ks) if st.ks else 0
        print(f"{a.label},{name},{len(st.tokens)},{st.decode_tps:.2f},{st.rounds},{st.drafted},{st.accepted},"
              f"{rate:.2f},{mk:.2f},{st.t_draft:.2f},{st.t_verify:.2f}", flush=True)
        if a.dump:
            os.makedirs(a.dump, exist_ok=True)
            with open(os.path.join(a.dump, f"{a.label}-{name}.txt"), "w") as f:
                f.write(tgt.detokenize(st.tokens))


if __name__ == "__main__":
    main()
