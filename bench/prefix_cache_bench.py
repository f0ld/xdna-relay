"""T-027: TTFT with and without the prefix cache (Qwen3 chat, iGPU target, no draft).

Scenario A (multi-turn): 4 turns, each prompt = previous prompt + answer + new user turn.
Scenario B (shared system prompt): long system prompt + 4 questions, each followed by an
unrelated conversation on the same executor (live reuse impossible -> snapshot restore).
"""
import argparse

from xdna_relay.backends.igpu.llama import LlamaCppExecutor
from xdna_relay.sched.prefix_cache import PrefixCache
from xdna_relay.sched.speculative import generate

SYSTEM = ("You are a careful assistant for a network operations team. Answer precisely, cite the "
          "relevant RFC when you can, and keep answers under five sentences. Context follows. ")
CONTEXT = " ".join(f"Router r{i} peers with r{i+1} over BGP, MTU 9000, OSPF area {i % 4}." for i in range(90))
QS = ["What is BGP?", "Which routers are in OSPF area 2?", "What MTU do the links use?", "Who peers with r17?"]


def chat(system, turns):
    s = f"<|im_start|>system\n{system}<|im_end|>\n"
    for role, text in turns:
        s += f"<|im_start|>{role}\n{text}<|im_end|>\n"
    return s + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--gen", type=int, default=48)
    a = ap.parse_args()
    ex = LlamaCppExecutor(a.target, n_ctx=8192)
    print("scenario,mode,step,prompt_tokens,reused,evaluated,ttft_ms,restore_ms,snapshot_ms")
    for mode in ["cold", "cached"]:
        cache = PrefixCache() if mode == "cached" else None
        # A: multi-turn
        turns, ex_held_ok = [], True
        ex.reset()
        for i, q in enumerate(QS):
            turns.append(("user", q))
            p = ex.tokenize(chat(SYSTEM + CONTEXT, turns), add_special=False)
            if mode == "cold":
                ex.reset()
            st = generate(ex, p, a.gen, cache=cache)
            pi = st.prefill
            turns.append(("assistant", ex.detokenize(st.tokens).replace("<|im_end|>", "")))
            print(f"A,{mode},{i},{len(p)},{pi.reused_live + pi.reused_cache},{pi.evaluated},"
                  f"{1e3 * st.t_prefill:.0f},{1e3 * pi.t_restore:.0f},{1e3 * pi.t_snapshot:.0f}", flush=True)
        # B: shared system prompt, interleaved with another conversation
        for i, q in enumerate(QS):
            p = ex.tokenize(chat(SYSTEM + CONTEXT, [("user", q)]), add_special=False)
            ex.reset()
            st = generate(ex, p, a.gen, cache=cache)
            pi = st.prefill
            print(f"B,{mode},{i},{len(p)},{pi.reused_live + pi.reused_cache},{pi.evaluated},"
                  f"{1e3 * st.t_prefill:.0f},{1e3 * pi.t_restore:.0f},{1e3 * pi.t_snapshot:.0f}", flush=True)
            other = ex.tokenize(chat("You write haiku.", [("user", f"A haiku about rain #{i}")]), add_special=False)
            generate(ex, other, 16, cache=cache)
        if cache:
            print(f"# cache: {len(cache)} entries, {cache.bytes / 2**20:.0f} MiB, hits {cache.hits}, misses {cache.misses}")


if __name__ == "__main__":
    main()
