"""T-044: server on Qwen3.6-35B-A3B: reasoning_content split; MTP state in snapshots.

1. Chat replies split like llama-server's reasoning_format deepseek: text inside
   <think>...</think> goes to reasoning_content, the rest to content. Qwen3.6's template opens
   <think> in the prompt, so the reply starts in reasoning without an opening tag. Streaming
   holds back a partial tag across pieces. ``reasoning_format: "none"`` returns raw text.
   (Stub engine: no hardware.)
2. A sequence restored from a snapshot drafts exactly like the live sequence it was taken
   from: the snapshot carries the MTP context's sequence state and the pending hidden state.
   Before T-044 a restore cleared both, so drafts after a prefix-cache hit started from a zero
   hidden state (weaker drafts, output still exact).
"""
import itertools
import json
import urllib.request

import pytest

from tests.reference.models import QWEN36_35B_A3B_MTP_Q4, require


# ---- 1. reasoning split (no hardware) -------------------------------------------------------

def _parts(text, thinking, step):
    from xdna_relay.server.http import _Reasoning
    r = _Reasoning(thinking)
    out = []
    for i in range(0, len(text), step):
        out += r.feed(text[i:i + step])
    out += r.flush()
    return [(f, "".join(t for _, t in g)) for f, g in itertools.groupby(out, key=lambda x: x[0])]


@pytest.mark.parametrize("step", [1, 2, 3, 5, 1000])
def test_split_prompt_opened_thinking(step):
    text = "First, 2+2.\nIt is 4.</think>\n\nThe answer is **4**."
    assert _parts(text, True, step) == [("reasoning_content", "First, 2+2.\nIt is 4."),
                                        ("content", "The answer is **4**.")]


@pytest.mark.parametrize("step", [1, 4, 1000])
def test_split_reply_opens_thinking(step):
    text = "<think>\nhm, short.\n</think>\n\nHi!"
    assert _parts(text, False, step) == [("reasoning_content", "\nhm, short.\n"), ("content", "Hi!")]


@pytest.mark.parametrize("step", [1, 1000])
def test_split_no_thinking(step):
    assert _parts("Hello <b>there</b> </thin", False, step) == [("content", "Hello <b>there</b> </thin")]


def test_split_cut_inside_reasoning():
    # max_tokens ended the reply while thinking: everything is reasoning, a partial tag too
    assert _parts("still thinking </thi", True, 1) == [("reasoning_content", "still thinking </thi")]


def test_split_helpers():
    from xdna_relay.server.http import _Reasoning as R
    assert R.opens_thinking("<|im_start|>assistant\n<think>\n")
    assert not R.opens_thinking("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert not R.opens_thinking("<|im_start|>assistant\n")
    assert R.split("a</think>\n\nb", True) == ("a", "b")


class _StubEngine:
    """Characters as tokens; replies with ``reply`` one character per token."""

    def __init__(self, reply):
        self.reply = reply

    def tokenize(self, text):
        return [ord(c) for c in text]

    def detokenize(self, toks):
        return "".join(chr(t) for t in toks)

    def is_eog(self, t):
        return t == 0

    def complete(self, prompt, max_tokens, on_token=None, t_arrival=None):
        from xdna_relay.server.engine import Result
        toks = [ord(c) for c in self.reply][:max_tokens] + [0]
        res = Result(ttft_s=0.01, total_s=0.02, evaluated=len(prompt))
        for t in toks:
            res.tokens.append(t)
            if on_token and on_token(t):
                break
        return res


class _StubTemplate:
    def render(self, msgs, enable_thinking=True, **kw):
        return "<|im_start|>assistant\n" + ("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n")


def _server(reply):
    import threading
    from xdna_relay.server.http import serve
    httpd = serve(_StubEngine(reply), _StubTemplate(), "stub", port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def _post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        if not body.get("stream"):
            return json.load(r)
        return [json.loads(line[6:]) for line in r.read().decode().splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"]


MSGS = [{"role": "user", "content": "2+2?"}]


def test_http_nonstream_reasoning():
    httpd, port = _server("Add them.</think>\n\n4")
    try:
        m = _post(port, {"messages": MSGS})["choices"][0]["message"]
        assert m["reasoning_content"] == "Add them." and m["content"] == "4"
        raw = _post(port, {"messages": MSGS, "reasoning_format": "none"})["choices"][0]["message"]
        assert raw["content"] == "Add them.</think>\n\n4" and "reasoning_content" not in raw
        off = _post(port, {"messages": MSGS, "chat_template_kwargs": {"enable_thinking": False}})
        assert off["choices"][0]["message"]["content"] == "Add them.</think>\n\n4"  # no open <think>
    finally:
        httpd.shutdown()


def test_http_stream_reasoning():
    httpd, port = _server("Add them.</think>\n\n4")
    try:
        chunks = _post(port, {"messages": MSGS, "stream": True})
        deltas = [c["choices"][0]["delta"] for c in chunks]
        reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
        content = "".join(d.get("content", "") for d in deltas)
        assert reasoning == "Add them." and content == "4"
        assert not any("<" in d.get("content", "") + d.get("reasoning_content", "") for d in deltas)
        assert "timings" in chunks[-1]
    finally:
        httpd.shutdown()


# ---- 2. MTP state in snapshots (hardware) ---------------------------------------------------

PROMPT = ("<|im_start|>user\nWrite a Python function that parses an ISO 8601 date string and returns "
          "the weekday name.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\ndef ")


@pytest.fixture(scope="module")
def engine(igpu):
    from xdna_relay.server.engine import Engine
    e = Engine(require(QWEN36_35B_A3B_MTP_Q4), slots=2, n_ctx=2048, mtp=True)
    yield e
    e.close()


def test_restored_sequence_drafts_like_live(engine):
    sh = engine.shared
    a, b = engine.slots[0].target, engine.slots[1].target
    toks = engine.tokenize(PROMPT)
    a.reset(), b.reset()
    a.eval(toks[:-1])
    snap = a.snapshot()
    live = sh.mtp_draft([a], [toks[-1]], [6])[0]
    b.restore(snap)
    restored = sh.mtp_draft([b], [toks[-1]], [6])[0]
    assert restored == live, (live, restored)
    # and the restored copy keeps drafting like the live one after a real target step
    a.eval([toks[-1]])
    t = int(a.logits(0).argmax())
    b.eval([toks[-1]])
    assert sh.mtp_draft([a], [t], [4])[0] == sh.mtp_draft([b], [t], [4])[0]
    a.reset(), b.reset()


def test_restore_rejects_a_foreign_blob(engine):
    b = engine.slots[1].target
    sh = engine.shared
    import numpy as np
    bad = np.zeros(64, dtype=np.uint8)
    with sh.lock:
        got = sh._l.rl_state_seq_set(sh._h, b.seq, bad.ctypes.data, bad.size)
    assert got == 0
    b.reset()
