"""OpenAI-compatible HTTP front end (stdlib ``ThreadingHTTPServer``; one thread per request).

Endpoints: ``GET /health``, ``GET /v1/models``, ``POST /v1/chat/completions`` and
``POST /v1/completions`` (``stream`` true: server-sent events, ``data: [DONE]`` last).
Chat replies are split like llama-server's ``reasoning_format`` deepseek: text inside
``<think>...</think>`` goes to ``reasoning_content`` (``delta.reasoning_content`` when
streaming), the rest to ``content``. A prompt that ends inside an open ``<think>`` (Qwen3.6's
template opens it in the prompt) starts the reply in reasoning. ``reasoning_format: "none"``
in the request (or ``--reasoning-format none``) returns the raw text in ``content``.
Greedy decoding only: a ``temperature`` above 0 is accepted and ignored (reported in
``x_relay.notes``). ``stop`` strings end the text early. Each response carries ``x_relay``:
where the prefill ran, prefix tokens reused, TTFT.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _Text:
    """Incremental detokenization: emit only text that is complete UTF-8."""

    def __init__(self, engine):
        self.engine, self.tokens, self.sent = engine, [], 0

    def add(self, tok: int) -> str:
        if self.engine.is_eog(tok):
            return ""
        self.tokens.append(tok)
        text = self.engine.detokenize(self.tokens)
        if text.endswith("�"):  # an incomplete multi-byte character: wait for more
            return ""
        out, self.sent = text[self.sent:], len(text)
        return out


class _Reasoning:
    """Splits generated text into (field, text) parts: ``reasoning_content`` or ``content``.
    Holds back a partial ``<think>``/``</think>`` across pieces."""
    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self, thinking: bool):
        self.mode = "think" if thinking else "start"  # start: a reply may open with <think>
        self.buf = ""

    @staticmethod
    def opens_thinking(prompt_text: str) -> bool:
        """The prompt ends inside an open <think> (the last one is not closed)."""
        return prompt_text.rfind(_Reasoning.OPEN) > prompt_text.rfind(_Reasoning.CLOSE)

    def feed(self, piece: str) -> list[tuple[str, str]]:
        self.buf += piece
        out = []
        while self.buf:
            if self.mode == "start":
                b = self.buf.lstrip()
                if b.startswith(self.OPEN):
                    self.buf, self.mode = b[len(self.OPEN):], "think"
                elif self.OPEN.startswith(b):
                    break  # maybe a <think> still arriving
                else:
                    self.mode = "content"
            elif self.mode == "think":
                i = self.buf.find(self.CLOSE)
                if i >= 0:
                    if i:
                        out.append(("reasoning_content", self.buf[:i]))
                    self.buf, self.mode = self.buf[i + len(self.CLOSE):], "after"
                    continue
                hold = next((n for n in range(min(len(self.CLOSE) - 1, len(self.buf)), 0, -1)
                             if self.CLOSE.startswith(self.buf[-n:])), 0)
                if len(self.buf) > hold:
                    out.append(("reasoning_content", self.buf[:len(self.buf) - hold]))
                    self.buf = self.buf[len(self.buf) - hold:]
                break
            elif self.mode == "after":  # the blank lines after </think>
                self.buf = self.buf.lstrip()
                if self.buf:
                    self.mode = "content"
            else:
                out.append(("content", self.buf))
                self.buf = ""
        return out

    def flush(self) -> list[tuple[str, str]]:
        b, self.buf = self.buf, ""
        if not b or self.mode == "after":
            return []
        return [("reasoning_content" if self.mode == "think" else "content", b)]

    @staticmethod
    def split(text: str, thinking: bool) -> tuple[str, str]:
        r = _Reasoning(thinking)
        parts = r.feed(text) + r.flush()
        return ("".join(t for f, t in parts if f == "reasoning_content").strip(),
                "".join(t for f, t in parts if f == "content"))


def _relay_info(res, notes):
    return {"placement": res.placement, "npu_tokens": res.npu_tokens, "slot": res.slot,
            "reused_tokens": res.reused, "evaluated_tokens": res.evaluated,
            "ttft_ms": round(1e3 * res.ttft_s, 1), "estimated_ttft_ms": round(1e3 * res.estimate_ttft_s, 1),
            "total_ms": round(1e3 * res.total_s, 1), "notes": notes,
            "others_keep": None if res.keep_observed is None else round(res.keep_observed, 3),
            "explore": res.explore}


def _timings(res):
    """llama-server's ``timings`` object (clients such as chat UIs read speeds from it)."""
    gen = max(len(res.tokens) - 1, 0)  # the first token comes with the prompt
    p_ms, d_ms = 1e3 * res.ttft_s, 1e3 * max(res.total_s - res.ttft_s, 0.0)
    return {"cache_n": res.reused, "prompt_n": res.evaluated, "prompt_ms": round(p_ms, 3),
            "prompt_per_token_ms": round(p_ms / res.evaluated, 3) if res.evaluated else 0.0,
            "prompt_per_second": round(1e3 * res.evaluated / p_ms, 2) if p_ms > 0 else 0.0,
            "predicted_n": gen, "predicted_ms": round(d_ms, 3),
            "predicted_per_token_ms": round(d_ms / gen, 3) if gen else 0.0,
            "predicted_per_second": round(1e3 * gen / d_ms, 2) if d_ms > 0 else 0.0}


def make_handler(engine, chat_template, model_name: str, default_max_tokens: int = 256,
                 reasoning_format: str = "deepseek"):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quiet; the engine reports per request
            pass

        def _json(self, code: int, obj) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self._json(200, {"status": "ok"})
            if self.path == "/x_relay/calibration":
                online = getattr(engine.policy, "online", None)
                store = getattr(engine, "calibration", None)
                return self._json(200, {"online": online is not None,
                                        "file": str(store.path) if store else None,
                                        "source": store.source if store else None,
                                        "fits": online.summary() if online else None,
                                        "observations": len(engine.policy.log)})
            if self.path == "/v1/models":
                return self._json(200, {"object": "list", "data": [
                    {"id": model_name, "object": "model", "owned_by": "xdna-relay"}]})
            self._json(404, {"error": {"message": f"no route {self.path}"}})

        def do_POST(self):
            self._t_arrival = time.perf_counter()
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            except json.JSONDecodeError as e:
                return self._json(400, {"error": {"message": f"bad JSON: {e}"}})
            notes = []
            if req.get("temperature", 0) not in (0, 0.0, None):
                notes.append("temperature ignored: greedy decoding only")
            if self.path == "/v1/chat/completions":
                msgs = req.get("messages")
                if not isinstance(msgs, list) or not msgs:
                    return self._json(400, {"error": {"message": "messages: a non-empty list"}})
                try:
                    text = chat_template.render(msgs, **(req.get("chat_template_kwargs") or {}))
                except Exception as e:
                    return self._json(400, {"error": {"message": f"chat template: {e}"}})
                chat = True
                fmt = req.get("reasoning_format", reasoning_format)
                split = None if fmt == "none" else _Reasoning.opens_thinking(text)
            elif self.path == "/v1/completions":
                text = req.get("prompt")
                if not isinstance(text, str) or not text:
                    return self._json(400, {"error": {"message": "prompt: a non-empty string"}})
                chat, split = False, None
            else:
                return self._json(404, {"error": {"message": f"no route {self.path}"}})
            prompt = engine.tokenize(text)
            max_tokens = int(req.get("max_tokens") or req.get("max_completion_tokens") or default_max_tokens)
            stops = req.get("stop") or []
            stops = [stops] if isinstance(stops, str) else list(stops)
            if req.get("stream"):
                self._stream(prompt, max_tokens, stops, chat, notes, split)
            else:
                self._complete(prompt, max_tokens, stops, chat, notes, split)

        def _finish(self, text: str, stops: list[str]):
            for s in stops:
                i = text.find(s)
                if i >= 0:
                    return text[:i], True
            return text, False

        def _complete(self, prompt, max_tokens, stops, chat, notes, split=None):
            out, acc = _Text(engine), []

            def on_token(tok):
                acc.append(out.add(tok))
                return bool(stops) and self._finish("".join(acc), stops)[1]

            res = engine.complete(prompt, max_tokens, on_token, t_arrival=self._t_arrival)
            text, hit = self._finish("".join(acc) + (engine.detokenize(out.tokens)[out.sent:]), stops)
            reason = "stop" if hit or (res.tokens and engine.is_eog(res.tokens[-1])) else "length"
            usage = {"prompt_tokens": len(prompt), "completion_tokens": len(res.tokens),
                     "total_tokens": len(prompt) + len(res.tokens)}
            rid = f"{'chatcmpl' if chat else 'cmpl'}-{uuid.uuid4().hex[:24]}"
            msg = {"role": "assistant", "content": text}
            if split is not None:
                reasoning, msg["content"] = _Reasoning.split(text, split)
                if reasoning:
                    msg["reasoning_content"] = reasoning
            choice = ({"index": 0, "message": msg, "finish_reason": reason}
                      if chat else {"index": 0, "text": text, "finish_reason": reason})
            self._json(200, {"id": rid, "object": "chat.completion" if chat else "text_completion",
                             "created": int(time.time()), "model": model_name, "choices": [choice],
                             "usage": usage, "timings": _timings(res), "x_relay": _relay_info(res, notes)})

        def _stream(self, prompt, max_tokens, stops, chat, notes, split=None):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            rid = f"{'chatcmpl' if chat else 'cmpl'}-{uuid.uuid4().hex[:24]}"
            created = int(time.time())
            kind = "chat.completion.chunk" if chat else "text_completion"
            out, acc, state = _Text(engine), [], {"gone": False, "hit": False}

            def send(obj) -> bool:
                try:
                    self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
                    self.wfile.flush()
                    return True
                except (BrokenPipeError, ConnectionResetError):
                    state["gone"] = True
                    return False

            rsn = _Reasoning(split) if split is not None else None

            def chunk(delta: str, reason=None, extra=None, field="content"):
                c = ({"index": 0, "delta": {field: delta} if delta else {}, "finish_reason": reason}
                     if chat else {"index": 0, "text": delta, "finish_reason": reason})
                obj = {"id": rid, "object": kind, "created": created, "model": model_name, "choices": [c]}
                if extra:
                    obj.update(extra)
                return send(obj)

            def emit(piece: str) -> bool:
                if rsn is None:
                    return chunk(piece)
                return all(chunk(t, field=f) for f, t in rsn.feed(piece))

            if chat:
                send({"id": rid, "object": kind, "created": created, "model": model_name,
                      "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})

            def on_token(tok):
                piece = out.add(tok)
                if not piece:
                    return state["gone"]
                before = "".join(acc)
                acc.append(piece)
                text, hit = self._finish("".join(acc), stops) if stops else ("".join(acc), False)
                if hit:
                    state["hit"] = True
                    if len(text) > len(before):
                        emit(text[len(before):])
                    return True
                return not emit(piece)

            res = engine.complete(prompt, max_tokens, on_token, t_arrival=self._t_arrival)
            if state["gone"]:
                return
            if rsn is not None:
                for f, t in rsn.flush():
                    chunk(t, field=f)
            reason = "stop" if state["hit"] or (res.tokens and engine.is_eog(res.tokens[-1])) else "length"
            usage = {"prompt_tokens": len(prompt), "completion_tokens": len(res.tokens),
                     "total_tokens": len(prompt) + len(res.tokens)}
            chunk("", reason, {"usage": usage, "timings": _timings(res), "x_relay": _relay_info(res, notes)})
            send_done = self.wfile
            try:
                send_done.write(b"data: [DONE]\n\n")
                send_done.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def serve(engine, chat_template, model_name: str, host: str = "127.0.0.1", port: int = 8090,
          default_max_tokens: int = 256, reasoning_format: str = "deepseek"):
    httpd = ThreadingHTTPServer((host, port), make_handler(engine, chat_template, model_name, default_max_tokens,
                                                           reasoning_format))
    httpd.daemon_threads = True
    return httpd


def serve_in_thread(engine, chat_template, model_name: str, host: str = "127.0.0.1", port: int = 0):
    """Start the server in a background thread; returns (httpd, port). For tests."""
    httpd = serve(engine, chat_template, model_name, host, port)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]
