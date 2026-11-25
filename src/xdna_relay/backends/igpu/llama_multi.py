"""One llama.cpp context holding several sequences, for batched decode across requests (T-035).

Decode on the iGPU is weight-bandwidth bound: one batched step for N sequences costs about
what one step costs (Qwen3-8B Q4_K_M, llama-batched-bench: 15.3 tok/s at 1 sequence, 15.1
per sequence at 2, 14.4 at 4), while N separate contexts decoding at once split the bandwidth
(8.0 tok/s each at 2). ``SharedLlamaContext`` loads the model once; ``LlamaSeqExecutor`` is an
``Executor`` view of one sequence (prefill, truncate, snapshot/restore as before), and
``decode_step`` advances several sequences by one token in one ``llama_decode``.

KV: one buffer per sequence by default (``kv_unified=False``): same speed here, and a
sequence's results do not depend on where other sequences' cells sit (with a unified buffer,
1 of 4 reruns flipped a near-tie; bench/results/2026-11-25-batched-decode.md).

The context is not thread-safe: every call holds ``lock``. A long ``eval`` releases it
between ``n_batch`` chunks, so a batched decode step can run between prefill chunks.
"""
from __future__ import annotations

import ctypes as C
import threading
from pathlib import Path

import numpy as np

from xdna_relay.sched.executor import Executor, Rows, Snapshot

from .llama import _lib


def _bind(lib) -> None:
    if getattr(lib, "_multi_bound", False):
        return
    P, I = C.c_void_p, C.c_int
    IP, FP = C.POINTER(C.c_int), C.POINTER(C.c_float)
    sig = {
        "rl_load_multi": (P, [C.c_char_p, I, I, I, I, I, I, I, I]),
        "rl_eval_seq": (I, [P, I, IP, I, I, I]),
        "rl_decode_multi": (I, [P, I, IP, IP, IP]),
        "rl_decode_batch": (I, [P, I, IP, IP, IP, IP]),
        "rl_argmax_at": (I, [P, I, IP, IP, FP]),
        "rl_seq_truncate": (I, [P, I, I]),
        "rl_seq_pos_max": (I, [P, I]),
        "rl_model_kind": (I, [P]),
        "rl_mtp_enable": (I, [P]),
        "rl_chat_template": (C.c_char_p, [P]),
        "rl_mtp_draft": (I, [P, I, IP, IP, IP, IP, IP, I]),
        "rl_n_rs_seq": (I, [P]),
        "rl_state_seq_size": (C.c_size_t, [P, I]),
        "rl_state_seq_get": (C.c_size_t, [P, I, C.c_void_p, C.c_size_t]),
        "rl_state_seq_set": (C.c_size_t, [P, I, C.c_void_p, C.c_size_t]),
    }
    for name, (res, args) in sig.items():
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = res, args
    lib._multi_bound = True


class SharedLlamaContext:
    """A GGUF model loaded once, ``n_seq`` sequences in one context (unified KV of ``n_ctx``
    tokens in total)."""

    def __init__(self, gguf: str, *, n_seq: int, n_ctx: int = 16384, n_batch: int = 512,
                 n_gpu_layers: int = -1, flash_attn: int = 1, name: str | None = None,
                 kv_unified: bool = False, n_rs_seq: int = 0, mtp: bool = False):
        self._l = _lib()
        _bind(self._l)
        self._h = self._l.rl_load_multi(str(gguf).encode(), n_gpu_layers, n_ctx, n_batch, n_rs_seq, flash_attn,
                                        n_seq, 1 if kv_unified else 0, 1 if mtp else 0)
        if not self._h:
            raise RuntimeError(f"llama.cpp could not load {gguf}")
        self.name = name or Path(gguf).stem
        self.n_seq, self.n_ctx, self.n_batch = n_seq, n_ctx, n_batch
        # each sequence's context: all of it shared (unified KV), else its own n_ctx / n_seq
        self.n_ctx_seq = n_ctx if kv_unified else n_ctx // n_seq
        self.n_vocab = self._l.rl_n_vocab(self._h)
        # recurrent state (kind 1 recurrent, 2 hybrid) rolls back at most ``rollback`` tokens,
        # within the last batch; attention-only models truncate anywhere
        self.kind = self._l.rl_model_kind(self._h)
        self.rollback = self._l.rl_n_rs_seq(self._h) if self.kind else None
        # MTP (T-043): the model's own next-token head drafts; every decode keeps it current
        self.mtp = False
        if mtp:
            if self._l.rl_mtp_enable(self._h) != 0:
                raise RuntimeError(f"{gguf}: no MTP (nextn) layer, or its context failed")
            self.mtp = True
        self.lock = threading.RLock()
        self.decodes = 0  # llama_decode calls; a row readout is valid until the next one
        self._tok = (C.c_int * n_batch)()
        self._gap = (C.c_float * n_batch)()
        self.seqs = [LlamaSeqExecutor(self, i) for i in range(n_seq)]

    def close(self) -> None:
        with self.lock:
            if self._h:
                self._l.rl_free(self._h)
                self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _rows(self, k: int) -> Rows:
        return Rows(np.frombuffer(self._tok, dtype=np.int32, count=k).copy(),
                    np.frombuffer(self._gap, dtype=np.float32, count=k).copy())

    def decode_step(self, seqs: list["LlamaSeqExecutor"], tokens: list[int]) -> Rows:
        """Append ``tokens[i]`` to ``seqs[i]`` (one each) in one batched decode; row i of the
        result is seqs[i]'s next greedy token and its top1-top2 gap."""
        n = len(seqs)
        if n == 0 or n != len(tokens) or n > self.n_batch:
            raise ValueError(f"decode_step of {n} sequences")
        if len({s.seq for s in seqs}) != n:
            raise ValueError("a sequence twice in one step")
        toks = (C.c_int * n)(*tokens)
        pos = (C.c_int * n)(*[s.n_past for s in seqs])
        ids = (C.c_int * n)(*[s.seq for s in seqs])
        with self.lock:
            rc = self._l.rl_decode_multi(self._h, n, toks, pos, ids)
            if rc != 0:
                raise RuntimeError(f"llama_decode (batched) failed: {rc}")
            self.decodes += 1
            k = self._l.rl_argmax_rows(self._h, self._tok, self._gap)
            rows = self._rows(k)
        for s, t in zip(seqs, tokens):
            s.n_past += 1
            s.held.append(int(t))
            s._last_decode = -1  # its rows were part of a batch: logits() not available
        return rows

    def step(self, parts: list[tuple["LlamaSeqExecutor", list[int], bool | str]]) -> list:
        """One mixed batch: per part, append ``tokens`` to the sequence (a decode token, or a
        prefill chunk), with logits for the part's last token if its flag is True, for every
        token if it is ``"all"`` (a speculative verify). Returns per part (next greedy token,
        top1-top2 gap), a list of those (``"all"``), or None. At most one part per sequence."""
        if len({p[0].seq for p in parts}) != len(parts):
            raise ValueError("a sequence twice in one step")
        n = sum(len(t) for _, t, _ in parts)
        if n == 0 or n > self.n_batch:
            raise ValueError(f"step of {n} tokens (n_batch {self.n_batch})")
        toks, pos, ids, want, out_idx = [], [], [], [], []
        for s, t, logits in parts:
            if s.n_past + len(t) > s.n_ctx:
                raise ValueError(f"seq {s.seq}: {s.n_past}+{len(t)} tokens exceed its n_ctx {s.n_ctx}")
            for j, tok in enumerate(t):
                toks.append(int(tok)); pos.append(s.n_past + j); ids.append(s.seq)
                out = logits == "all" or (j == len(t) - 1 and logits)
                want.append(1 if out else 0)
                if out:
                    out_idx.append(len(toks) - 1)
        A = C.c_int * n
        idx = (C.c_int * max(1, len(out_idx)))(*out_idx)
        with self.lock:
            rc = self._l.rl_decode_batch(self._h, n, A(*toks), A(*pos), A(*ids), A(*want))
            if rc < 0 or rc != len(out_idx):
                raise RuntimeError(f"llama_decode (mixed batch) failed: {rc}")
            self.decodes += 1
            if out_idx:
                self._l.rl_argmax_at(self._h, len(out_idx), idx, self._tok, self._gap)
                rows = self._rows(len(out_idx))
        res, k = [], 0
        for s, t, logits in parts:
            s.n_past += len(t)
            s.held.extend(int(x) for x in t)
            s._last_decode = -1
            if logits == "all":
                res.append([(int(rows.tokens[k + j]), float(rows.gaps[k + j])) for j in range(len(t))])
                k += len(t)
            elif logits:
                res.append((int(rows.tokens[k]), float(rows.gaps[k])))
                k += 1
            else:
                res.append(None)
        return res

    def mtp_draft(self, seqs: list["LlamaSeqExecutor"], last: list[int], ks: list[int]) -> list[list[int]]:
        """Greedy MTP drafts: seqs[i] (holding seq[:-1]) continues from ``last[i]`` and drafts
        ks[i] tokens; the steps are batched across sequences."""
        n, kmax = len(seqs), max(ks, default=0)
        if n == 0 or kmax == 0:
            return [[] for _ in seqs]
        A = C.c_int * n
        out = (C.c_int * (n * kmax))()
        with self.lock:
            rc = self._l.rl_mtp_draft(self._h, n, A(*[s.seq for s in seqs]), A(*last),
                                      A(*[s.n_past for s in seqs]), A(*ks), out, kmax)
        if rc != 0:
            raise RuntimeError(f"MTP draft failed: {rc}")
        return [list(out[i * kmax:i * kmax + k]) for i, k in enumerate(ks)]

    @property
    def chat_template(self) -> str | None:
        """The GGUF's Jinja chat template (tokenizer.chat_template), if it carries one."""
        t = self._l.rl_chat_template(self._h)
        return t.decode() if t else None

    # shared vocabulary helpers
    def tokenize(self, text: str, add_special: bool = True) -> list[int]:
        b = text.encode()
        cap = len(b) + 16
        buf = (C.c_int * cap)()
        n = self._l.rl_tokenize(self._h, b, len(b), buf, cap, int(add_special), 1)
        if n < 0:
            raise RuntimeError("tokenize buffer too small")
        return list(buf[:n])

    def detokenize(self, tokens: list[int]) -> str:
        out = bytearray()
        buf = C.create_string_buffer(256)
        for t in tokens:
            n = self._l.rl_piece(self._h, t, buf, 256)
            out += buf.raw[:max(n, 0)]
        return out.decode("utf-8", errors="replace")

    def is_eog(self, token: int) -> bool:
        return bool(self._l.rl_is_eog(self._h, token))


class LlamaSeqExecutor(Executor):
    """Sequence ``seq`` of a ``SharedLlamaContext``. Snapshots carry the model name, so they
    restore into any sequence (and into a ``LlamaCppExecutor`` of the same GGUF name)."""

    def __init__(self, ctx: SharedLlamaContext, seq: int):
        self.ctx, self.seq = ctx, seq
        self.name = ctx.name
        self.n_vocab = ctx.n_vocab
        self.n_batch = ctx.n_batch
        self.n_ctx = ctx.n_ctx_seq
        self.rollback = ctx.rollback  # None: truncates anywhere; else at most this many tokens
        self.kv_streams = 1 if ctx.n_ctx_seq == ctx.n_ctx else ctx.n_seq  # streams in its state
        self.n_past = 0
        self.held: list[int] = []
        self._last_decode = -1
        self._last_rows = 0

    def eval(self, tokens: list[int], all_logits: bool = False) -> Rows:
        n = len(tokens)
        if n == 0:
            raise ValueError("eval of no tokens")
        if all_logits and n > self.n_batch:
            raise ValueError("all_logits for more than n_batch tokens")
        if self.n_past + n > self.n_ctx:
            raise ValueError(f"{self.n_past}+{n} tokens exceed the sequence's n_ctx {self.n_ctx}")
        c = self.ctx
        rows = None
        for beg in range(0, n, self.n_batch):  # the lock is released between chunks
            chunk = tokens[beg:beg + self.n_batch]
            last = beg + len(chunk) == n
            arr = (C.c_int * len(chunk))(*chunk)
            with c.lock:
                rc = c._l.rl_eval_seq(c._h, self.seq, arr, len(chunk), self.n_past, 1 if (last and all_logits) else 0)
                if rc != 0:
                    raise RuntimeError(f"llama_decode failed: {rc}")
                c.decodes += 1
                self.n_past += len(chunk)
                self.held.extend(int(t) for t in chunk)
                if last:
                    k = c._l.rl_argmax_rows(c._h, c._tok, c._gap)
                    rows = c._rows(k)
                    self._last_decode, self._last_rows = c.decodes, k
        return rows

    def logits(self, row: int) -> np.ndarray:
        c = self.ctx
        with c.lock:
            if self._last_decode != c.decodes:
                raise RuntimeError("logits: another decode ran since this sequence's last eval")
            out = np.empty(self.n_vocab, dtype=np.float32)
            if c._l.rl_logits_row(c._h, row, out.ctypes.data_as(C.POINTER(C.c_float))) != 0:
                raise IndexError(f"row {row} not in the last eval's output")
            return out

    def truncate(self, n_keep: int) -> None:
        if not 0 <= n_keep <= self.n_past:
            raise ValueError(f"truncate to {n_keep} of {self.n_past}")
        if n_keep == self.n_past:
            return
        with self.ctx.lock:
            if not self.ctx._l.rl_seq_truncate(self.ctx._h, self.seq, n_keep):
                raise RuntimeError(f"{self.name}[{self.seq}]: memory cannot roll back to {n_keep}")
        self.n_past = n_keep
        del self.held[n_keep:]

    def reset(self) -> None:
        self.truncate(0)

    def snapshot(self) -> Snapshot:
        c = self.ctx
        with c.lock:
            size = c._l.rl_state_seq_size(c._h, self.seq)
            buf = np.empty(size, dtype=np.uint8)
            got = c._l.rl_state_seq_get(c._h, self.seq, buf.ctypes.data, size)
        if got != size:
            raise RuntimeError(f"state copy returned {got} of {size} bytes")
        return Snapshot(self.name, tuple(self.held), buf, exact=self.rollback is not None)

    def restore(self, snap: Snapshot) -> None:
        if snap.model != self.name:
            raise ValueError(f"snapshot of {snap.model} restored into {self.name}")
        c = self.ctx
        with c.lock:
            got = c._l.rl_state_seq_set(c._h, self.seq, snap.data.ctypes.data, snap.data.size)
        if got == 0:
            raise RuntimeError("state restore failed")
        self.n_past = len(snap.tokens)
        self.held = list(snap.tokens)

    def tokenize(self, text: str, add_special: bool = True) -> list[int]:
        return self.ctx.tokenize(text, add_special)

    def detokenize(self, tokens: list[int]) -> str:
        return self.ctx.detokenize(tokens)

    def is_eog(self, token: int) -> bool:
        return self.ctx.is_eog(token)

    def close(self) -> None:  # the context owns the model
        pass
