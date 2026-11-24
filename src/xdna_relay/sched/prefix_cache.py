"""Prefix cache (T-027): reuse a sequence's KV state across requests.

Two levels, used by ``prefill``:
1. The executor's live sequence (``Executor.held``): a new prompt that extends or shares a
   prefix with it truncates to the common prefix instead of resetting (multi-turn chat).
2. Host-RAM snapshots (``Executor.snapshot``), LRU by bytes, keyed per model by the tokens
   they hold. The entry with the longest common prefix is restored, then truncated to it
   (dense KV caches truncate freely; recurrent models need snapshots at exact lengths —
   ``truncate`` raises and the caller falls back to a reset).
"""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass

from .executor import Executor, Snapshot


def common_prefix(a, b) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


@dataclass
class PrefillInfo:
    reused_live: int = 0  # tokens kept from the live sequence
    reused_cache: int = 0  # tokens restored from a snapshot
    evaluated: int = 0  # tokens actually run through the model
    t_restore: float = 0.0
    t_eval: float = 0.0
    t_snapshot: float = 0.0


class PrefixCache:
    def __init__(self, max_bytes: int = 4 << 30, min_tokens: int = 64):
        self.max_bytes = max_bytes
        self.min_tokens = min_tokens  # shorter prompts are cheaper to prefill than to store
        self._entries: OrderedDict[tuple[str, tuple[int, ...]], Snapshot] = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    def best(self, model: str, prompt: list[int]) -> tuple[Snapshot | None, int]:
        """The snapshot of ``model`` sharing the longest prefix with ``prompt``, and that length."""
        best, best_n = None, 0
        for (m, _), snap in self._entries.items():
            if m != model:
                continue
            n = common_prefix(snap.tokens, prompt)
            if n > best_n:
                best, best_n = snap, n
        if best is not None:
            self._entries.move_to_end((model, best.tokens))
        return best, best_n

    def put(self, snap: Snapshot) -> None:
        key = (snap.model, snap.tokens)
        if len(snap.tokens) < self.min_tokens or snap.nbytes > self.max_bytes:
            return
        old = self._entries.pop(key, None)
        if old is not None:
            self.bytes -= old.nbytes
        # an entry that is a strict prefix of the new one is redundant (the new one truncates)
        for k in [k for k in self._entries if k[0] == snap.model and snap.tokens[:len(k[1])] == k[1]]:
            self.bytes -= self._entries.pop(k).nbytes
        self._entries[key] = snap
        self.bytes += snap.nbytes
        while self.bytes > self.max_bytes:
            _, ev = self._entries.popitem(last=False)
            self.bytes -= ev.nbytes


def prefill(ex: Executor, prompt: list[int], cache: PrefixCache | None = None, store: bool = True):
    """Make ``ex`` hold ``prompt`` and return (last-row readout, PrefillInfo).

    At least the last prompt token is always evaluated (its logits start generation).
    """
    info = PrefillInfo()
    limit = len(prompt) - 1
    live = min(common_prefix(ex.held, prompt), limit)
    snap, n_snap = (cache.best(ex.name, prompt) if cache else (None, 0))
    n_snap = min(n_snap, limit)
    t = time.perf_counter()
    keep = live
    if snap is not None and n_snap > live:
        ex.restore(snap)
        keep = n_snap
        info.reused_cache = n_snap
        cache.hits += 1
    else:
        info.reused_live = live
        if cache is not None and live < limit:
            cache.misses += 1
    try:
        ex.truncate(min(keep, ex.n_past))
    except RuntimeError:  # e.g. recurrent state that cannot roll back to `keep`
        ex.reset()
        info.reused_cache = info.reused_live = 0
    info.t_restore = time.perf_counter() - t
    t = time.perf_counter()
    n0 = ex.n_past
    rows = ex.eval(prompt[n0:])
    info.evaluated = len(prompt) - n0
    info.t_eval = time.perf_counter() - t
    if cache is not None and store:
        t = time.perf_counter()
        cache.put(ex.snapshot())
        info.t_snapshot = time.perf_counter() - t
    return rows, info
