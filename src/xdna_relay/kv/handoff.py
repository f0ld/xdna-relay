"""KV handoff: a prefix prefilled on one executor continues on another (T-016).

NPU (IRON, HF RoPE layout, bf16) -> iGPU (llama.cpp, interleaved RoPE layout, f16 cache).
The llama.cpp side receives a synthesized ``llama_state_seq`` blob (``llama_state``), so
llama.cpp needs no patch for it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from xdna_relay.sched.executor import Snapshot

from . import llama_state as ls

_TEMPLATES: dict[str, ls.KVState] = {}


@dataclass
class HandoffInfo:
    tokens: int
    t_export: float  # reading the source KV
    t_convert: float  # permute, cast, build blob
    t_import: float  # restore into the destination
    nbytes: int


def _template(dst, token: int) -> ls.KVState:
    """Header fields of ``dst``'s cache, from a one-token snapshot (cached per model)."""
    if dst.name not in _TEMPLATES:
        held = list(dst.held)
        dst.reset()
        dst.eval([token])
        _TEMPLATES[dst.name] = ls.parse(dst.snapshot().data.tobytes())
        dst.reset()
        if held:  # leave dst as found
            dst.eval(held)
    return _TEMPLATES[dst.name]


def npu_to_llama(src, dst, n: int | None = None, interleave_k: bool | None = None) -> HandoffInfo:
    """Make ``dst`` (LlamaCppExecutor) hold ``src``'s first ``n`` positions (IronLmExecutor).

    ``interleave_k``: re-order K from IRON's (Hugging Face) two-halves RoPE layout to
    interleaved pairs, as llama.cpp caches Llama (not Qwen); default: the source's
    ``llama_cpp_k_interleaved``.
    """
    if interleave_k is None:
        interleave_k = src.llama_cpp_k_interleaved
    n = src.n_past if n is None else n
    if n < 1:
        raise ValueError("nothing to hand off")
    tpl = _template(dst, src.held[0])
    t0 = time.perf_counter()
    ks, vs = src.export_kv(n)
    t1 = time.perf_counter()
    if interleave_k:
        ks = [ls.halves_to_interleaved(k) for k in ks]
    ks = [k.reshape(n, -1) for k in ks]
    vs = [v.reshape(n, -1) for v in vs]
    blob = ls.build(ls.from_kv(tpl, ks, vs))
    t2 = time.perf_counter()
    dst.restore(Snapshot(dst.name, tuple(src.held[:n]), np.frombuffer(blob, dtype=np.uint8)))
    t3 = time.perf_counter()
    return HandoffInfo(n, t1 - t0, t2 - t1, t3 - t2, len(blob))


def worker_to_llama(worker, dst, tokens: list[int]) -> HandoffInfo:
    """Prefill ``tokens`` on ``worker`` (NpuWorker) and make ``dst`` (LlamaCppExecutor) hold
    them. The worker writes the finished blob into shared memory; ``dst`` restores from it
    in place. ``t_export`` is the worker's eval + KV read + pack, ``t_convert`` IPC overhead."""
    tpl = ls.meta(_template(dst, tokens[0]))
    t = {}

    def use(blob):
        t["got"] = time.perf_counter()
        dst.restore(Snapshot(dst.name, tuple(tokens), blob))
        t["restored"] = time.perf_counter()
        t["nbytes"] = blob.size

    t0 = time.perf_counter()
    worker.prefill_blob(tokens, tpl, use)
    w = sum(worker.last_timing.values())
    return HandoffInfo(len(tokens), w, (t["got"] - t0) - w, t["restored"] - t["got"], t["nbytes"])
