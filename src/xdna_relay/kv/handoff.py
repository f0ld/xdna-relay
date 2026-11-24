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


def npu_to_llama(src, dst, n: int | None = None, k_layout: str = "halves") -> HandoffInfo:
    """Make ``dst`` (LlamaCppExecutor) hold ``src``'s first ``n`` positions (IronLmExecutor).

    ``k_layout``: the source's K channel order ("halves" = Hugging Face, IRON's).
    """
    n = src.n_past if n is None else n
    if n < 1:
        raise ValueError("nothing to hand off")
    tpl = _template(dst, src.held[0])
    t0 = time.perf_counter()
    ks, vs = src.export_kv(n)
    t1 = time.perf_counter()
    if k_layout == "halves":
        ks = [ls.halves_to_interleaved(k) for k in ks]
    ks = [k.reshape(n, -1) for k in ks]
    vs = [v.reshape(n, -1) for v in vs]
    blob = ls.build(ls.from_kv(tpl, ks, vs))
    t2 = time.perf_counter()
    dst.restore(Snapshot(dst.name, tuple(src.held[:n]), np.frombuffer(blob, dtype=np.uint8)))
    t3 = time.perf_counter()
    return HandoffInfo(n, t1 - t0, t2 - t1, t3 - t2, len(blob))
