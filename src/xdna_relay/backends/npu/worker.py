"""NPU executor in its own process (T-017).

IRON's host loop holds the GIL while it drives the NPU, which starves iGPU decode threads in
the same process (measured: A's decode 29 -> 6 tok/s during an in-process NPU prefill). The
worker owns the IronLmExecutor and writes the prefilled KV straight into shared memory as a
complete llama.cpp sequence-state blob (``kv.llama_state.layout``): K re-ordered to llama.cpp's RoPE
layout where it differs (Llama) and both cast to the destination cache type in one pass, so the
parent restores it without converting or copying.
"""
from __future__ import annotations

import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory

import numpy as np


def _f32(x: np.ndarray) -> np.ndarray:
    """bf16 -> f32 by bit shift (exact; ~2x faster than ml_dtypes' cast)."""
    if x.dtype.itemsize == 2 and x.dtype.name == "bfloat16":
        return (x.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    return x


def _pack(k: np.ndarray, v: np.ndarray, kd: np.ndarray, vd: np.ndarray, interleave: bool) -> None:
    """One layer, both cast to the cache type; K from HF halves to interleaved pairs if
    ``interleave`` (llama.cpp's Llama layout; Qwen keeps the halves)."""
    D = k.shape[-1]
    k = _f32(k)
    if interleave:
        kd = kd.reshape(k.shape)
        kd[..., 0::2] = k[..., : D // 2]
        kd[..., 1::2] = k[..., D // 2:]
    else:
        kd[:] = k.reshape(kd.shape)
    vd[:] = _f32(v).reshape(vd.shape)


def _serve(conn, hf_dir: str, max_seq_len: int, precision: str | None = None,
           profile_dir: str | None = None, packed: bool = False) -> None:
    from xdna_relay.backends.npu.iron_lm import IronLmExecutor
    from xdna_relay.kv import llama_state as ls

    ex = IronLmExecutor(hf_dir, max_seq_len=max_seq_len, precision=precision, profile_dir=profile_dir,
                        packed=packed)
    pool = ThreadPoolExecutor(4)  # numpy casts release the GIL; this process has no other work
    conn.send(("ready", ex.name))
    while True:
        cmd, arg = conn.recv()
        if cmd == "stop":
            return
        if cmd == "prefill":
            tokens, tpl = list(arg[0]), arg[1]
            keep = 0
            while keep < min(len(ex.held), len(tokens) - 1) and ex.held[keep] == tokens[keep]:
                keep += 1
            t0 = time.perf_counter()
            ex.truncate(keep)  # reuse a held prefix (free for a plain KV cache)
            ex.eval(tokens[keep:])
            t1 = time.perf_counter()
            n = len(tokens)
            ks, vs = ex.export_kv(n, dtype=None)
            t2 = time.perf_counter()
            shm = shared_memory.SharedMemory(create=True, size=ls.blob_size(tpl, n))
            _, kv, vv = ls.layout(tpl, n, out=shm.buf)
            il = [ex.llama_cpp_k_interleaved] * len(ks)
            list(pool.map(_pack, ks, vs, kv, vv, il))
            del kv, vv
            t3 = time.perf_counter()
            conn.send(("blob", (shm.name, shm.size, dict(eval=t1 - t0, export=t2 - t1, pack=t3 - t2))))
            shm.close()
            conn.recv()  # wait until the parent restored/unlinked it


class NpuWorker:
    def __init__(self, hf_dir: str, max_seq_len: int = 2048, precision: str | None = None,
                 profile_dir: str | None = None, packed: bool = False):
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_serve, args=(child, hf_dir, max_seq_len, precision, profile_dir, packed), daemon=True)
        self.proc.start()
        tag, self.name = self.conn.recv()
        assert tag == "ready"
        self.last_timing: dict[str, float] = {}

    def prefill_blob(self, tokens: list[int], tpl, use) -> None:
        """Prefill ``tokens`` on the NPU; call ``use(blob)`` with the llama.cpp state blob
        (uint8 array over shared memory, valid only during the call; header from ``tpl``,
        ``llama_state.meta`` of the destination's template)."""
        self.conn.send(("prefill", (list(tokens), tpl)))
        tag, (name, size, self.last_timing) = self.conn.recv()
        shm = shared_memory.SharedMemory(name=name)
        try:
            use(np.ndarray(size, np.uint8, buffer=shm.buf))
        finally:
            try:
                shm.close()
            except BufferError:  # a view escaped ``use``; unlink anyway
                pass
            shm.unlink()
            self.conn.send(("ack", None))

    def close(self) -> None:
        try:
            self.conn.send(("stop", None))
        except Exception:
            pass
        self.proc.join(timeout=10)
