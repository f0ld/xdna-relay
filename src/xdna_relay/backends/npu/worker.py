"""NPU executor in its own process (T-017).

IRON's host loop holds the GIL while it drives the NPU, which starves iGPU decode threads in
the same process (measured: A's decode 29 -> 6 tok/s during an in-process NPU prefill). The
worker owns the IronLmExecutor; prefilled KV comes back through shared memory already in
llama.cpp's channel order, as float16 ``[layers, 2 (K,V), n, kv_heads * head_dim]``.
"""
from __future__ import annotations

import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np


def _serve(conn, hf_dir: str, max_seq_len: int) -> None:
    from xdna_relay.backends.npu.iron_lm import IronLmExecutor
    from xdna_relay.kv.llama_state import halves_to_interleaved

    ex = IronLmExecutor(hf_dir, max_seq_len=max_seq_len)
    conn.send(("ready", ex.name))
    while True:
        cmd, arg = conn.recv()
        if cmd == "stop":
            return
        if cmd == "prefill":
            tokens = list(arg)
            keep = 0
            while keep < min(len(ex.held), len(tokens) - 1) and ex.held[keep] == tokens[keep]:
                keep += 1
            ex.truncate(keep)  # reuse a held prefix (free for a plain KV cache)
            ex.eval(tokens[keep:])
            ks, vs = ex.export_kv(len(tokens))
            n, L = len(tokens), len(ks)
            row = ks[0].shape[1] * ks[0].shape[2]
            shm = shared_memory.SharedMemory(create=True, size=L * 2 * n * row * 2)
            out = np.ndarray((L, 2, n, row), dtype=np.float16, buffer=shm.buf)
            for i in range(L):
                out[i, 0] = halves_to_interleaved(ks[i]).reshape(n, row)
                out[i, 1] = vs[i].reshape(n, row)
            conn.send(("kv", (shm.name, out.shape)))
            shm.close()
            conn.recv()  # wait until the parent copied/unlinked it


class NpuWorker:
    def __init__(self, hf_dir: str, max_seq_len: int = 2048):
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_serve, args=(child, hf_dir, max_seq_len), daemon=True)
        self.proc.start()
        tag, self.name = self.conn.recv()
        assert tag == "ready"

    def prefill_kv(self, tokens: list[int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Prefill ``tokens`` on the NPU; per-layer K, V float16 ``[n, row]`` in llama.cpp order."""
        self.conn.send(("prefill", list(tokens)))
        tag, (name, shape) = self.conn.recv()
        shm = shared_memory.SharedMemory(name=name)
        try:
            arr = np.ndarray(shape, dtype=np.float16, buffer=shm.buf).copy()
        finally:
            shm.close()
            shm.unlink()
            self.conn.send(("ack", None))
        return [arr[i, 0] for i in range(shape[0])], [arr[i, 1] for i in range(shape[0])]

    def close(self) -> None:
        try:
            self.conn.send(("stop", None))
        except Exception:
            pass
        self.proc.join(timeout=10)
