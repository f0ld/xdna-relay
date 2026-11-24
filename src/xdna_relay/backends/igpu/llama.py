"""llama.cpp (Vulkan, iGPU) executor over native/llama_shim (T-024).

Build the shim first: ``scripts/build-llama-shim.sh`` → ``build/native/libllama_shim.so``.
Override the library path with ``XDNA_RELAY_LLAMA_SHIM``.
"""
from __future__ import annotations

import ctypes as C
import os
from pathlib import Path

import numpy as np

from xdna_relay.sched.executor import Executor, Rows

_LIB = None
_DEFAULT = Path(__file__).resolve().parents[4] / "build" / "native" / "libllama_shim.so"


def _lib() -> C.CDLL:
    global _LIB
    if _LIB is None:
        path = os.environ.get("XDNA_RELAY_LLAMA_SHIM", str(_DEFAULT))
        lib = C.CDLL(path)
        P, I, F = C.c_void_p, C.c_int, C.c_float
        IP, FP = C.POINTER(C.c_int), C.POINTER(C.c_float)
        sig = {
            "rl_init": (None, [I]),
            "rl_load": (P, [C.c_char_p, I, I, I, I, I]),
            "rl_free": (None, [P]),
            "rl_n_vocab": (I, [P]),
            "rl_tokenize": (I, [P, C.c_char_p, I, IP, I, I, I]),
            "rl_piece": (I, [P, I, C.c_char_p, I]),
            "rl_is_eog": (I, [P, I]),
            "rl_eval": (I, [P, IP, I, I, I]),
            "rl_argmax_rows": (I, [P, IP, FP]),
            "rl_logits_row": (I, [P, I, FP]),
            "rl_truncate": (I, [P, I]),
            "rl_clear": (None, [P]),
            "rl_pos_max": (I, [P]),
            "rl_synchronize": (None, [P]),
        }
        for name, (res, args) in sig.items():
            fn = getattr(lib, name)
            fn.restype, fn.argtypes = res, args
        lib.rl_init(0 if os.environ.get("XDNA_RELAY_LLAMA_VERBOSE") else 1)
        _LIB = lib
    return _LIB


class LlamaCppExecutor(Executor):
    """One GGUF model on llama.cpp's Vulkan backend, one sequence."""

    def __init__(self, gguf: str, *, n_ctx: int = 8192, n_batch: int = 512, n_gpu_layers: int = -1,
                 n_rs_seq: int = 0, flash_attn: int = 1, name: str | None = None):
        self._l = _lib()
        self._h = self._l.rl_load(str(gguf).encode(), n_gpu_layers, n_ctx, n_batch, n_rs_seq, flash_attn)
        if not self._h:
            raise RuntimeError(f"llama.cpp could not load {gguf}")
        self.name = name or Path(gguf).stem
        self.n_vocab = self._l.rl_n_vocab(self._h)
        self.n_batch = n_batch
        self.n_ctx = n_ctx
        self.n_past = 0
        self._rows_tok = (C.c_int * n_batch)()
        self._rows_gap = (C.c_float * n_batch)()

    def close(self) -> None:
        if self._h:
            self._l.rl_free(self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def eval(self, tokens: list[int], all_logits: bool = False) -> Rows:
        n = len(tokens)
        if n == 0:
            raise ValueError("eval of no tokens")
        if self.n_past + n > self.n_ctx:
            raise ValueError(f"{self.n_past}+{n} tokens exceed n_ctx {self.n_ctx}")
        arr = (C.c_int * n)(*tokens)
        rc = self._l.rl_eval(self._h, arr, n, self.n_past, 1 if all_logits else 0)
        if rc != 0:
            raise RuntimeError(f"llama_decode failed: {rc}")
        self.n_past += n
        k = self._l.rl_argmax_rows(self._h, self._rows_tok, self._rows_gap)
        return Rows(np.frombuffer(self._rows_tok, dtype=np.int32, count=k).copy(),
                    np.frombuffer(self._rows_gap, dtype=np.float32, count=k).copy())

    def logits(self, row: int) -> np.ndarray:
        out = np.empty(self.n_vocab, dtype=np.float32)
        if self._l.rl_logits_row(self._h, row, out.ctypes.data_as(C.POINTER(C.c_float))) != 0:
            raise IndexError(f"row {row} not in the last eval's output")
        return out

    def truncate(self, n_keep: int) -> None:
        if not 0 <= n_keep <= self.n_past:
            raise ValueError(f"truncate to {n_keep} of {self.n_past}")
        if n_keep == self.n_past:
            return
        if n_keep == 0:
            self._l.rl_clear(self._h)
        elif not self._l.rl_truncate(self._h, n_keep):
            raise RuntimeError(f"{self.name}: memory cannot roll back to {n_keep} (recurrent model without n_rs_seq?)")
        self.n_past = n_keep

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
