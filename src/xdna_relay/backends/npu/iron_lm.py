"""NPU executor over an IRON full-model graph (T-023), driven step by step from the host.

IRON's device-side sampling loop needs a newer amdxdna than this kernel ships
(docs/knowledge/pitfalls.md), so every call goes through ``Session.logits``.

Limits of this first version:
- ``eval(all_logits=True)`` with more than one token is not supported: IRON's prompt graph
  returns only the last row's logits (``CausalLM.forward``).
- No snapshot/restore yet; ``truncate`` uses ``Session.forget`` (free for a plain KV cache).
"""
from __future__ import annotations

import numpy as np

from xdna_relay.sched.executor import Executor, Rows, Snapshot


def _variant(model_class, precision: str | None, profile_dir: str | None):
    """``model_class`` with its matmul precision and/or its tuning-profile directory replaced."""
    import dataclasses
    from pathlib import Path

    from iron import Precision

    p = Precision(precision) if precision else None

    class Model(model_class):
        if profile_dir:
            profile = Path(profile_dir)

        @classmethod
        def default_config(cls, weights):
            c = super().default_config(weights)
            return dataclasses.replace(c, precision=p) if p else c

    Model.__name__ = f"{model_class.__name__}_{p.value if p else 'default'}"
    return Model


class IronLmExecutor(Executor):
    def __init__(self, hf_dir: str, model_class=None, max_seq_len: int = 2048, name: str | None = None,
                 precision: str | None = None, profile_dir: str | None = None):
        """``precision``: override the model's matmul precision ("high" = bf16 macs,
        "medium" = bfp16 macs; ``iron.Precision``). IRON picks "high" from 3B up.
        ``profile_dir``: a directory of tuned operator settings (``<device>.json``,
        ``bench/iron_tune.py``) in place of the application's shipped one."""
        if model_class is None:
            from iron.applications.llama.model import Llama as model_class
        if precision or profile_dir:
            model_class = _variant(model_class, precision, profile_dir)
        self.model = model_class.from_checkpoint(hf_dir, max_seq_len=max_seq_len)
        self.model.load_versions(release=self.model.checkpoint.release)
        self.session = self.model.session
        self.tok = self.model.tokenizer
        self.name = name or f"iron:{hf_dir.rstrip('/').split('/')[-1]}" + (f"@{precision}" if precision else "")
        self.n_ctx = max_seq_len
        self.held: list[int] = []
        self.n_past = 0
        self._last = None

    def eval(self, tokens: list[int], all_logits: bool = False) -> Rows:
        if all_logits and len(tokens) > 1:
            raise NotImplementedError("IRON prompt graph returns the last row only")
        if not tokens:
            raise ValueError("eval of no tokens")
        logits = self.session.logits(self.held + list(tokens))
        self.held += [int(t) for t in tokens]
        self.n_past = len(self.held)
        self._last = logits.astype(np.float32)
        top2 = np.partition(self._last, -2)[-2:]
        return Rows(np.array([int(np.argmax(self._last))], np.int32),
                    np.array([float(top2[1] - top2[0])], np.float32))

    def logits(self, row: int) -> np.ndarray:
        if row != 0 or self._last is None:
            raise IndexError("only the last eval's single row is available")
        return self._last

    def truncate(self, n_keep: int) -> None:
        if not 0 <= n_keep <= self.n_past:
            raise ValueError(f"truncate to {n_keep} of {self.n_past}")
        self.session.forget(n_keep)
        del self.held[n_keep:]
        self.n_past = n_keep

    def tokenize(self, text: str, add_special: bool = True) -> list[int]:
        ids = self.tok.encoding.encode(text) if not add_special else self.tok(text)
        return [int(t) for t in ids]

    def detokenize(self, tokens: list[int]) -> str:
        return self.tok.decode(list(tokens))

    def is_eog(self, token: int) -> bool:
        return token in self._eog

    @property
    def _eog(self) -> set[int]:
        if not hasattr(self, "_eog_ids"):
            ids = set()
            for t in ("<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"):
                try:
                    ids.add(int(self.tok.encoding.encode(t)[0]))
                except Exception:
                    pass
            self._eog_ids = ids
        return self._eog_ids

    def snapshot(self) -> Snapshot:
        raise NotImplementedError("IRON executor snapshots: not yet")

    def restore(self, snap: Snapshot) -> None:
        raise NotImplementedError("IRON executor snapshots: not yet")

    # ---- KV export (T-016) ------------------------------------------------------------

    def export_kv(self, n: int | None = None, dtype=np.float32) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Per layer K and V of positions ``0..n-1`` as ``[n, kv_heads, head_dim]``, cast to
        ``dtype`` (None: IRON's own cache dtype, bf16, no copy beyond the read).

        K is post-RoPE in IRON's (Hugging Face) two-halves layout.
        """
        n = self.n_past if n is None else n
        g = self.model.prompt_chunk
        ks, vs = [], []
        for layer in self.model.layers:
            c = layer.cache
            ks.append(np.asarray(g.read(c.keys)[:n], dtype=dtype))
            vs.append(np.asarray(g.read(c.values)[:n], dtype=dtype))
        return ks, vs
