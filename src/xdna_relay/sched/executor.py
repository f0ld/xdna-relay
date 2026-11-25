"""Executor interface (T-023): what the scheduler needs from a device running one model.

An executor holds one model and one sequence (its KV cache / recurrent state). The
scheduler appends tokens, reads greedy results or logits of the last call's rows, and rolls
the sequence back. It never sees device buffers (ADR 0003).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Rows:
    """Greedy readout of the last ``eval``: per output row, argmax token and top1-top2 logit gap."""

    tokens: np.ndarray  # int32 [rows]
    gaps: np.ndarray  # float32 [rows]


@dataclass
class Snapshot:
    """A sequence's device state copied to host memory, for the prefix cache."""

    model: str  # executor name; restore only into the same model
    tokens: tuple[int, ...]
    data: np.ndarray  # opaque bytes (uint8)
    exact: bool = False  # recurrent state: usable only as a whole prefix (cannot truncate)

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes)


class Executor(ABC):
    """One model, one sequence, on one device."""

    name: str
    n_past: int  # tokens the sequence holds (positions 0..n_past-1)
    held: list[int]  # those tokens

    @abstractmethod
    def eval(self, tokens: list[int], all_logits: bool = False) -> Rows:
        """Append ``tokens`` at positions ``n_past..``; return the greedy readout.

        ``all_logits``: one row per token (verify), else only the last token's row.
        """

    @abstractmethod
    def logits(self, row: int) -> np.ndarray:
        """Logits (float32 [vocab]) of ``row`` of the last ``eval``."""

    @abstractmethod
    def truncate(self, n_keep: int) -> None:
        """Keep positions ``0..n_keep-1``; later ``eval`` continues from ``n_keep``."""

    @abstractmethod
    def tokenize(self, text: str, add_special: bool = True) -> list[int]: ...

    @abstractmethod
    def detokenize(self, tokens: list[int]) -> str: ...

    @abstractmethod
    def is_eog(self, token: int) -> bool: ...

    @abstractmethod
    def snapshot(self) -> Snapshot: ...

    @abstractmethod
    def restore(self, snap: Snapshot) -> None: ...

    def reset(self) -> None:
        self.truncate(0)
