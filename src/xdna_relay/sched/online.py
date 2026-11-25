"""Online calibration of the placement cost model (T-038).

The hand-fitted presets (``placement._PRESETS``) are priors: each was measured on one model
pair, one engine mode and one power state (AC vs battery changed times by up to ~10%). The
server measures every prefill it runs anyway, so the model can keep fitting itself:

- Prefill time per device: a line in one feature, fitted by recency-weighted least squares
  with a ridge pull toward the preset's line (``LinFit``). With no data it *is* the preset;
  data moves it; observations at a single length cannot make it degenerate.
    iGPU: seconds = a + b * (n / 1000), per (others decoding or not) and per size: a tail of
          at most SMALL tokens (after a handoff: 1 token, or a split's rest) costs about a
          decode step in the batched engine, far below the bulk line's intercept (measured:
          the single bulk line made every NPU option 0.3 s too slow; 2026-11-25-calibration.md).
    NPU:  seconds = a + b * ceil(n / 256), per (prompt version short/long, decoding or not);
          the versions split at ``short_rows`` (packed Qwen3's 640) even when the prior has one
          curve (it then seeds both);
          the NPU part is worker eval + KV pack + handoff, up to the slot holding n tokens.
  Every option is composed from parts: igpu = iGPU(n); npu = NPU(n - 1) + iGPU(1);
  split = NPU(k) + iGPU(n - k). Each part learns from its own observations.
- Decode slowdown ("keep"): the share of its speed a decoding request keeps while a prefill
  runs, from the stepper's step times: plain decode steps vs steps carrying a prefill chunk
  (iGPU) vs steps while the NPU worker is busy. EMA from the preset's value.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LinFit:
    """y = c0 + c1 * x by exponentially forgetting least squares, ridge-pulled to a prior."""

    c0: float  # prior intercept
    c1: float  # prior slope
    tau: float = 1.0  # prior strength, in observations
    forget: float = 0.97  # per-observation decay of older data (~33 effective)
    n: int = 0
    _s: np.ndarray = field(default_factory=lambda: np.zeros(5))  # sw, sx, sxx, sy, sxy

    def add(self, x: float, y: float) -> None:
        self._s *= self.forget
        self._s += np.array([1.0, x, x * x, y, x * y])
        self.n += 1

    def coef(self) -> tuple[float, float]:
        sw, sx, sxx, sy, sxy = self._s
        t = self.tau
        A = np.array([[sw + t, sx], [sx, sxx + t]])
        b = np.array([sy + t * self.c0, sxy + t * self.c1])
        c0, c1 = np.linalg.solve(A, b)
        return float(c0), float(c1)

    def __call__(self, x: float) -> float:
        c0, c1 = self.coef()
        return c0 + c1 * x


@dataclass
class Ema:
    value: float
    alpha: float = 0.05
    n: int = 0

    def add(self, x: float) -> None:
        self.value = (1 - self.alpha) * self.value + self.alpha * x
        self.n += 1


SMALL = 64  # iGPU parts of at most this many tokens get their own fit


class OnlineCosts:
    """Per-part prefill-time fits and decode-keep estimates, seeded from a ``DeviceCosts``."""

    def __init__(self, prior, tau: float = 1.0, forget: float = 0.97, short_rows: int = 640):
        self.prior = prior
        self.short_rows = prior.npu_short_rows or short_rows
        p = prior
        self._lock = threading.Lock()
        mk = lambda c0, c1: LinFit(c0, c1, tau=tau, forget=forget)  # noqa: E731
        # iGPU: x = n / 1000
        self.igpu = {}
        for small in (False, True):
            self.igpu[(False, small)] = mk(0.0, 1000.0 / p.igpu_prefill_tps)
            self.igpu[(True, small)] = mk(p.igpu_wait_s, p.igpu_slow * 1000.0 / p.igpu_prefill_tps)
        # NPU part (eval + pack + handoff), x = blocks. The preset's contended NPU time also
        # held the last token's iGPU eval; take that out so the parts compose.
        last = self.igpu[(True, True)](1 / 1000.0)
        hb = p.handoff_row_s * p.npu_block
        self.npu = {}
        for version, (fx, blk) in self._versions().items():
            self.npu[(version, False)] = mk(fx, blk + hb)
            self.npu[(version, True)] = mk(fx * p.npu_slow + p.npu_wait_s - last, blk * p.npu_slow)
        self.keep_igpu = Ema(p.keep_igpu)
        self.keep_npu = Ema(p.keep_npu)
        self._plain_step = Ema(0.0, alpha=0.1)

    def _versions(self) -> dict:
        p = self.prior
        long = (p.npu_fixed_s, p.npu_block_s)
        short = (p.npu_short_fixed_s, p.npu_short_block_s) if p.npu_short_rows else long
        return {"long": long, "short": short}

    def _version(self, k: int) -> str:
        return "short" if k <= self.short_rows else "long"

    def _blocks(self, k: int) -> int:
        return math.ceil(k / self.prior.npu_block)

    # ---- predictions --------------------------------------------------------------------

    def igpu_s(self, n: int, active: int) -> float:
        if n <= 0:
            return 0.0
        with self._lock:
            return max(0.0, self.igpu[(bool(active), n <= SMALL)](n / 1000.0))

    def npu_s(self, k: int, active: int) -> float:
        if k <= 0:
            return 0.0
        with self._lock:
            return max(0.0, self.npu[(self._version(k), bool(active))](self._blocks(k)))

    # ---- observations -------------------------------------------------------------------

    def observe_igpu(self, n: int, seconds: float, active: int) -> None:
        if n > 0 and seconds > 0:
            with self._lock:
                self.igpu[(bool(active), n <= SMALL)].add(n / 1000.0, seconds)

    def observe_npu(self, k: int, seconds: float, active: int) -> None:
        if k > 0 and seconds > 0:
            with self._lock:
                self.npu[(self._version(k), bool(active))].add(self._blocks(k), seconds)

    def observe_step(self, seconds: float, prefill_chunk: bool, npu_busy: bool) -> None:
        """One decode step's wall time; with a prefill chunk riding along, or while the NPU
        worker prefills. A decoder gets one token per step, so the speed it keeps is
        plain step time / this step's time."""
        with self._lock:
            if not prefill_chunk and not npu_busy:
                p = self._plain_step
                p.value = seconds if p.n == 0 else p.value
                p.add(seconds)
                return
            if self._plain_step.n < 3:
                return  # no baseline yet
            keep = min(1.0, self._plain_step.value / seconds)
            (self.keep_igpu if prefill_chunk else self.keep_npu).add(keep)

    def summary(self) -> dict:
        with self._lock:
            out = {f"igpu{':small' if sm else ''}{'+' if a else ''}": (f.n, *map(lambda v: round(v, 4), f.coef()))
                   for (a, sm), f in self.igpu.items()}
            out.update({f"npu:{v}{'+' if a else ''}": (f.n, *map(lambda x: round(x, 4), f.coef()))
                        for (v, a), f in self.npu.items()})
            out["keep_igpu"] = (self.keep_igpu.n, round(self.keep_igpu.value, 3))
            out["keep_npu"] = (self.keep_npu.n, round(self.keep_npu.value, 3))
            out["plain_step_s"] = (self._plain_step.n, round(self._plain_step.value, 4))
            return out
