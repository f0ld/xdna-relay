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
- Decode slowdown ("keep"): the share of its speed a decoding request keeps while another
  request is prefilled, per placement, measured over that request's whole window (arrival to
  first token): decode steps completed in the window x plain step time / window length.
  (Per-step ratios were biased: steps carrying the 1-token tails of NPU / split requests
  count as "prefill" steps yet barely slow anyone, and time outside the steps — request
  handling — was invisible: keep_iGPU 0.29 from steps vs 0.14 seen by the client;
  2026-11-25-calibration.md.) EMA from the preset's value.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LinFit:
    """y = c0 + c1 * x by exponentially forgetting least squares, ridge-pulled to a prior,
    plus a per-bin correction where observed.

    The line extrapolates to sizes never seen; a bin's correction (EMA of the residual of its
    observations against the current line) makes observed sizes predict what was measured
    where the cost is not linear (batched iGPU prefill moves in 256-token chunks: measured
    0.41 s at 83 tokens where the line, dominated by long prompts, said 0.62)."""

    c0: float  # prior intercept
    c1: float  # prior slope
    tau: float = 1.0  # prior strength, in observations
    forget: float = 0.97  # per-observation decay of older data (~33 effective)
    bin_width: float = 0.0  # x per bin (0: no bins)
    alpha: float = 0.3  # bin correction EMA weight
    n: int = 0
    _s: np.ndarray = field(default_factory=lambda: np.zeros(5))  # sw, sx, sxx, sy, sxy
    _bins: dict = field(default_factory=dict)  # bin -> residual EMA

    def _bin(self, x: float):
        return None if self.bin_width <= 0 else int(x // self.bin_width)

    def line(self, x: float) -> float:
        c0, c1 = self.coef()
        return c0 + c1 * x

    def add(self, x: float, y: float) -> None:
        self._s *= self.forget
        self._s += np.array([1.0, x, x * x, y, x * y])
        self.n += 1
        b = self._bin(x)
        if b is not None:
            r = y - self.line(x)
            old = self._bins.get(b)
            self._bins[b] = r if old is None else (1 - self.alpha) * old + self.alpha * r

    def coef(self) -> tuple[float, float]:
        sw, sx, sxx, sy, sxy = self._s
        t = self.tau
        A = np.array([[sw + t, sx], [sx, sxx + t]])
        b = np.array([sy + t * self.c0, sxy + t * self.c1])
        c0, c1 = np.linalg.solve(A, b)
        return float(c0), float(c1)

    def __call__(self, x: float) -> float:
        return self.line(x) + self._bins.get(self._bin(x), 0.0)


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
        mk = lambda c0, c1, w=0.0: LinFit(c0, c1, tau=tau, forget=forget, bin_width=w)  # noqa: E731
        IG, NP = 0.128, 1.0  # bins: 128 tokens (iGPU, x = n / 1000), one block (NPU)
        # iGPU: x = n / 1000
        self.igpu = {}
        for small in (False, True):
            self.igpu[(False, small)] = mk(0.0, 1000.0 / p.igpu_prefill_tps, IG)
            self.igpu[(True, small)] = mk(p.igpu_wait_s, p.igpu_slow * 1000.0 / p.igpu_prefill_tps, IG)
        # NPU part (eval + pack + handoff), x = blocks. The preset's contended NPU time also
        # held the last token's iGPU eval; take that out so the parts compose.
        last = self.igpu[(True, True)](1 / 1000.0)
        hb = p.handoff_row_s * p.npu_block
        self.npu = {}
        for version, (fx, blk) in self._versions().items():
            self.npu[(version, False)] = mk(fx, blk + hb, NP)
            self.npu[(version, True)] = mk(fx * p.npu_slow + p.npu_wait_s - last, blk * p.npu_slow, NP)
        self.keep_igpu = Ema(p.keep_igpu, alpha=0.2)  # one observation per request (reporting)
        self.keep_npu = Ema(p.keep_npu, alpha=0.2)
        # decode-seconds one other decoder loses while a request is prefilled, fitted like the
        # times (same features): short iGPU prefills are mostly a fixed wait during which the
        # others keep stepping, long ones stall them, so one keep fraction cannot fit both
        # (measured keep_iGPU 0.31-0.46 at 82-139 tokens, 0.13-0.19 at 398-1327).
        self.lost_igpu = {sm: mk(*(c * (1 - p.keep_igpu) for c in self.igpu[(True, sm)].coef()), IG)
                          for sm in (False, True)}
        self.lost_npu = {v: mk(*(c * (1 - p.keep_npu) for c in self.npu[(v, True)].coef()), NP)
                         for v in ("short", "long")}
        # a split's loss, measured from its own windows (x = total tokens / 1000): its iGPU tail
        # is a real prefill chunk, so it costs the others more than NPU blocks + a composed tail
        # (measured at 398 tokens: 0.81 s vs 0.42 composed). Prior: the long NPU loss per token.
        c0, c1 = self.lost_npu["long"].coef()
        self.lost_split = mk(c0, c1 * 1000.0 / p.npu_block, IG)
        self._plain_step = Ema(0.0, alpha=0.1)
        self._seen: dict[tuple, int] = {}  # observations per (part, bin), for exploration

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

    def _bin(self, device: str, n: int) -> tuple:
        if device == "igpu":
            return ("igpu", n <= SMALL, n // 256)
        return ("npu", self._version(n), self._blocks(n))

    def seen(self, device: str, n: int) -> int:
        """Observations of the ``device`` part near size ``n`` (same 256-row bin)."""
        with self._lock:
            return self._seen.get(self._bin(device, n), 0)

    def observe_igpu(self, n: int, seconds: float, active: int) -> None:
        if n > 0 and seconds > 0:
            with self._lock:
                self.igpu[(bool(active), n <= SMALL)].add(n / 1000.0, seconds)
                b = self._bin("igpu", n)
                self._seen[b] = self._seen.get(b, 0) + 1

    def observe_npu(self, k: int, seconds: float, active: int) -> None:
        if k > 0 and seconds > 0:
            with self._lock:
                self.npu[(self._version(k), bool(active))].add(self._blocks(k), seconds)
                b = self._bin("npu", k)
                self._seen[b] = self._seen.get(b, 0) + 1

    def observe_step(self, seconds: float, prefill_chunk: bool, npu_busy: bool) -> None:
        """One decode step's wall time; plain steps (no prefill chunk, NPU idle) set the
        baseline decode step that ``observe_window`` compares against."""
        if prefill_chunk or npu_busy:
            return
        with self._lock:
            p = self._plain_step
            p.value = seconds if p.n == 0 else p.value
            p.add(seconds)

    @property
    def plain_step_s(self) -> float | None:
        return self._plain_step.value if self._plain_step.n >= 3 else None

    def observe_window(self, device: str, seconds: float, steps: int, n: int = 0) -> float | None:
        """A request placed on ``device`` ("igpu" or "npu"; ``n`` tokens on it) took ``seconds``
        from arrival to its first token while the others completed ``steps`` decode steps: each
        lost seconds - steps x plain step of decode time (fitted per device and size) and kept
        steps x plain step / seconds of its speed (reported). A window covers the whole
        request, so for "npu" (``n`` = NPU tokens) it includes the 1-token iGPU tail's loss;
        "split" windows (``n`` = the prompt's total tokens) feed the split's own loss fit.
        Returns the keep (None: no baseline yet)."""
        if device not in ("igpu", "npu", "split") or seconds <= 0:
            return None
        with self._lock:
            if self._plain_step.n < 3:
                return None
            if device == "split":  # n = the prompt's total tokens
                ran = min(seconds, steps * self._plain_step.value)
                if n > 0:
                    self.lost_split.add(n / 1000.0, seconds - ran)
                return ran / seconds
            ran = min(seconds, steps * self._plain_step.value)
            keep = ran / seconds
            lost = seconds - ran
            if device == "igpu":
                self.keep_igpu.add(keep)
                if n > 0:
                    self.lost_igpu[n <= SMALL].add(n / 1000.0, lost)
            else:
                self.keep_npu.add(keep)
                if n > 0:
                    self.lost_npu[self._version(n)].add(self._blocks(n), lost)
            return keep

    def igpu_lost_s(self, n: int) -> float:
        """Decode-seconds one decoder loses while the iGPU prefills n tokens."""
        if n <= 0:
            return 0.0
        with self._lock:
            return max(0.0, self.lost_igpu[n <= SMALL](n / 1000.0))

    def split_lost_s(self, n: int, composed: float) -> float:
        """A split's loss for an n-token prompt: its fitted line + bin correction once split
        windows were observed, else the ``composed`` estimate."""
        with self._lock:
            if self.lost_split.n == 0:
                return composed
            return max(0.0, self.lost_split(n / 1000.0))

    def npu_lost_s(self, k: int) -> float:
        if k <= 0:
            return 0.0
        with self._lock:
            return max(0.0, self.lost_npu[self._version(k)](self._blocks(k)))

    def summary(self) -> dict:
        with self._lock:
            out = {f"igpu{':small' if sm else ''}{'+' if a else ''}": (f.n, *map(lambda v: round(v, 4), f.coef()))
                   for (a, sm), f in self.igpu.items()}
            out.update({f"npu:{v}{'+' if a else ''}": (f.n, *map(lambda x: round(x, 4), f.coef()))
                        for (v, a), f in self.npu.items()})
            out.update({f"lost_igpu{':small' if sm else ''}": (f.n, *map(lambda x: round(x, 4), f.coef()))
                        for sm, f in self.lost_igpu.items()})
            out.update({f"lost_npu:{v}": (f.n, *map(lambda x: round(x, 4), f.coef()))
                        for v, f in self.lost_npu.items()})
            out["lost_split"] = (self.lost_split.n, *map(lambda x: round(x, 4), self.lost_split.coef()))
            out["keep_igpu"] = (self.keep_igpu.n, round(self.keep_igpu.value, 3))
            out["keep_npu"] = (self.keep_npu.n, round(self.keep_npu.value, 3))
            out["plain_step_s"] = (self._plain_step.n, round(self._plain_step.value, 4))
            return out
