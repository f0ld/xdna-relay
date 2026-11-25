"""Prefill placement (T-032): run a request's prefill on the iGPU or on the NPU (+ KV handoff).

The choice trades the new request's time to first token against the decode time the
requests already decoding on the iGPU lose while it prefills. Both devices share DRAM, so
either prefill slows those decoders (T-019, bench/results/2026-11-24-hybrid.md): an iGPU
prefill takes the iGPU itself (decoders keep ~20% of their speed), an NPU prefill only
memory bandwidth (decoders keep ~50%); the NPU prefill slows down under that contention too.

Cost of an option = TTFT + ``decode_weight`` x (decode-seconds lost by all active decoders).
A decode-second lost is a second of one request's output delayed, so ``decode_weight = 1``
weighs every waiting user equally.

Device models, from measurements (``costs_for`` presets, ``observe`` refines them online):
- iGPU prefill: ``n / igpu_prefill_tps``.
- NPU prefill (warm): ``npu_fixed_s + npu_block_s * ceil(n / npu_block)``, 256-row blocks
  (bench/npu_prefill_curve.py: steps every 256 rows, fit within 12% over 64..2000 tokens),
  plus ``handoff_row_s * n``.
- Contention: while decoders run, a prefill takes ``alone x slow + wait`` on either device
  (the iGPU's queues behind decode steps; the NPU's wait includes the handoff), fitted per
  model pair; a decoder keeps ``keep_igpu`` / ``keep_npu`` of its speed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace


@dataclass
class DeviceCosts:
    igpu_prefill_tps: float  # iGPU prefill, tokens/s, nothing else running
    npu_fixed_s: float  # per prefill call, s
    npu_block_s: float  # per started block of ``npu_block`` rows, s
    handoff_row_s: float  # KV pack + import per row, s
    igpu_slow: float  # iGPU prefill time factor while others decode
    npu_slow: float  # NPU prefill time factor while the iGPU decodes
    keep_igpu: float  # decoder speed kept during an iGPU prefill (fraction)
    keep_npu: float  # decoder speed kept during an NPU prefill (fraction)
    source: str = ""
    npu_block: int = 256
    igpu_wait_s: float = 0.0  # fixed extra per iGPU prefill while others decode
    npu_wait_s: float = 0.0  # fixed extra per NPU prefill while others decode (incl. handoff)
    # a second NPU prompt version for calls of at most npu_short_rows (0: none), e.g. packed
    # Qwen3's streamed-B version (models/qwen3.py Qwen3Packed)
    npu_short_rows: int = 0
    npu_short_fixed_s: float = 0.0
    npu_short_block_s: float = 0.0

    def npu_alone_s(self, n: int) -> float:
        blocks = math.ceil(n / self.npu_block)
        if n <= self.npu_short_rows:
            return self.npu_short_fixed_s + self.npu_short_block_s * blocks
        return self.npu_fixed_s + self.npu_block_s * blocks

    def igpu_alone_s(self, n: int) -> float:
        return n / self.igpu_prefill_tps


# Measured presets: (iGPU GGUF, NPU HF checkpoint) pairs, from bench/hybrid_bench.py runs
# with request A decoding (bench/results/2026-11-25-placement.md: raw points, fits).
# Contended TTFT = slow x (time alone) + wait, least squares over three prompt lengths; the
# NPU's wait includes handoff, IPC and the last token's iGPU eval.
_PRESETS: dict[str, DeviceCosts] = {
    # Llama 3.2 3B: iGPU Q4_K_M ~744 tok/s alone; NPU IRON medium 0.042 s + 0.266 s/256 rows.
    # B TTFT under A, 64/350/1494 tokens: iGPU 0.26-0.38 / 0.68-0.77 / 1.85-2.20 s,
    # NPU 0.60-0.62 / 0.94 / 2.30-2.39 s (fit errors 9% / 1%). A alone 35 tok/s; during an
    # iGPU prefill of 1494 tokens 6-8.9, during the NPU's 19.5-20.1.
    "llama3.2-3b": DeviceCosts(
        igpu_prefill_tps=744.0, npu_fixed_s=0.042, npu_block_s=0.266, handoff_row_s=185e-3 / 1494,
        igpu_slow=0.889, npu_slow=1.309, keep_igpu=7.4 / 35.0, keep_npu=19.8 / 35.0,
        igpu_wait_s=0.272, npu_wait_s=0.199,
        source="bench/results/2026-11-25-placement.md"),
    # Qwen3-8B: iGPU Q4_K_M ~404 tok/s alone; NPU IRON medium + MHA profile entry 0.024 s +
    # 0.620 s/256 rows. B TTFT under A, 67/382/1702 tokens: iGPU 0.57-0.72 / 1.81-1.84 /
    # 5.52-6.05 s, NPU 1.06 / 1.77-1.80 / 5.59-5.66 s (fit errors 14% / 2%). A alone
    # 15.7-16 tok/s; during an iGPU prefill of 1702 tokens 2.8-3.7, during the NPU's 8.1-8.6.
    "qwen3-8b": DeviceCosts(
        igpu_prefill_tps=404.0, npu_fixed_s=0.024, npu_block_s=0.620, handoff_row_s=265e-3 / 1702,
        igpu_slow=1.259, npu_slow=1.229, keep_igpu=3.2 / 15.8, keep_npu=8.4 / 15.8,
        igpu_wait_s=0.527, npu_wait_s=0.251,
        source="bench/results/2026-11-25-placement.md"),
    # Qwen3-8B with packed bfp16 projections (T-033), two prompt versions: up to 640 rows
    # streamed B (0.100 s + 0.459 s/256 rows), above resident B (0.510 s + 0.362 s/256 rows).
    # B TTFT under A, 67/382/1702 tokens: NPU 0.96 / 1.57-1.63 / 4.50-4.78 s (fit 2%); iGPU
    # side as "qwen3-8b". A during the NPU's 1702-token prefill 10.4-10.7 tok/s (alone 16).
    "qwen3-8b-packed": DeviceCosts(
        igpu_prefill_tps=404.0, npu_fixed_s=0.510, npu_block_s=0.362, handoff_row_s=250e-3 / 1702,
        igpu_slow=1.259, npu_slow=1.487, keep_igpu=3.2 / 15.8, keep_npu=10.5 / 16.0,
        igpu_wait_s=0.527, npu_wait_s=0.110,
        npu_short_rows=640, npu_short_fixed_s=0.100, npu_short_block_s=0.459,
        source="bench/results/2026-11-25-packed-gemm.md"),
    # The same pair on the batched server engine (T-035/T-036: one shared context, mixed
    # prefill+decode steps, 256-token chunks). B TTFT under A, 85/401/1721 tokens: iGPU
    # 0.37-0.43 / 1.41 / 5.43 s, NPU 0.94-0.95 / 1.55-1.57 / 4.55-4.69 s (fits within 4%);
    # A (alone ~15.5 tok/s) during: iGPU 4.7-5.4 / 2.1 / 1.5, NPU 9.5-10.7 / 9.6 / 10.1.
    # On battery, platform balanced (2026-11-25-batched-decode.md).
    "qwen3-8b-packed-batched": DeviceCosts(
        igpu_prefill_tps=404.0, npu_fixed_s=0.510, npu_block_s=0.362, handoff_row_s=250e-3 / 1702,
        igpu_slow=1.240, npu_slow=1.488, keep_igpu=0.067, keep_npu=0.643,
        igpu_wait_s=0.157, npu_wait_s=0.082,
        npu_short_rows=640, npu_short_fixed_s=0.100, npu_short_block_s=0.459,
        source="bench/results/2026-11-25-batched-decode.md"),
}


def costs_for(name: str) -> DeviceCosts:
    """A copy of the measured preset ``name`` (``presets()``)."""
    return replace(_PRESETS[name])


def presets() -> list[str]:
    return sorted(_PRESETS)


@dataclass
class Choice:
    device: str  # "igpu", "npu", or "split" (NPU prefills npu_tokens, then the iGPU the rest)
    ttft_s: float  # estimated
    lost_decode_s: float  # estimated decode-seconds lost by the active decoders
    cost: float
    other: "Choice | None" = None  # the runner-up
    npu_tokens: int = 0  # tokens the NPU prefills (npu: all, split: whole blocks)
    explore: bool = False  # picked to learn its cost, not as the estimated best


@dataclass
class PlacementPolicy:
    costs: DeviceCosts
    decode_weight: float = 1.0
    ema: float = 0.3  # weight of a new observation in ``observe``
    npu_available: bool = True
    npu_max_tokens: int = 2048  # the NPU executor's max_seq_len
    log: list = field(default_factory=list)
    online: object | None = None  # sched.online.OnlineCosts: fitted parts replace the preset
    # Online exploration: an option within this fraction of the best cost whose main part was
    # (almost) never observed at this size is tried, so a pessimistic extrapolation cannot
    # lock it out (measured: the NPU at 2 blocks predicted 1.71 s, never tried, really ~1.35 s).
    explore_margin: float = 0.2
    explore_until: int = 2  # observations at a size after which an option is known

    @classmethod
    def calibrating(cls, costs: DeviceCosts, **kw) -> "PlacementPolicy":
        """A policy whose estimates start at ``costs`` and follow what ``observe_*`` measure."""
        from .online import OnlineCosts
        return cls(costs, online=OnlineCosts(costs), **kw)

    def _igpu_s(self, n: int, active: int) -> float:
        c = self.costs
        if n <= 0:
            return 0.0
        return c.igpu_alone_s(n) * c.igpu_slow + c.igpu_wait_s if active else c.igpu_alone_s(n)

    def estimate(self, device: str, n: int, active: int, npu_busy_s: float = 0.0,
                 npu_tokens: int | None = None) -> Choice:
        """Estimated TTFT and lost decode time of prefilling ``n`` tokens on ``device`` while
        ``active`` requests decode on the iGPU; ``npu_busy_s``: until the NPU is free.
        ``split``: the NPU prefills the first ``npu_tokens``, the iGPU the rest."""
        if self.online is not None:
            return self._estimate_online(device, n, active, npu_busy_s, npu_tokens)
        c = self.costs
        # Decoders lose speed only while a prefill computes: not during the fixed waits
        # (the iGPU prefill queued behind their decode steps; the handoff), nor while a
        # request waits for the NPU.
        if device == "igpu":
            t = self._igpu_s(n, active)
            work = c.igpu_alone_s(n) * (c.igpu_slow if active else 1.0)
            lost = active * work * (1.0 - c.keep_igpu)
            k = 0
        elif device in ("npu", "split"):
            k = n if device == "npu" else (npu_tokens if npu_tokens is not None
                                            else n // c.npu_block * c.npu_block)
            if active:
                npu_work = c.npu_alone_s(k) * c.npu_slow
                run = npu_work + c.npu_wait_s
            else:
                npu_work = 0.0
                run = c.npu_alone_s(k) + k * c.handoff_row_s
            tail = self._igpu_s(n - k, active)
            tail_work = c.igpu_alone_s(n - k) * (c.igpu_slow if active else 1.0)
            t = npu_busy_s + run + tail
            lost = active * (npu_work * (1.0 - c.keep_npu) + tail_work * (1.0 - c.keep_igpu))
        else:
            raise ValueError(device)
        return Choice(device, t, lost, t + self.decode_weight * lost, npu_tokens=k)

    def _estimate_online(self, device, n, active, npu_busy_s, npu_tokens) -> Choice:
        """Composed from fitted parts: NPU(k) + iGPU(n - k), k = 0 / n - 1 / whole blocks."""
        o, c = self.online, self.costs
        if device == "igpu":
            k = 0
        elif device == "npu":
            k = n - 1 if n > 1 else 0
        elif device == "split":
            k = npu_tokens if npu_tokens is not None else n // c.npu_block * c.npu_block
        else:
            raise ValueError(device)
        npu_t = o.npu_s(k, active)
        igpu_t = o.igpu_s(n - k, active)
        t = (npu_busy_s if k else 0.0) + npu_t + igpu_t
        # decode-seconds the others lose, fitted per part from measured request windows
        lost = active * (o.npu_lost_s(k) + o.igpu_lost_s(n - k))
        dev = device if not (device == "npu" and k == 0) else "igpu"
        return Choice(dev, t, lost, t + self.decode_weight * lost, npu_tokens=k)

    def choose(self, n: int, active: int, npu_busy_s: float = 0.0, n_npu: int | None = None,
               total: int | None = None) -> Choice:
        """Where to prefill a request with ``active`` requests decoding on the iGPU.

        ``n``: tokens the iGPU would evaluate (after its prefix-cache hit); ``n_npu``: tokens
        the NPU would evaluate (after the prefix it holds; default ``n``) — KV moves only
        NPU -> iGPU, so an iGPU cache hit does not shorten an NPU prefill; ``total``: the
        prompt's length (default ``n_npu``), which must fit the NPU's context."""
        n_npu = n if n_npu is None else n_npu
        total = n_npu if total is None else total
        g = self.estimate("igpu", n, active)
        if not self.npu_available or n_npu < 1 or total > self.npu_max_tokens:
            return g
        opts = [g, self.estimate("npu", n_npu, active, npu_busy_s)]
        k = n_npu // self.costs.npu_block * self.costs.npu_block
        if 0 < k < n_npu:
            opts.append(self.estimate("split", n_npu, active, npu_busy_s, npu_tokens=k))
        opts.sort(key=lambda o: o.cost)
        best = opts[0]
        if self.online is not None and active:
            for o in opts[1:]:
                if o.cost > best.cost * (1 + self.explore_margin):
                    break
                part = ("npu", o.npu_tokens) if o.npu_tokens else ("igpu", n)
                if self.online.seen(*part) < self.explore_until:
                    o.explore = True
                    o.other = best
                    return o
        best.other = opts[1]
        return best

    def observe_part(self, device: str, n: int, seconds: float, active: int) -> None:
        """A measured prefill part (online calibration): ``device`` "igpu" for n tokens the
        iGPU evaluated, "npu" for n tokens the NPU prefilled and handed over."""
        if self.online is None:
            return self.observe(device, n, seconds, active)
        if device == "igpu":
            self.online.observe_igpu(n, seconds, active)
        else:
            self.online.observe_npu(n, seconds, active)
        self.log.append((device, n, seconds, active))

    def observe(self, device: str, n: int, seconds: float, active: int) -> None:
        """Refine the device model from a measured prefill of ``n`` tokens (device time
        only, without NPU queueing or handoff), taken with ``active`` decoders running."""
        c = self.costs
        a = self.ema
        if device == "igpu":
            alone = (seconds - c.igpu_wait_s) / c.igpu_slow if active else seconds
            c.igpu_prefill_tps = (1 - a) * c.igpu_prefill_tps + a * (n / alone)
        else:
            alone = (seconds - c.npu_wait_s) / c.npu_slow if active else seconds
            # scale the NPU terms of the version that ran by the observed/predicted ratio
            r = alone / c.npu_alone_s(n)
            f = (1 - a) + a * r
            if n <= c.npu_short_rows:
                c.npu_short_fixed_s *= f
                c.npu_short_block_s *= f
            else:
                c.npu_fixed_s *= f
                c.npu_block_s *= f
        self.log.append((device, n, seconds, active))
