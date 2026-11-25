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

    def npu_alone_s(self, n: int) -> float:
        return self.npu_fixed_s + self.npu_block_s * math.ceil(n / self.npu_block)

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


@dataclass
class PlacementPolicy:
    costs: DeviceCosts
    decode_weight: float = 1.0
    ema: float = 0.3  # weight of a new observation in ``observe``
    npu_available: bool = True
    npu_max_tokens: int = 2048  # the NPU executor's max_seq_len
    log: list = field(default_factory=list)

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
        c = self.costs
        lost = 0.0
        if device == "igpu":
            t = self._igpu_s(n, active)
            lost = active * t * (1.0 - c.keep_igpu)
            k = 0
        elif device in ("npu", "split"):
            k = n if device == "npu" else (npu_tokens if npu_tokens is not None
                                            else n // c.npu_block * c.npu_block)
            if active:
                run = c.npu_alone_s(k) * c.npu_slow + c.npu_wait_s
            else:
                run = c.npu_alone_s(k) + k * c.handoff_row_s
            tail = self._igpu_s(n - k, active)
            t = npu_busy_s + run + tail
            # decoders slow only while a prefill runs (not while it waits for the NPU)
            lost = active * (run * (1.0 - c.keep_npu) + tail * (1.0 - c.keep_igpu))
        else:
            raise ValueError(device)
        return Choice(device, t, lost, t + self.decode_weight * lost, npu_tokens=k)

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
        opts[0].other = opts[1]
        return opts[0]

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
            # scale both NPU terms by the observed/predicted ratio (keeps the fitted shape)
            r = alone / c.npu_alone_s(n)
            f = (1 - a) + a * r
            c.npu_fixed_s *= f
            c.npu_block_s *= f
        self.log.append((device, n, seconds, active))
