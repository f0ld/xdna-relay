"""The serving engine: slots on the iGPU, one NPU prefill worker, the prefix cache.

Each slot is a llama.cpp context (``LlamaCppExecutor``) that holds one sequence, plus an
optional draft context for speculative decoding. A request runs in its caller's thread:

1. take the free slot whose live sequence shares the longest prefix with the prompt;
2. ask ``PlacementPolicy`` where to prefill, given the slots decoding right now, what the
   slot / prefix cache / NPU worker already hold, and when the NPU is free;
3. NPU (or split): the worker prefills, the KV goes into the slot (``worker_to_llama``);
4. ``generate`` prefills the rest (reusing what the slot holds), then decodes, streaming
   tokens through ``on_token``; the slot's state goes into the prefix cache afterwards;
5. the measured prefill time refines the policy (``observe``).

The NPU worker is one process (IRON holds the GIL; T-017), used by one request at a time.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from xdna_relay.sched.placement import PlacementPolicy
from xdna_relay.sched.prefix_cache import PrefixCache, common_prefix
from xdna_relay.sched.speculative import GenStats, generate


@dataclass
class Slot:
    index: int
    target: object  # LlamaCppExecutor
    draft: object | None = None
    phase: str = "idle"  # idle | prefill | decode
    busy: bool = False


@dataclass
class Result:
    tokens: list[int] = field(default_factory=list)
    stats: GenStats | None = None
    slot: int = -1
    placement: str = "igpu"
    npu_tokens: int = 0
    estimate_ttft_s: float = 0.0
    ttft_s: float = 0.0
    total_s: float = 0.0
    reused: int = 0
    evaluated: int = 0
    stopped: bool = False


class Engine:
    def __init__(self, gguf: str, *, slots: int = 2, n_ctx: int = 8192, draft_gguf: str | None = None,
                 hf_dir: str | None = None, npu: bool = False, packed: bool = False,
                 npu_max_seq_len: int = 2048, policy: PlacementPolicy | None = None,
                 placement: str = "auto", cache_bytes: int = 4 << 30):
        from xdna_relay.backends.igpu.llama import LlamaCppExecutor

        if placement not in ("auto", "igpu", "npu"):
            raise ValueError(placement)
        self.slots = [Slot(i, LlamaCppExecutor(gguf, n_ctx=n_ctx),
                           LlamaCppExecutor(draft_gguf, n_ctx=n_ctx) if draft_gguf else None)
                      for i in range(slots)]
        self.model = self.slots[0].target.name
        self.cache = PrefixCache(max_bytes=cache_bytes)
        self.npu = None
        if npu:
            if hf_dir is None:
                raise ValueError("NPU prefill needs the Hugging Face checkpoint (hf_dir)")
            from xdna_relay.backends.npu.worker import NpuWorker
            self.npu = NpuWorker(hf_dir, max_seq_len=npu_max_seq_len, packed=packed)
            # warm the worker and the handoff template once
            from xdna_relay.kv.handoff import worker_to_llama
            s = self.slots[0]
            warm = s.target.tokenize("Hello there, this warms the NPU worker.", add_special=False)
            worker_to_llama(self.npu, s.target, warm)
            s.target.reset()
        if policy is None:
            from xdna_relay.sched.placement import DeviceCosts
            policy = PlacementPolicy(DeviceCosts(400.0, 0.02, 0.6, 1e-4, 1.25, 1.25, 0.2, 0.5))
        policy.npu_available = self.npu is not None
        policy.npu_max_tokens = npu_max_seq_len
        self.policy = policy
        self.placement = placement
        self._cv = threading.Condition()
        self._npu_lock = threading.Lock()
        self._npu_free_at = 0.0
        self.requests = 0

    # ---- slots ---------------------------------------------------------------------------

    def _take_slot(self, prompt: list[int]) -> Slot:
        with self._cv:
            while all(s.busy for s in self.slots):
                self._cv.wait()
            free = [s for s in self.slots if not s.busy]
            slot = max(free, key=lambda s: common_prefix(s.target.held, prompt))
            slot.busy, slot.phase = True, "prefill"
            return slot

    def _release(self, slot: Slot) -> None:
        with self._cv:
            slot.busy, slot.phase = False, "idle"
            self._cv.notify()

    # ---- requests ------------------------------------------------------------------------

    def _place(self, slot: Slot, prompt: list[int]):
        with self._cv:
            active = sum(1 for s in self.slots if s is not slot and s.phase == "decode")
        limit = len(prompt) - 1
        reuse = min(max(common_prefix(slot.target.held, prompt),
                        self.cache.peek(slot.target.name, prompt)), limit)
        n_igpu = len(prompt) - reuse
        n_npu = 0
        if self.npu is not None:
            n_npu = limit - min(common_prefix(self.npu.held, prompt[:limit]), limit - 1)
        busy = max(0.0, self._npu_free_at - time.perf_counter()) if self._npu_lock.locked() else 0.0
        choice = self.policy.choose(n_igpu, active, npu_busy_s=busy, n_npu=n_npu, total=len(prompt))
        if self.placement == "igpu" or (self.placement == "npu" and self.npu is None):
            choice = self.policy.estimate("igpu", n_igpu, active)
        elif self.placement == "npu" and n_npu > 0 and len(prompt) <= self.policy.npu_max_tokens:
            choice = self.policy.estimate("npu", n_npu, active, busy)
        return choice, active, n_igpu, n_npu

    def complete(self, prompt: list[int], max_tokens: int = 256, on_token=None) -> Result:
        """Generate greedily after ``prompt`` (token ids); ``on_token(t) -> bool`` streams each
        token and returns True to stop."""
        if len(prompt) < 1:
            raise ValueError("empty prompt")
        from xdna_relay.kv.handoff import worker_to_llama

        res = Result()
        t0 = time.perf_counter()
        slot = self._take_slot(prompt)
        res.slot = slot.index
        self.requests += 1
        try:
            choice, active, n_igpu, n_npu = self._place(slot, prompt)
            res.placement, res.estimate_ttft_s = choice.device, choice.ttft_s
            if choice.device in ("npu", "split"):
                k = len(prompt) - 1 if choice.device == "npu" else choice.npu_tokens
                res.npu_tokens = k
                with self._npu_lock:
                    self._npu_free_at = time.perf_counter() + choice.ttft_s
                    worker_to_llama(self.npu, slot.target, prompt[:k])
                t = self.npu.last_timing
                self.policy.observe("npu", n_npu if choice.device == "npu" else k,
                                    t.get("eval", 0.0), active)

            first = []

            def cb(tok: int) -> bool:
                if not first:
                    first.append(time.perf_counter())
                    with self._cv:
                        slot.phase = "decode"
                stop = bool(on_token(tok)) if on_token is not None else False
                res.stopped = res.stopped or stop
                return stop

            st = generate(slot.target, prompt, max_tokens, draft=slot.draft, cache=self.cache, on_token=cb)
            res.stats, res.tokens = st, list(st.tokens)
            res.ttft_s = (first[0] if first else time.perf_counter()) - t0
            res.total_s = time.perf_counter() - t0
            if st.prefill is not None:
                res.reused = st.prefill.reused_live + st.prefill.reused_cache
                res.evaluated = st.prefill.evaluated
                if choice.device == "igpu" and st.prefill.evaluated > 1:
                    self.policy.observe("igpu", st.prefill.evaluated, st.prefill.t_eval, active)
            return res
        finally:
            self._release(slot)

    def reset_state(self) -> None:
        """Forget every slot's sequence and the prefix cache (tests; a cold start)."""
        with self._cv:
            for s in self.slots:
                for ex in (s.target, s.draft):
                    if ex is not None:
                        ex.reset()
            self.cache = PrefixCache(max_bytes=self.cache.max_bytes)

    def tokenize(self, text: str) -> list[int]:
        return self.slots[0].target.tokenize(text, add_special=False)

    def detokenize(self, tokens: list[int]) -> str:
        return self.slots[0].target.detokenize(tokens)

    def is_eog(self, token: int) -> bool:
        return self.slots[0].target.is_eog(token)

    def close(self) -> None:
        if self.npu is not None:
            self.npu.close()
        for s in self.slots:
            for ex in (s.target, s.draft):
                if ex is not None:
                    ex.close()
