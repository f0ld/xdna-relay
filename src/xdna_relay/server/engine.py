"""The serving engine: slots on the iGPU, one NPU prefill worker, the prefix cache.

Each slot is a llama.cpp context (``LlamaCppExecutor``) that holds one sequence, plus an
optional draft context for speculative decoding. A request runs in its caller's thread:

1. take the free slot whose live sequence shares the longest prefix with the prompt;
2. ask ``PlacementPolicy`` where to prefill, given the slots decoding right now, what the
   slot / prefix cache / NPU worker already hold, and when the NPU is free;
3. NPU (or split): the worker prefills, the KV goes into the slot (``worker_to_llama``);
4. ``generate`` prefills the rest (reusing what the slot holds), then decodes, streaming
   tokens through ``on_token``; the slot's state goes into the prefix cache afterwards;
5. the measured prefill parts refine the policy; with online calibration (``sched.online``)
   the stepper's step times also measure how much a prefill slows the decoders.

The NPU worker is one process (IRON holds the GIL; T-017), used by one request at a time.

Batched mode (default without a draft model; T-035): the slots are sequences of one
``SharedLlamaContext`` (the model loaded once), and one stepper thread decodes every decoding
slot in a single ``llama_decode`` per step (decode is weight-bandwidth bound: 2 sequences
decode at ~15 tok/s each instead of 8). Request threads prefill their own sequence (the
context lock is released between 512-token chunks, so steps interleave), then take their
tokens from a queue the stepper fills. With a draft model, each slot keeps its own contexts
and decodes speculatively (``generate``), unbatched.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

from xdna_relay.sched.placement import PlacementPolicy
from xdna_relay.sched.prefix_cache import PrefixCache, common_prefix, prepare
from xdna_relay.sched.speculative import GenStats, generate


@dataclass
class _Decoding:
    """A request in the stepper's hands: prefilling (``pending`` tokens left), then decoding."""
    slot: "Slot"
    last: int | None
    remaining: int
    out: queue.Queue
    pending: list = field(default_factory=list)
    cancel: bool = False
    t_first: float = 0.0


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
    keep_observed: float | None = None  # others' decode speed kept during this prefill
    explore: bool = False  # placement chosen to learn its cost


class Engine:
    def __init__(self, gguf: str, *, slots: int = 2, n_ctx: int = 8192, draft_gguf: str | None = None,
                 hf_dir: str | None = None, npu: bool = False, packed: bool = False,
                 npu_max_seq_len: int = 2048, policy: PlacementPolicy | None = None,
                 placement: str = "auto", cache_bytes: int = 4 << 30, batched: bool | None = None,
                 prefill_chunk: int = 256, calibration_path: str | None = None):
        from xdna_relay.backends.igpu.llama import LlamaCppExecutor

        if placement not in ("auto", "igpu", "npu", "split"):
            raise ValueError(placement)
        self.batched = draft_gguf is None if batched is None else batched
        if self.batched and draft_gguf:
            raise ValueError("batched decode does not run a draft model (speculative is per slot)")
        self.shared = None
        if self.batched:
            from xdna_relay.backends.igpu.llama_multi import SharedLlamaContext
            self.shared = SharedLlamaContext(gguf, n_seq=slots, n_ctx=n_ctx * slots)
            self.slots = [Slot(i, self.shared.seqs[i]) for i in range(slots)]
        else:
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
            # no measured preset: a rough prior, calibrated online from what this engine measures
            policy = PlacementPolicy.calibrating(DeviceCosts(400.0, 0.02, 0.6, 1e-4, 1.25, 1.25, 0.2, 0.5))
        policy.npu_available = self.npu is not None
        policy.npu_max_tokens = npu_max_seq_len
        self.policy = policy
        self.placement = placement
        self.calibration = None  # sched.calibration_store.CalibrationStore, when persisted
        if calibration_path and policy.online is not None:
            from xdna_relay.sched.calibration_store import CalibrationStore
            self.calibration = CalibrationStore(policy.online, calibration_path)
            self.calibration.load()
        self._cv = threading.Condition()
        self._npu_lock = threading.Lock()
        self._npu_free_at = 0.0
        self.requests = 0
        self.prefill_chunk = prefill_chunk  # prompt tokens per mixed step (batched mode)
        self.steps = 0  # batched decode steps
        self._step_ends: list[float] = []  # end times of recent steps that advanced decoders
        self.step_rows = 0  # sequences advanced, summed over steps
        self._decoding: list[_Decoding] = []
        self._step_cv = threading.Condition()
        self._closing = False
        if self.batched:
            threading.Thread(target=self._stepper, name="decode-stepper", daemon=True).start()

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
        elif (self.placement == "split" and self.npu is not None and len(prompt) <= self.policy.npu_max_tokens
              and (k := limit // self.policy.costs.npu_block * self.policy.costs.npu_block) > 0):
            choice = self.policy.estimate("split", limit, active, busy, npu_tokens=k)
        return choice, active, n_igpu, n_npu

    def complete(self, prompt: list[int], max_tokens: int = 256, on_token=None,
                 t_arrival: float | None = None) -> Result:
        """Generate greedily after ``prompt`` (token ids); ``on_token(t) -> bool`` streams each
        token and returns True to stop."""
        if len(prompt) < 1:
            raise ValueError("empty prompt")
        from xdna_relay.kv.handoff import worker_to_llama

        res = Result()
        # TTFT and the others' loss count from the request's arrival (HTTP parsing, chat
        # template and tokenizing run in Python, under the GIL, beside the stepper)
        t0 = t_arrival if t_arrival is not None else time.perf_counter()
        slot = self._take_slot(prompt)
        res.slot = slot.index
        self.requests += 1
        try:
            choice, active, n_igpu, n_npu = self._place(slot, prompt)
            res.placement, res.estimate_ttft_s = choice.device, choice.ttft_s
            res.explore = choice.explore
            if choice.device in ("npu", "split"):
                k = len(prompt) - 1 if choice.device == "npu" else choice.npu_tokens
                res.npu_tokens = k
                with self._npu_lock:
                    self._npu_free_at = time.perf_counter() + choice.ttft_s
                    t_h = time.perf_counter()
                    worker_to_llama(self.npu, slot.target, prompt[:k])
                    t_h = time.perf_counter() - t_h
                if self.policy.online is not None:
                    # the NPU part as the cost model composes it: eval + pack + handoff
                    n_part = (n_npu if choice.device == "npu" else k)
                    self.policy.observe_part("npu", n_part, t_h, active)
                else:
                    t = self.npu.last_timing
                    self.policy.observe("npu", n_npu if choice.device == "npu" else k,
                                        t.get("eval", 0.0), active)

            if self.batched:
                self._complete_batched(slot, prompt, max_tokens, on_token, res, t0, active, choice)
            else:
                self._complete_slot(slot, prompt, max_tokens, on_token, res, t0, active, choice)
            return res
        finally:
            self._release(slot)
            if self.calibration is not None:
                self.calibration.maybe_save()

    def _complete_slot(self, slot, prompt, max_tokens, on_token, res, t0, active, choice):
        """Unbatched: the slot's own contexts, ``generate`` (speculative with a draft)."""
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
        self._finish_prefill_stats(res, st.prefill, choice, active)

    def _complete_batched(self, slot, prompt, max_tokens, on_token, res, t0, active, choice):
        """Batched: restore what can be reused here, then the stepper prefills the rest in
        chunks riding along with the other sequences' decode tokens, and decodes."""
        st = GenStats()
        n0, info = prepare(slot.target, prompt, self.cache)
        info.evaluated = len(prompt) - n0
        entry = _Decoding(slot, None, max_tokens, queue.Queue(), pending=list(prompt[n0:]))
        t_sub = time.perf_counter()
        with self._step_cv:
            self._decoding.append(entry)
            self._step_cv.notify()
        stop = False
        while True:
            t = entry.out.get()
            if t is None:
                break
            if entry.cancel:  # the stepper may run one step past a stop: drop it
                continue
            if not st.tokens:
                res.ttft_s = entry.t_first - t0
                info.t_eval = entry.t_first - t_sub
                st.t_prefill = entry.t_first - t0
                with self._cv:
                    slot.phase = "decode"
            st.tokens.append(t)
            if on_token is not None and on_token(t):
                entry.cancel = stop = True
        st.prefill = info
        res.stopped = stop
        st.t_total = time.perf_counter() - t0
        res.stats, res.tokens = st, list(st.tokens)
        res.total_s = st.t_total
        self.cache.put(slot.target.snapshot())
        self._finish_prefill_stats(res, info, choice, active)
        if active and self.policy.online is not None and st.tokens:
            # how much the others kept while this request was prefilled (arrival -> first token)
            t_first = t0 + res.ttft_s
            steps = sum(1 for t in list(self._step_ends) if t0 <= t <= t_first)
            n_dev = {"npu": res.npu_tokens, "split": len(prompt)}.get(choice.device, info.evaluated)
            res.keep_observed = self.policy.online.observe_window(choice.device, res.ttft_s, steps, n_dev)

    def _stepper(self) -> None:
        """Each step, one llama_decode: every decoding sequence's next token, plus the next
        chunk (``prefill_chunk`` tokens) of the oldest request still prefilling."""
        while True:
            with self._step_cv:
                while not self._decoding and not self._closing:
                    self._step_cv.wait()
                if self._closing:
                    for e in self._decoding:
                        e.out.put(None)
                    return
                entries = list(self._decoding)
            dec = [e for e in entries if not e.pending and not e.cancel]
            pre = [e for e in entries if e.pending and not e.cancel]
            parts, owners = [(e.slot.target, [e.last], True) for e in dec], list(dec)
            if pre:
                job = pre[0]
                budget = min(self.prefill_chunk, self.shared.n_batch - len(dec))
                chunk = job.pending[:budget]
                parts.append((job.slot.target, chunk, len(chunk) == len(job.pending)))
                owners.append(job)
            finished = [e for e in entries if e.cancel]
            if parts:
                npu_busy = self._npu_lock.locked()
                t_step = time.perf_counter()
                try:
                    out = self.shared.step(parts)
                except Exception:
                    with self._step_cv:
                        for e in entries:
                            if e in self._decoding:
                                self._decoding.remove(e)
                            e.out.put(None)
                    raise
                self.steps += 1
                self.step_rows += len(dec)
                if dec:
                    self._step_ends.append(time.perf_counter())
                    if len(self._step_ends) > 4096:
                        del self._step_ends[:2048]
                if dec and self.policy.online is not None:
                    self.policy.online.observe_step(time.perf_counter() - t_step,
                                                    prefill_chunk=bool(pre), npu_busy=npu_busy)
                now = time.perf_counter()
                for e, (_, toks, _), r in zip(owners, parts, out):
                    if e.pending:  # the prefill chunk
                        del e.pending[:len(toks)]
                        if e.pending:
                            continue
                        e.t_first = now
                    t = r[0]
                    e.last = t
                    e.remaining -= 1
                    e.out.put(t)
                    if e.remaining <= 0 or e.slot.target.is_eog(t):
                        finished.append(e)
            if finished:
                with self._step_cv:
                    for e in finished:
                        if e in self._decoding:
                            self._decoding.remove(e)
                        e.out.put(None)

    def _finish_prefill_stats(self, res, info, choice, active) -> None:
        if info is not None:
            res.reused = info.reused_live + info.reused_cache
            res.evaluated = info.evaluated
            if self.policy.online is not None:
                # the iGPU part: whatever it evaluated (all of it, a split's tail, or the
                # last token after a handoff), submit to first token
                if info.evaluated > 0 and info.t_eval > 0:
                    self.policy.observe_part("igpu", info.evaluated, info.t_eval, active)
            elif choice.device == "igpu" and info.evaluated > 1:
                self.policy.observe("igpu", info.evaluated, info.t_eval, active)

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
        if self.calibration is not None:
            try:
                self.calibration.save()
            except OSError as e:
                print(f"calibration: could not save on close: {e}")
        with self._step_cv:
            self._closing = True
            self._step_cv.notify()
        if self.npu is not None:
            self.npu.close()
        for s in self.slots:
            for ex in (s.target, s.draft):
                if ex is not None:
                    ex.close()
        if self.shared is not None:
            self.shared.close()
