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

Batched mode (default; T-035): the slots are sequences of one
``SharedLlamaContext`` (the model loaded once), and one stepper thread decodes every decoding
slot in a single ``llama_decode`` per step (decode is weight-bandwidth bound: 2 sequences
decode at ~15 tok/s each instead of 8). Request threads prefill their own sequence (the
context lock is released between 512-token chunks, so steps interleave), then take their
tokens from a queue the stepper fills.

With a draft model (T-041), the draft is a second shared context (one sequence per slot) and
every stepper round is speculative: ``BatchK`` picks a draft length per decoding sequence
(a common k, or 0 for sequences not worth drafting), the draft runs k batched steps, one
target batch verifies every sequence's [last, d1..dk] together with any prefill chunk, and
each sequence keeps its accepted drafts plus the correction. A draft that lags (after an NPU
prefill or a restore) catches up in chunks before its sequence drafts again. The target keeps
one KV buffer per sequence (faster attention, deterministic per sequence), so llama.cpp runs
a verify batch with uneven token counts as one pass per distinct count; ``BatchK`` costs that
(``split_ragged``) and prefers even shapes. The draft context uses one unified buffer. Unbatched mode
(``batched=False``) keeps one context pair per slot and runs ``generate`` per request.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

from xdna_relay.sched.placement import PlacementPolicy
from xdna_relay.sched.prefix_cache import PrefixCache, common_prefix, prepare
from xdna_relay.sched.speculative import BatchK, GenStats, generate

DRAFT_CATCH_UP = 256  # draft catch-up tokens per sequence per round
DRAFT_MAX_LAG = 32  # a decoder whose draft misses more tokens catches up instead of drafting


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
    seq: list = field(default_factory=list)  # speculative: prompt + generated tokens
    acc: object = None  # speculative: this sequence's Acceptance
    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    ks: list = field(default_factory=list)


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
                 prefill_chunk: int = 256, calibration_path: str | None = None,
                 spec_k: int | None = None):
        from xdna_relay.backends.igpu.llama import LlamaCppExecutor

        if placement not in ("auto", "igpu", "npu", "split"):
            raise ValueError(placement)
        self.batched = True if batched is None else batched
        self.shared = self.dshared = self.spec = None
        if self.batched:
            from xdna_relay.backends.igpu.llama_multi import SharedLlamaContext
            # hybrid models: recurrent-state snapshots to reject up to k_max drafts (ignored by
            # llama.cpp for architectures without recurrent rollback)
            self.shared = SharedLlamaContext(gguf, n_seq=slots, n_ctx=n_ctx * slots,
                                             n_rs_seq=BatchK.k_max if draft_gguf else 0)
            if draft_gguf:
                # the draft's batches are uneven every round (catch-up of 1 or 2 tokens, chunks):
                # one KV buffer, so llama.cpp runs them in one pass (T-041)
                self.dshared = SharedLlamaContext(draft_gguf, n_seq=slots, n_ctx=n_ctx * slots, kv_unified=True)
                if self.dshared.n_vocab != self.shared.n_vocab:
                    raise ValueError(f"draft vocabulary {self.dshared.n_vocab} != target {self.shared.n_vocab}")
                self.spec = BatchK(fixed=spec_k, split_ragged=True)
                if self.shared.rollback is not None:
                    self.spec.k_max = min(self.spec.k_max, self.shared.rollback)
            self.slots = [Slot(i, self.shared.seqs[i], self.dshared.seqs[i] if self.dshared else None)
                          for i in range(slots)]
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
        if self.spec is not None:
            self._sweep_spec_costs()
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
        exact = getattr(slot.target, "rollback", None) is not None
        live = common_prefix(slot.target.held, prompt)
        if exact and live < len(slot.target.held):
            live = 0
        reuse = min(max(live, self.cache.peek(slot.target.name, prompt, exact=exact)), limit)
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
        if self.spec is not None:
            prepare(slot.draft, prompt, self.cache)  # the rest catches up in the stepper
            entry.seq, entry.acc = list(prompt), self.spec.new_sequence()
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
        st.rounds, st.drafted, st.accepted, st.ks = entry.rounds, entry.drafted, entry.accepted, entry.ks
        res.stopped = stop
        st.t_total = time.perf_counter() - t0
        res.stats, res.tokens = st, list(st.tokens)
        res.total_s = st.t_total
        self.cache.put(slot.target.snapshot())
        if self.spec is not None:
            self.cache.put(slot.draft.snapshot())
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
            if self.spec is not None:
                try:
                    finished = self._round_spec(entries)
                except Exception:
                    with self._step_cv:
                        for e in entries:
                            if e in self._decoding:
                                self._decoding.remove(e)
                            e.out.put(None)
                    raise
                if finished:
                    with self._step_cv:
                        for e in finished:
                            if e in self._decoding:
                                self._decoding.remove(e)
                            e.out.put(None)
                continue
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
                        self._snapshot_prompt(e)
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

    def _snapshot_prompt(self, e: _Decoding) -> None:
        """Hybrid targets cannot truncate back to the prompt later; a next chat turn extends
        the prompt (its history may rewrite this answer), so keep a snapshot of it now
        (~40 ms for the 35B-A3B; the target holds exactly the prompt here)."""
        if getattr(e.slot.target, "rollback", None) is None:
            return
        snap = e.slot.target.snapshot()
        if len(snap.tokens) >= self.cache.min_tokens:
            self.cache.put(snap)

    def _sweep_spec_costs(self, reps: int = 3) -> None:
        """Seed ``BatchK`` with measured batch costs of both models. Costs are not smooth in
        the batch size (Qwen3-8B Q4_K_M on the 890M: ~75 ms up to 4 tokens, ~100 ms at 8,
        ~190 ms from 9: llama.cpp's Vulkan matrix-vector path covers up to 8 columns), so
        interpolating between a few online measurements drafts across the cliff (T-041)."""
        k_max = self.spec.k_max
        top = len(self.slots) * (k_max + 1)
        counts = sorted(set(range(1, 13)) | {16, 24, 32, top} | {64, 128, self.prefill_chunk,
                                                                   self.prefill_chunk + 8})
        filler = (self.shared.tokenize("The quick brown fox jumps over the lazy dog. ", False) * 64)
        for ctx, d in ((self.shared, self.spec.verify), (self.dshared, self.spec.draft)):
            ex = ctx.seqs[0]
            ex.reset()
            ex.eval(filler[:32])
            ctx.step([(ex, filler[32:33], True)])  # warm
            ex.truncate(32)
            for n in counts:
                if n > ctx.n_batch:
                    continue
                best = None
                for _ in range(reps):
                    t = time.perf_counter()
                    ctx.step([(ex, filler[32:32 + n], "all" if n <= top else True)])
                    best = min(best or 1e9, time.perf_counter() - t)
                    ex.truncate(32)
                d[n] = best
            ex.reset()

    def _round_spec(self, entries: list[_Decoding]) -> list[_Decoding]:
        """One speculative round (see the module doc); returns the entries that finished."""
        finished = [e for e in entries if e.cancel]
        dec = [e for e in entries if not e.pending and not e.cancel]
        pre = [e for e in entries if e.pending and not e.cancel]
        job = pre[0] if pre else None
        if not dec and job is None:
            return finished
        drafting = [e for e in dec if len(e.seq) - e.slot.draft.n_past <= DRAFT_MAX_LAG]
        lagging = [e for e in dec if e not in drafting]
        # draft catch-up without logits, up to seq[:-1]: lagging decoders, the prefilling request
        dpre, room = [], self.dshared.n_batch - sum(len(e.seq) - e.slot.draft.n_past for e in drafting)
        for e in lagging + ([job] if job is not None else []):
            t = e.seq[e.slot.draft.n_past:len(e.seq) - 1][:min(DRAFT_CATCH_UP, room)]
            if t:  # logits wanted only to make the call wait for the GPU: without an output,
                # llama_decode returns early and its work lands in the next call's time
                dpre.append((e.slot.draft, t, True))
                room -= len(t)
        missing = [len(e.seq) - e.slot.draft.n_past for e in drafting]
        chunk_n = min(self.prefill_chunk, len(job.pending)) if job is not None else 0
        ks = self.spec.choose([e.acc for e in drafting], missing, n_plain=len(lagging),
                              extra_draft=sum(len(t) for _, t, _ in dpre), extra_verify=chunk_n)
        ks = [max(0, min(k, e.remaining - 1)) for k, e in zip(ks, drafting)]

        npu_busy = self._npu_lock.locked()
        t0 = time.perf_counter()
        drafts = [[] for _ in drafting]
        owners = [i for i, k in enumerate(ks) if k > 0]
        parts = [(drafting[i].slot.draft, drafting[i].seq[drafting[i].slot.draft.n_past:], True)
                 for i in owners] + dpre
        first_tokens = sum(len(t) for _, t, _ in parts)
        if parts:
            for i, r in zip(owners, self.dshared.step(parts)):
                drafts[i].append(r[0])
        t1 = time.perf_counter()
        for j in range(2, max(ks, default=0) + 1):
            idx = [i for i, k in enumerate(ks) if k >= j]
            for i, r in zip(idx, self.dshared.step([(drafting[i].slot.draft, [drafts[i][-1]], True) for i in idx])):
                drafts[i].append(r[0])
        t2 = time.perf_counter()

        pos = {id(e): i for i, e in enumerate(drafting)}
        vparts = [(e.slot.target, [e.seq[-1]] + (drafts[pos[id(e)]] if id(e) in pos else []), "all") for e in dec]
        n_verify = sum(len(t) for _, t, _ in vparts)
        chunk = []
        if job is not None:
            chunk = job.pending[:min(self.prefill_chunk, self.shared.n_batch - n_verify)]
            if chunk:
                vparts.append((job.slot.target, chunk, len(chunk) == len(job.pending)))
        out = self.shared.step(vparts)
        t3 = time.perf_counter()

        accepted = [0] * len(drafting)
        emitted = 0
        for e, rows in zip(dec, out):
            d = drafts[pos[id(e)]] if id(e) in pos else []
            base = len(e.seq) - 1  # the target held seq[:-1]
            j = 0
            while j < len(d) and d[j] == rows[j][0]:
                j += 1
            e.slot.target.truncate(base + 1 + j)
            if e.slot.draft.n_past > base + 1 + j:
                e.slot.draft.truncate(base + 1 + j)
            if id(e) in pos:
                accepted[pos[id(e)]] = j
            e.rounds += 1
            e.drafted += len(d)
            e.accepted += j
            e.ks.append(len(d))
            for t in d[:j] + [rows[j][0]]:
                emitted += 1
                e.seq.append(t)
                e.remaining -= 1
                e.out.put(t)
                if e.remaining <= 0 or e.slot.target.is_eog(t):
                    finished.append(e)
                    break
        if chunk:
            del job.pending[:len(chunk)]
            if not job.pending:
                t = out[-1][0]
                job.t_first = time.perf_counter()
                self._snapshot_prompt(job)
                job.seq.append(t)
                job.remaining -= 1
                job.out.put(t)
                if job.remaining <= 0 or job.slot.target.is_eog(t):
                    finished.append(job)

        self.spec.observe(ks, accepted, [e.acc for e in drafting], t_first=t1 - t0, first_tokens=first_tokens,
                          t_rest=t2 - t1, t_verify=t3 - t2, verify_shape=[len(t) for _, t, _ in vparts])
        if dec and not chunk:
            self.spec.observe_round(len(dec), any(ks), emitted, time.perf_counter() - t0)
        self.steps += 1
        self.step_rows += len(dec)
        if dec:
            self._step_ends.append(time.perf_counter())
            if len(self._step_ends) > 4096:
                del self._step_ends[:2048]
            if self.policy.online is not None:
                self.policy.online.observe_step(t3 - t0, prefill_chunk=bool(chunk), npu_busy=npu_busy)
        return finished

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
        if self.dshared is not None:
            self.dshared.close()
