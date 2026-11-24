"""Greedy speculative decoding over two executors, with an adaptive draft length (T-030).

Invariant at the start of every round: ``seq`` = prompt + generated tokens; the target holds
``seq[:-1]`` (its last token is not evaluated yet); the draft holds a prefix of ``seq[:-1]``
and catches up at the start of its drafting.

Round with draft length k (k >= 1):
  draft:  feed the tokens it is missing (ends with seq[-1]) -> d1; feed d1 -> d2; ... -> dk
  verify: target.eval([seq[-1], d1..dk], all rows) -> y0..yk
  accept: j = longest prefix with d_{i+1} == y_i; append d1..dj and y_j (correction/bonus)
  roll back target to hold seq[:-1] + [seq[-1], d1..dj], draft to at most that.
k = 0 is a plain decode step (target.eval([seq[-1]])).

The policy maximises expected tokens per second: E(k) = (1 - a^(k+1)) / (1 - a) tokens per
round, cost = k * t_draft + t_verify(k), with a (per-position acceptance), t_draft and
t_verify(k) measured online.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .executor import Executor
from .prefix_cache import PrefillInfo, PrefixCache, prefill


@dataclass
class AdaptiveK:
    """Choose the draft length per round from measured acceptance and costs.

    Cost of a round with k >= 1: t_first (first draft call, includes the draft's catch-up)
    + (k - 1) * t_step + t_verify(k). Tokens per round: 1 + sum_i prod_{m<=i} p_m with
    p_m = P(position m accepted | m-1 accepted), each an EMA.
    """

    k_max: int = 8
    k_init: int = 3
    ema: float = 0.15  # weight of the newest observation
    p_prior: float = 0.6
    t_first: float | None = None  # seconds: first draft call of a round
    t_step: float | None = None  # seconds per further draft step
    t_verify: dict[int, float] = field(default_factory=dict)  # k -> seconds (k=0: plain step)
    verify_slope: float = 0.03  # unseen k: t_verify(k) ~ t_verify(k0) * (1 + slope*k) / (1 + slope*k0)
    probe_every: int = 8  # when k=0 wins, still draft every Nth round to keep p fresh
    fixed: int | None = None  # fixed k (baselines); None = adaptive
    p: list[float] = field(default_factory=list)
    _rounds_at_zero: int = 0

    def __post_init__(self) -> None:
        if not self.p:
            self.p = [self.p_prior] * self.k_max

    def _tv(self, k: int) -> float | None:
        if k in self.t_verify:
            return self.t_verify[k]
        if not self.t_verify:
            return None
        k0 = min(self.t_verify, key=lambda x: abs(x - k))
        return self.t_verify[k0] * (1 + self.verify_slope * k) / (1 + self.verify_slope * k0)

    def expected_tokens(self, k: int) -> float:
        e, run = 1.0, 1.0
        for i in range(k):
            run *= self.p[i]
            e += run
        return e

    def cost(self, k: int) -> float:
        if k == 0:
            return self._tv(0)
        return self.t_first + (k - 1) * self.t_step + self._tv(k)

    def choose(self) -> int:
        if self.fixed is not None:
            return self.fixed
        if self.t_first is None or self.t_step is None or self._tv(0) is None:
            # warm-up: one plain step (measures t_verify(0)), then drafts of k_init
            return 0 if 0 not in self.t_verify else self.k_init
        best_k = max(range(self.k_max + 1), key=lambda k: self.expected_tokens(k) / self.cost(k))
        if best_k == 0:
            self._rounds_at_zero += 1
            if self._rounds_at_zero % self.probe_every == 0:
                return self.k_init  # a probe long enough to re-measure several positions
        return best_k

    def _upd(self, old: float | None, new: float) -> float:
        return new if old is None else (1 - self.ema) * old + self.ema * new

    def observe(self, k: int, accepted: int, t_first: float, t_rest: float, t_verify: float,
                catch_up: int = 1) -> None:
        """``catch_up``: tokens the draft's first call fed. After plain rounds the draft lags
        many tokens; that one-off cost must not count as the per-round first-call cost, or
        the policy talks itself out of drafting for good."""
        self.t_verify[k] = self._upd(self.t_verify.get(k), t_verify)
        if k > 0:
            if catch_up <= 2 or self.t_first is None:
                self.t_first = self._upd(self.t_first, t_first)
            if k > 1:
                self.t_step = self._upd(self.t_step, t_rest / (k - 1))
            elif self.t_step is None:
                self.t_step = t_first  # until a k>1 round measures it
            # positions 0..accepted-1 accepted; position `accepted` rejected (if < k)
            for i in range(min(accepted + 1, k)):
                self.p[i] = self._upd(self.p[i], 1.0 if i < accepted else 0.0)


@dataclass
class GenStats:
    tokens: list[int] = field(default_factory=list)
    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    t_prefill: float = 0.0
    t_draft: float = 0.0
    t_verify: float = 0.0
    t_total: float = 0.0
    ks: list[int] = field(default_factory=list)
    prefill: PrefillInfo | None = None
    prefill_draft: PrefillInfo | None = None

    @property
    def decode_tps(self) -> float:
        dec = self.t_total - self.t_prefill
        return len(self.tokens) / dec if dec > 0 else 0.0


def generate(target: Executor, prompt: list[int], max_tokens: int, draft: Executor | None = None,
             policy: AdaptiveK | None = None, cache: PrefixCache | None = None) -> GenStats:
    """Greedy generation of up to ``max_tokens`` after ``prompt``; speculative when ``draft`` is given.

    Prefill reuses what the executors still hold, and ``cache`` snapshots when given
    (``prefix_cache.prefill``). Stops at an end-of-generation token (included).
    """
    policy = policy or AdaptiveK(fixed=None if draft else 0)
    st = GenStats()
    t0 = time.perf_counter()
    # snapshots are taken after generation (off the TTFT path; they then include the answer)
    rows, st.prefill = prefill(target, prompt, cache, store=False)
    seq = list(prompt) + [int(rows.tokens[-1])]
    st.tokens.append(seq[-1])
    if draft is not None:
        if len(prompt) > 1:
            _, st.prefill_draft = prefill(draft, prompt[:-1], cache, store=False)
        else:
            draft.reset()
    st.t_prefill = time.perf_counter() - t0

    while len(st.tokens) < max_tokens and not target.is_eog(seq[-1]):
        k = 0 if draft is None else policy.choose()
        k = min(k, max_tokens - len(st.tokens))  # never draft past the budget
        base = len(seq) - 1  # target holds seq[:base]
        drafts: list[int] = []
        td = time.perf_counter()
        tf = td
        missing: list[int] = []
        if k > 0:
            missing = seq[draft.n_past:]
            r = draft.eval(missing)
            drafts.append(int(r.tokens[-1]))
            tf = time.perf_counter()
            while len(drafts) < k:
                drafts.append(int(draft.eval([drafts[-1]]).tokens[-1]))
        tv = time.perf_counter()
        ys = target.eval([seq[-1]] + drafts, all_logits=k > 0).tokens
        t_end = time.perf_counter()

        j = 0
        while j < k and drafts[j] == int(ys[j]):
            j += 1
        new = drafts[:j] + [int(ys[j] if k > 0 else ys[-1])]
        target.truncate(base + 1 + j)
        if k > 0:
            draft.truncate(min(draft.n_past, base + 1 + j))
        done = False
        for t in new:  # an accepted draft may already end the generation
            seq.append(t)
            st.tokens.append(t)
            if target.is_eog(t) or len(st.tokens) >= max_tokens:
                done = True
                break
        st.rounds += 1
        st.drafted += k
        st.accepted += j
        st.ks.append(k)
        st.t_draft += tv - td
        st.t_verify += t_end - tv
        policy.observe(k, j, tf - td, tv - tf, t_end - tv, catch_up=len(missing))
        if done:
            break
    st.tokens = st.tokens[:max_tokens]
    st.t_total = time.perf_counter() - t0
    if cache is not None:
        for ex in (target, draft):
            if ex is not None:
                cache.put(ex.snapshot())
    return st
