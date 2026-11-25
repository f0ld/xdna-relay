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
             policy: AdaptiveK | None = None, cache: PrefixCache | None = None,
             on_token=None) -> GenStats:
    """Greedy generation of up to ``max_tokens`` after ``prompt``; speculative when ``draft`` is given.

    Prefill reuses what the executors still hold, and ``cache`` snapshots when given
    (``prefix_cache.prefill``). Stops at an end-of-generation token (included).
    ``on_token(token)`` is called for every generated token as it is accepted; returning True
    stops generation there (a client went away).
    """
    stopped = False

    def emit(t: int) -> None:
        nonlocal stopped
        if on_token is not None and not stopped and on_token(t):
            stopped = True

    policy = policy or AdaptiveK(fixed=None if draft else 0)
    st = GenStats()
    t0 = time.perf_counter()
    # snapshots are taken after generation (off the TTFT path; they then include the answer)
    rows, st.prefill = prefill(target, prompt, cache, store=False)
    seq = list(prompt) + [int(rows.tokens[-1])]
    st.tokens.append(seq[-1])
    emit(seq[-1])
    if draft is not None:
        if len(prompt) > 1:
            _, st.prefill_draft = prefill(draft, prompt[:-1], cache, store=False)
        else:
            draft.reset()
    st.t_prefill = time.perf_counter() - t0

    while len(st.tokens) < max_tokens and not target.is_eog(seq[-1]) and not stopped:
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
            emit(t)
            if target.is_eog(t) or len(st.tokens) >= max_tokens or stopped:
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


# ---- batched speculation (T-041) ------------------------------------------------------

def _interp(d: dict[int, float], x: int, slope: float) -> float | None:
    """Cost at ``x`` tokens from costs measured at other token counts: linear between the
    nearest measured counts; outside them, the nearest one scaled by a relative ``slope``."""
    if x in d:
        return d[x]
    if not d:
        return None
    ks = sorted(d)
    if ks[0] < x < ks[-1]:
        hi = next(k for k in ks if k > x)
        lo = max(k for k in ks if k < x)
        return d[lo] + (d[hi] - d[lo]) * (x - lo) / (hi - lo)
    k0 = ks[0] if x < ks[0] else ks[-1]
    return d[k0] * (1 + slope * x) / (1 + slope * k0)


def passes(counts: list[int]) -> list[int]:
    """Tokens per pass over the weights for a batch giving ``counts[i]`` tokens to sequence i,
    when llama.cpp splits it into equal-count ubatches (per-sequence KV buffers): one pass per
    distinct count. [2, 1, 1, 1] -> [4, 1]; [3, 3, 1, 1] -> [4, 4]; [2, 2] -> [4]."""
    out, prev = [], 0
    for c in sorted({c for c in counts if c > 0}):
        out.append(sum(1 for x in counts if x >= c) * (c - prev))
        prev = c
    return out


@dataclass
class Acceptance:
    """Per-position acceptance of one sequence's drafts (EMAs): p[i] = P(draft i+1 accepted
    | drafts 1..i accepted)."""

    p: list[float]
    ema: float = 0.15

    def expected(self, k: int) -> float:
        """Expected tokens a round with draft length k emits (accepted drafts + 1)."""
        e, run = 1.0, 1.0
        for i in range(k):
            run *= self.p[i]
            e += run
        return e

    def observe(self, k: int, accepted: int) -> None:
        for i in range(min(accepted + 1, k)):
            self.p[i] = (1 - self.ema) * self.p[i] + self.ema * (1.0 if i < accepted else 0.0)


@dataclass
class BatchK:
    """Draft lengths for one round of the batched stepper (T-041).

    A round: the draft model runs k steps for the drafting sequences in batched decodes (the
    first also feeds each sequence's missing tokens and any draft catch-up chunks), then one
    target batch verifies every decoding sequence's [last, d1..dk] (plus any prefill chunk
    riding along). Costs are measured per batch size in tokens: verify(T) for the target
    batch, draft(T) for a draft call; so a draft catch-up or a prefill chunk is costed by its
    size, not folded into a per-token rate (T-030's catch-up pitfall cannot recur).
    ``split_ragged``: the target splits a batch whose sequences get different token counts
    into one pass per distinct count (per-sequence KV buffers; ``passes``): verify costs the
    sum of its passes, so uneven shapes (some sequences drafting, a prefill chunk) pay extra.

    The component model misses some costs (Python bookkeeping, rounds after requests end,
    GPU work that overlaps); so ``choose`` also checks the outcome: per number of decoding
    sequences, the tokens/s that drafting and plain rounds actually achieved (``rates``,
    pure decode rounds only). When drafting measured slower, it does not draft; every
    ``explore_every``-th round at that size takes the other arm to keep both rates fresh.

    ``choose`` maximises expected tokens / round seconds over a common k and the subset of
    sequences that draft (those with the highest expected gain first); the rest verify one
    token. A batch of several sequences makes extra verify rows costlier relative to what
    each returns, so k shrinks as the batch grows once verify stops being flat.
    """

    k_max: int = 8
    k_init: int = 3
    ema: float = 0.15
    p_prior: float = 0.6
    verify_slope: float = 0.02  # relative cost per extra token, until measured
    draft_slope: float = 0.02
    probe_every: int = 8
    fixed: int | None = None
    split_ragged: bool = False
    verify: dict[int, float] = field(default_factory=dict)
    draft: dict[int, float] = field(default_factory=dict)
    p: list[float] = field(default_factory=list)  # pooled acceptance: the prior of new sequences
    rate_ema: float = 0.1
    explore_every: int = 16
    rates: dict = field(default_factory=dict)  # (decoding seqs, drafted) -> [tokens/s EMA, rounds]
    _zero_rounds: int = 0
    _size_rounds: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.p:
            self.p = [self.p_prior] * self.k_max

    def new_sequence(self) -> Acceptance:
        return Acceptance(list(self.p), self.ema)

    def _upd(self, d: dict[int, float], key: int, v: float) -> None:
        d[key] = v if key not in d else (1 - self.ema) * d[key] + self.ema * v

    def verify_cost(self, shape: list[int]) -> float:
        """Target batch giving ``shape[i]`` tokens to sequence i."""
        if self.split_ragged:
            return sum(_interp(self.verify, t, self.verify_slope) for t in passes(shape))
        return _interp(self.verify, sum(shape), self.verify_slope)

    def round_cost(self, k: int, drafting: list[int], catch_up: list[int], n_rows: int,
                   extra_draft: int = 0, extra_verify: int = 0) -> float:
        """Seconds of a round: ``drafting`` = indices drafting k, ``catch_up[i]`` = tokens
        sequence i's first draft call feeds, ``n_rows`` = decoding sequences in all."""
        t = 0.0
        if k > 0 and drafting:
            t += _interp(self.draft, sum(catch_up[i] for i in drafting) + extra_draft, self.draft_slope)
            if k > 1:
                t += (k - 1) * _interp(self.draft, len(drafting), self.draft_slope)
        elif extra_draft:
            t += _interp(self.draft, extra_draft, self.draft_slope) or 0.0
        m = len(drafting) if k else 0
        return t + self.verify_cost([k + 1] * m + [1] * (n_rows - m) + ([extra_verify] if extra_verify else []))

    def choose(self, accs: list[Acceptance], catch_up: list[int], n_plain: int = 0,
               extra_draft: int = 0, extra_verify: int = 0) -> list[int]:
        """Draft length per sequence in ``accs`` (each 0 or a common k). ``n_plain``: other
        decoding sequences that cannot draft this round (verify one token each)."""
        n = len(accs)
        if n == 0:
            return []
        if self.fixed is not None:
            return [self.fixed] * n
        if not self.verify:
            return [0] * n  # warm-up: a plain round measures verify
        if not self.draft:
            return [self.k_init] * n
        rows = n + n_plain
        best, best_rate = [0] * n, (n + n_plain) / self.round_cost(0, [], catch_up, rows, extra_draft, extra_verify)
        for k in range(1, self.k_max + 1):
            order = sorted(range(n), key=lambda i: -accs[i].expected(k))
            for m in range(1, n + 1):
                sub = order[:m]
                toks = sum(accs[i].expected(k) for i in sub) + (rows - m)
                rate = toks / self.round_cost(k, sub, catch_up, rows, extra_draft, extra_verify)
                if rate > best_rate:
                    best_rate = rate
                    best = [k if i in sub else 0 for i in range(n)]
        if not extra_verify:  # outcome check, pure decode rounds
            c = self._size_rounds[rows] = self._size_rounds.get(rows, 0) + 1
            on, off = self.rates.get((rows, True)), self.rates.get((rows, False))
            if c % self.explore_every == 0:  # the arm not chosen below
                draft_wins = on is None or off is None or on[0] >= off[0]
                if any(best) and draft_wins:
                    return [0] * n
                return best if any(best) else [self.k_init] * n
            if any(best) and on and off and on[0] < off[0]:
                return [0] * n
        if not any(best):
            self._zero_rounds += 1
            if self._zero_rounds % self.probe_every == 0:
                return [self.k_init] * n  # keep acceptance fresh
        return best

    def observe_round(self, n_rows: int, drafted: bool, tokens: int, seconds: float) -> None:
        """A pure decode round's outcome: ``tokens`` emitted by ``n_rows`` sequences in
        ``seconds`` (the whole round, bookkeeping included)."""
        if seconds <= 0 or n_rows <= 0:
            return
        key, v = (n_rows, drafted), tokens / seconds
        r = self.rates.get(key)
        self.rates[key] = [v, 1] if r is None else [(1 - self.rate_ema) * r[0] + self.rate_ema * v, r[1] + 1]

    def observe(self, ks: list[int], accepted: list[int], accs: list[Acceptance], *, t_first: float,
                first_tokens: int, t_rest: float, t_verify: float, verify_shape: list[int]) -> None:
        """One round: per drafting sequence its k and accepted drafts; the first draft call
        (``first_tokens`` fed), the further draft calls, the verify batch (tokens per
        sequence). A split verify (several passes) is not attributed to its passes."""
        p = passes(verify_shape) if self.split_ragged else [sum(verify_shape)]
        if len(p) == 1:
            self._upd(self.verify, p[0], t_verify)
        k_hi = max(ks, default=0)
        if first_tokens > 0:  # also a catch-up-only draft call (no drafting this round)
            self._upd(self.draft, first_tokens, t_first)
        if k_hi > 0:
            if k_hi > 1:
                self._upd(self.draft, sum(1 for k in ks if k > 1), t_rest / (k_hi - 1))
        pooled = Acceptance(self.p, self.ema)
        for k, j, acc in zip(ks, accepted, accs):
            if k > 0:
                acc.observe(k, j)
                pooled.observe(k, j)
