from __future__ import annotations

from dataclasses import dataclass
from math import comb
from time import perf_counter
from typing import Sequence

import numpy as np
from scipy.special import logsumexp

from .static_world import Query, World, observation_probability, terminal_loss

Action = tuple[str, int]


def evidence_count_state_count(query_count: int, horizon: int) -> int:
    q, h = int(query_count), int(horizon)
    if q < 0 or h < 0:
        raise ValueError("query_count and horizon must be non-negative")
    return comb(h + 2*q, h)


def ordered_history_state_count(query_count: int, horizon: int) -> int:
    q, h = int(query_count), int(horizon)
    if q < 0 or h < 0:
        raise ValueError("query_count and horizon must be non-negative")
    return sum((2*q)**t for t in range(h+1))


@dataclass(frozen=True)
class CountDPResult:
    value: float
    action: Action
    seconds: float
    states: int
    query_evaluations: int
    posterior_evaluations: int


class EvidenceCountDP:
    def __init__(self, initial: Sequence[float], worlds: Sequence[World], models: Sequence[Sequence[int]], queries: Sequence[Query], *, horizon: int, false_allow: float = 2.0, false_block: float = 1.0) -> None:
        self.worlds = tuple(worlds)
        self.models = tuple(tuple(m) for m in models)
        self.queries = tuple(queries)
        self.horizon = int(horizon)
        if self.horizon < 0:
            raise ValueError("horizon must be non-negative")
        b = np.asarray(initial, dtype=float)
        if len(b) != len(self.worlds) or len(b) == 0 or np.any(b < 0) or b.sum() <= 0:
            raise ValueError("invalid initial belief")
        b = b / b.sum()
        self._log_initial = np.log(np.clip(b, 1e-300, None))
        self._lik = np.empty((len(self.queries),2,len(self.worlds)), dtype=float)
        for qi, q in enumerate(self.queries):
            for o in (0,1):
                self._lik[qi,o,:] = [observation_probability(q,w,self.models,o) for w in self.worlds]
        self._log_lik = np.log(np.clip(self._lik, 1e-300, None))
        self._loss = np.asarray([[terminal_loss(d,w,self.models,false_allow=false_allow,false_block=false_block) for w in self.worlds] for d in (0,1)], dtype=float)
        self.stats = {"states":0, "query_evaluations":0, "posterior_evaluations":0}
        self._value_cache: dict[tuple[int, ...], tuple[float, Action]] = {}

    @property
    def zero_counts(self) -> tuple[int, ...]:
        return (0,) * (2*len(self.queries))

    def belief_from_counts(self, counts: tuple[int, ...]) -> tuple[float, ...]:
        if len(counts) != 2*len(self.queries) or any(c < 0 for c in counts):
            raise ValueError("invalid count state")
        self.stats["posterior_evaluations"] += 1
        logw = self._log_initial.copy()
        for qi in range(len(self.queries)):
            c0, c1 = counts[2*qi], counts[2*qi+1]
            if c0:
                logw += c0*self._log_lik[qi,0]
            if c1:
                logw += c1*self._log_lik[qi,1]
        logw -= logsumexp(logw)
        return tuple(np.exp(logw).tolist())

    @staticmethod
    def _inc(counts: tuple[int, ...], qi: int, obs: int) -> tuple[int, ...]:
        xs = list(counts)
        xs[2*qi + int(obs)] += 1
        return tuple(xs)

    def V(self, counts: tuple[int, ...]) -> tuple[float, Action]:
        cached = self._value_cache.get(counts)
        if cached is not None:
            return cached
        self.stats["states"] += 1
        used = sum(counts)
        b = np.asarray(self.belief_from_counts(counts), dtype=float)
        risks = self._loss @ b
        d = int(np.argmin(risks))
        best = float(risks[d])
        action: Action = ("DECIDE", d)
        if used >= self.horizon:
            result = (best, action)
            self._value_cache[counts] = result
            return result
        for qi, q in enumerate(self.queries):
            if q.cost >= best - 1e-15:
                continue
            self.stats["query_evaluations"] += 1
            value = float(q.cost)
            for o in (0,1):
                po = float(self._lik[qi,o] @ b)
                if po > 1e-14:
                    value += po * self.V(self._inc(counts, qi, o))[0]
            if value < best - 1e-12:
                best, action = value, ("QUERY", qi)
        result = (float(best), action)
        self._value_cache[counts] = result
        return result

    def root_action_values(self) -> dict[Action, float]:
        counts = self.zero_counts
        b = np.asarray(self.belief_from_counts(counts), dtype=float)
        risks = self._loss @ b
        out: dict[Action, float] = {("DECIDE",0):float(risks[0]), ("DECIDE",1):float(risks[1])}
        if self.horizon <= 0:
            return out
        for qi, q in enumerate(self.queries):
            value = float(q.cost)
            for o in (0,1):
                po = float(self._lik[qi,o] @ b)
                if po > 1e-14:
                    value += po * self.V(self._inc(counts,qi,o))[0]
            out[("QUERY",qi)] = float(value)
        return out

    def solve(self) -> CountDPResult:
        t0 = perf_counter()
        value, action = self.V(self.zero_counts)
        return CountDPResult(value=value, action=action, seconds=perf_counter()-t0, states=int(self.stats["states"]), query_evaluations=int(self.stats["query_evaluations"]), posterior_evaluations=int(self.stats["posterior_evaluations"]))
