from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import numpy as np
from scipy.special import logsumexp

from .static_world import Query, World, observation_probability, terminal_loss

Action = tuple[str, int]
History = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class HistoryDPResult:
    value: float
    action: Action
    seconds: float
    states: int
    posterior_evaluations: int


class OrderedHistoryDP:
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
        self._lik = np.empty((len(self.queries), 2, len(self.worlds)), dtype=float)
        for qi, q in enumerate(self.queries):
            for o in (0,1):
                self._lik[qi,o,:] = [observation_probability(q,w,self.models,o) for w in self.worlds]
        self._log_lik = np.log(np.clip(self._lik, 1e-300, None))
        self._loss = np.asarray([[terminal_loss(d,w,self.models,false_allow=false_allow, false_block=false_block) for w in self.worlds] for d in (0,1)], dtype=float)
        self.stats = {"states":0, "posterior_evaluations":0}
        self._value_cache: dict[History, tuple[float, Action]] = {}

    def belief(self, history: History) -> tuple[float, ...]:
        self.stats["posterior_evaluations"] += 1
        logw = self._log_initial.copy()
        for qi, obs in history:
            logw += self._log_lik[int(qi), int(obs)]
        logw -= logsumexp(logw)
        return tuple(np.exp(logw).tolist())

    def V(self, history: History) -> tuple[float, Action]:
        cached = self._value_cache.get(history)
        if cached is not None:
            return cached
        self.stats["states"] += 1
        b = np.asarray(self.belief(history), dtype=float)
        risks = self._loss @ b
        d = int(np.argmin(risks))
        best = float(risks[d])
        action: Action = ("DECIDE", d)
        if len(history) >= self.horizon:
            result = (best, action)
            self._value_cache[history] = result
            return result
        for qi, q in enumerate(self.queries):
            if q.cost >= best - 1e-15:
                continue
            value = float(q.cost)
            for obs in (0,1):
                po = float(self._lik[qi,obs] @ b)
                if po > 1e-14:
                    value += po * self.V(history + ((qi,obs),))[0]
            if value < best - 1e-12:
                best, action = value, ("QUERY", qi)
        result = (float(best), action)
        self._value_cache[history] = result
        return result

    def solve(self) -> HistoryDPResult:
        t0 = perf_counter()
        value, action = self.V(tuple())
        return HistoryDPResult(value=value, action=action, seconds=perf_counter()-t0, states=int(self.stats["states"]), posterior_evaluations=int(self.stats["posterior_evaluations"]))
