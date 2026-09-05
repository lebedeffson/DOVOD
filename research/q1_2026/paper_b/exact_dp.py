from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np

from .static_world import Query, World, observation_probability, terminal_loss

Action = tuple[str, int]


def normalize(xs: Sequence[float]) -> tuple[float, ...]:
    z = float(sum(xs))
    if z <= 0:
        raise ValueError("zero probability mass")
    return tuple(float(x) / z for x in xs)


def canonical(b: Sequence[float], ndigits: int | None = 12) -> tuple[float, ...]:
    vals = normalize(b)
    if ndigits is None:
        return vals
    return tuple(round(float(x), int(ndigits)) for x in vals)


def solve_exact_belief_dp(
    initial: Sequence[float], worlds: Sequence[World], models: Sequence[Sequence[int]], queries: Sequence[Query],
    *, horizon: int, false_allow: float = 2.0, false_block: float = 1.0, canonical_ndigits: int | None = 12,
) -> tuple[tuple[float, Action], object]:
    worlds = tuple(worlds)
    queries = tuple(queries)
    models = tuple(tuple(m) for m in models)
    likelihood = np.asarray([[[observation_probability(q, w, models, o) for w in worlds] for o in (0,1)] for q in queries], dtype=float)
    losses = np.asarray([[terminal_loss(d, w, models, false_allow=false_allow, false_block=false_block) for w in worlds] for d in (0,1)], dtype=float)

    @lru_cache(None)
    def V(bkey: tuple[float, ...], h: int) -> tuple[float, Action]:
        b = np.asarray(bkey, dtype=float)
        b = b / b.sum()
        risks = losses @ b
        d = int(np.argmin(risks))
        best = float(risks[d])
        action: Action = ("DECIDE", d)
        if h <= 0:
            return best, action
        for qi, q in enumerate(queries):
            if q.cost >= best - 1e-15:
                continue
            value = float(q.cost)
            for o in (0,1):
                unnorm = b * likelihood[qi,o]
                po = float(unnorm.sum())
                if po > 1e-14:
                    b2 = canonical(unnorm / po, canonical_ndigits)
                    value += po * V(b2, h-1)[0]
            if value < best - 1e-12:
                best, action = value, ("QUERY", qi)
        return float(best), action

    return V(canonical(initial, canonical_ndigits), int(horizon)), V
