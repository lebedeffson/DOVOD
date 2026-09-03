from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class PlannedIntervention:
    kind: str  # PHYSICAL_QUERY | SEMANTIC_REVIEW | RESOLVED
    target: int
    expected_remaining_cost: float


class SourceAwareResolutionPlanner:
    """Exact finite-horizon intervention planner for one target action.

    Physical queries perfectly reveal only completion/not-completion for one component.
    A semantic review reveals which train-equivalent prerequisite alternative applies to
    the target action. The planner minimizes expected cost to zero action-admissibility
    variance. It is an analysis-layer planner; real RGB/noisy review are separate gates.
    """

    def __init__(
        self,
        graphs: Sequence[dict[int, list[int]]],
        action: int,
        queryable_components: Sequence[int],
        *,
        physical_cost: float = 1.0,
        semantic_cost: float = 1.0,
    ):
        self.graphs = list(graphs)
        self.action = int(action)
        self.queryable = tuple(sorted(set(map(int, queryable_components))))
        self.physical_cost = float(physical_cost)
        self.semantic_cost = float(semantic_cost)
        if self.physical_cost <= 0 or self.semantic_cost <= 0:
            raise ValueError("Intervention costs must be positive")

    @staticmethod
    def _qkey(q: Sequence[float]) -> tuple[float, ...]:
        return tuple(round(float(x), 12) for x in q)

    @lru_cache(maxsize=None)
    def _uncertainty(self, qkey: tuple[float, ...], active: tuple[int, ...]) -> float:
        not_complete = 1.0 - float(qkey[self.action])
        probs = []
        for i in active:
            p = not_complete
            for pred in self.graphs[i].get(self.action, []):
                p *= float(qkey[int(pred)])
            probs.append(p)
        mean_p = float(sum(probs) / len(probs))
        return mean_p * (1.0 - mean_p)

    @lru_cache(maxsize=None)
    def _semantic_groups(self, active: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
        groups: dict[tuple[int, ...], list[int]] = {}
        for i in active:
            alt = tuple(sorted(self.graphs[i].get(self.action, [])))
            groups.setdefault(alt, []).append(i)
        return tuple(tuple(v) for _, v in sorted(groups.items()))

    def solve(self, q: Sequence[float], mode: str = "optimal") -> PlannedIntervention:
        qkey = self._qkey(q)
        active = tuple(range(len(self.graphs)))
        cost, decision = self._value(qkey, active, mode)
        if decision is None:
            return PlannedIntervention("RESOLVED", -1, float(cost))
        return PlannedIntervention(decision[0], int(decision[1]), float(cost))

    @lru_cache(maxsize=None)
    def _value(
        self, qkey: tuple[float, ...], active: tuple[int, ...], mode: str
    ) -> tuple[float, tuple[str, int] | None]:
        if self._uncertainty(qkey, active) <= EPS:
            return 0.0, None

        unresolved = [j for j in self.queryable if EPS < qkey[j] < 1.0 - EPS]
        groups = self._semantic_groups(active)
        semantic_available = len(groups) > 1
        candidates: list[tuple[float, tuple[str, int]]] = []

        def add_physical_candidates() -> None:
            for j in unresolved:
                pj = float(qkey[j])
                q1 = list(qkey); q1[j] = 1.0
                q0 = list(qkey); q0[j] = 0.0
                c1, _ = self._value(tuple(q1), active, mode)
                c0, _ = self._value(tuple(q0), active, mode)
                expected = self.physical_cost + pj * c1 + (1.0 - pj) * c0
                candidates.append((float(expected), ("PHYSICAL_QUERY", int(j))))

        def add_semantic_candidate() -> None:
            if not semantic_available:
                return
            expected = self.semantic_cost
            denom = float(len(active))
            for group in groups:
                mass = len(group) / denom
                cg, _ = self._value(qkey, group, mode)
                expected += mass * cg
            candidates.append((float(expected), ("SEMANTIC_REVIEW", self.action)))

        if mode == "physical_first":
            if unresolved:
                add_physical_candidates()
            else:
                add_semantic_candidate()
        elif mode == "semantic_first":
            if semantic_available:
                add_semantic_candidate()
            else:
                add_physical_candidates()
        elif mode == "optimal":
            add_physical_candidates()
            add_semantic_candidate()
        else:
            raise ValueError(mode)

        if not candidates:
            return float("inf"), None
        return min(candidates, key=lambda x: (x[0], x[1][0] != "SEMANTIC_REVIEW", x[1][1]))

    def realized_cost(
        self,
        q: Sequence[float],
        true_state: Sequence[int],
        latent_graph_index: int,
        mode: str = "optimal",
    ) -> dict:
        qkey = self._qkey(q)
        active = tuple(range(len(self.graphs)))
        latent = int(latent_graph_index)
        total_cost = 0.0
        physical = 0
        semantic = 0
        trace: list[str] = []

        for _ in range(len(self.queryable) + 3):
            if self._uncertainty(qkey, active) <= EPS:
                break
            _, decision = self._value(qkey, active, mode)
            if decision is None:
                break
            kind, target = decision
            if kind == "PHYSICAL_QUERY":
                ql = list(qkey)
                ql[target] = 1.0 if int(true_state[target]) == 1 else 0.0
                qkey = tuple(ql)
                total_cost += self.physical_cost
                physical += 1
                trace.append(f"PHYSICAL:{target}")
            elif kind == "SEMANTIC_REVIEW":
                latent_alt = tuple(sorted(self.graphs[latent].get(self.action, [])))
                active = tuple(
                    i for i in active
                    if tuple(sorted(self.graphs[i].get(self.action, []))) == latent_alt
                )
                total_cost += self.semantic_cost
                semantic += 1
                trace.append(f"SEMANTIC:{self.action}")
            else:
                raise AssertionError(kind)

        final = self._uncertainty(qkey, active)
        return {
            "realized_cost": float(total_cost),
            "physical_queries": int(physical),
            "semantic_queries": int(semantic),
            "resolved": int(final <= EPS),
            "final_variance": float(final),
            "trace": ">".join(trace),
        }
