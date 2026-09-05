from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .planning import MyopicSourceSelector, PlannedIntervention, SourceAwareResolutionPlanner


def exact_state_upper_bound(unresolved_physical: int, semantic_alternatives: int) -> int:
    """Conservative memoization envelope for the current perfect-reveal model."""
    k = int(unresolved_physical); g = int(semantic_alternatives)
    if k < 0 or g < 1:
        raise ValueError
    return (g + 1) * (3 ** k)


@dataclass(frozen=True)
class HybridPlan:
    decision: PlannedIntervention
    policy: str
    unresolved_physical: int
    semantic_alternatives: int
    state_upper_bound: int


class HybridSourceSelector:
    """Exact Bellman below a state-envelope threshold, gain-per-cost myopic above it."""
    def __init__(self, graphs: Sequence[dict[int, list[int]]], action: int, queryable_components: Sequence[int], *, physical_cost: float = 1.0, semantic_cost: float = 1.0, max_exact_states: int = 100_000):
        self.graphs = list(graphs)
        if not self.graphs or max_exact_states < 1:
            raise ValueError
        self.action = int(action)
        self.queryable = tuple(sorted(set(map(int, queryable_components))))
        self.physical_cost = float(physical_cost); self.semantic_cost = float(semantic_cost)
        self.max_exact_states = int(max_exact_states)

    def solve(self, q: Sequence[float]) -> HybridPlan:
        qkey = SourceAwareResolutionPlanner._qkey(q)
        unresolved = sum(1 for j in self.queryable if 1e-12 < qkey[j] < 1 - 1e-12)
        alternatives = len({tuple(sorted(g.get(self.action, []))) for g in self.graphs})
        envelope = exact_state_upper_bound(unresolved, max(1, alternatives))
        if envelope <= self.max_exact_states:
            planner = SourceAwareResolutionPlanner(self.graphs, self.action, self.queryable, physical_cost=self.physical_cost, semantic_cost=self.semantic_cost)
            decision = planner.solve(q, mode="optimal"); policy = "exact"
        else:
            planner = MyopicSourceSelector(self.graphs, self.action, self.queryable, physical_cost=self.physical_cost, semantic_cost=self.semantic_cost)
            decision = planner.solve_myopic(q); policy = "myopic"
        return HybridPlan(decision, policy, unresolved, alternatives, envelope)
