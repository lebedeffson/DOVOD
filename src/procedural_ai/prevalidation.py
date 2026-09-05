from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .planning import SourceAwareResolutionPlanner


@dataclass(frozen=True)
class PrevalidationResult:
    reviewed_actions: tuple[int, ...]
    semantic_groups: int
    expected_runtime_cost: float
    prevalidation_cost: float = 0.0
    amortized_total_cost: float | None = None


def partition_graphs_by_actions(
    graphs: Sequence[dict[int, list[int]]], reviewed_actions: Iterable[int]
) -> tuple[tuple[dict[int, list[int]], ...], ...]:
    """Partition the frozen semantic version space by reviewed action semantics."""
    actions = tuple(sorted(set(map(int, reviewed_actions))))
    groups: dict[tuple[tuple[int, tuple[int, ...]], ...], list[dict[int, list[int]]]] = {}
    for graph in graphs:
        key = tuple((a, tuple(sorted(graph.get(a, [])))) for a in actions)
        groups.setdefault(key, []).append(graph)
    return tuple(tuple(group) for _, group in sorted(groups.items(), key=lambda kv: kv[0]))


def expected_runtime_cost_after_prevalidation(
    *,
    q: Sequence[float],
    graphs: Sequence[dict[int, list[int]]],
    action: int,
    queryable_components: Sequence[int],
    reviewed_actions: Iterable[int],
    physical_cost: float = 1.0,
    semantic_cost: float = 1.0,
    prevalidation_review_cost: float = 0.0,
    amortization_horizon: int | None = None,
) -> PrevalidationResult:
    """Expected downstream runtime cost after a one-time semantic prevalidation.

    The current short-paper model uses a uniform prior over the frozen semantic version
    space. ``expected_runtime_cost`` excludes the up-front prevalidation effort. If a
    per-review cost and horizon are supplied, ``amortized_total_cost`` adds that one-time
    cost transparently instead of treating prevalidation as free.
    """
    if not graphs:
        raise ValueError("graphs must be non-empty")
    actions = tuple(sorted(set(map(int, reviewed_actions))))
    groups = partition_graphs_by_actions(graphs, actions)
    runtime_total = 0.0
    for group in groups:
        mass = len(group) / len(graphs)
        planner = SourceAwareResolutionPlanner(
            group,
            action=int(action),
            queryable_components=queryable_components,
            physical_cost=physical_cost,
            semantic_cost=semantic_cost,
        )
        runtime_total += mass * planner.solve(q, mode="optimal").expected_remaining_cost

    per_review = float(prevalidation_review_cost)
    if per_review < 0:
        raise ValueError("prevalidation_review_cost must be non-negative")
    upfront = len(actions) * per_review
    amortized: float | None = None
    if upfront == 0:
        amortized = float(runtime_total)
    elif amortization_horizon is not None:
        horizon = int(amortization_horizon)
        if horizon < 1:
            raise ValueError("amortization_horizon must be positive")
        amortized = float(runtime_total + upfront / horizon)

    return PrevalidationResult(
        reviewed_actions=actions,
        semantic_groups=len(groups),
        expected_runtime_cost=float(runtime_total),
        prevalidation_cost=float(upfront),
        amortized_total_cost=amortized,
    )


def rank_prevalidation_subsets(
    *,
    q: Sequence[float],
    graphs: Sequence[dict[int, list[int]]],
    action: int,
    queryable_components: Sequence[int],
    candidate_actions: Sequence[int],
    max_reviews: int | None = None,
    physical_cost: float = 1.0,
    semantic_cost: float = 1.0,
    prevalidation_review_cost: float = 0.0,
    amortization_horizon: int | None = None,
) -> list[PrevalidationResult]:
    candidates = tuple(sorted(set(map(int, candidate_actions))))
    limit = len(candidates) if max_reviews is None else min(int(max_reviews), len(candidates))
    if limit < 0:
        raise ValueError("max_reviews must be non-negative")
    results: list[PrevalidationResult] = []
    for k in range(limit + 1):
        for subset in itertools.combinations(candidates, k):
            results.append(
                expected_runtime_cost_after_prevalidation(
                    q=q,
                    graphs=graphs,
                    action=action,
                    queryable_components=queryable_components,
                    reviewed_actions=subset,
                    physical_cost=physical_cost,
                    semantic_cost=semantic_cost,
                    prevalidation_review_cost=prevalidation_review_cost,
                    amortization_horizon=amortization_horizon,
                )
            )
    return sorted(
        results,
        key=lambda r: (
            r.amortized_total_cost if r.amortized_total_cost is not None else r.expected_runtime_cost,
            len(r.reviewed_actions),
            r.reviewed_actions,
        ),
    )
