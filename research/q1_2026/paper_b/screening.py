from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .count_dp import CountDPResult, EvidenceCountDP
from .static_world import Query, World

Action = tuple[str, int]


@dataclass(frozen=True)
class TypedCoreResult:
    value: float
    action: Action
    selected_query_indices: tuple[int, ...]
    reduced_query_count: int
    states: int
    seconds: float


def decision_bearing_query_indices(queries: Sequence[Query]) -> tuple[int, ...]:
    """Keep direct state/model evidence and drop explicit calibration-only queries.

    This is an engineering screen, not an exactness guarantee. It should be enabled
    only after a workload-specific ablation shows that calibration queries have
    negligible policy value.
    """
    return tuple(
        i for i, query in enumerate(queries)
        if query.kind in ("state", "model_feature")
    )


def map_reduced_action(action: Action, selected_query_indices: Sequence[int]) -> Action:
    if action[0] == "DECIDE":
        return ("DECIDE", int(action[1]))
    if action[0] != "QUERY":
        raise ValueError(f"unknown action type {action[0]}")
    return ("QUERY", int(selected_query_indices[int(action[1])]))


def solve_typed_core(
    initial: Sequence[float],
    worlds: Sequence[World],
    models: Sequence[Sequence[int]],
    queries: Sequence[Query],
    *,
    horizon: int,
    false_allow: float = 2.0,
    false_block: float = 1.0,
) -> TypedCoreResult:
    """Run exact count-DP on the decision-bearing typed query core.

    Returned QUERY actions are mapped back to indices in the original vocabulary.
    """
    selected = decision_bearing_query_indices(queries)
    reduced_queries = tuple(queries[i] for i in selected)
    solver = EvidenceCountDP(
        initial,
        worlds,
        models,
        reduced_queries,
        horizon=horizon,
        false_allow=false_allow,
        false_block=false_block,
    )
    result: CountDPResult = solver.solve()
    action = map_reduced_action(result.action, selected)
    return TypedCoreResult(
        value=float(result.value),
        action=action,
        selected_query_indices=selected,
        reduced_query_count=len(selected),
        states=int(result.states),
        seconds=float(result.seconds),
    )
