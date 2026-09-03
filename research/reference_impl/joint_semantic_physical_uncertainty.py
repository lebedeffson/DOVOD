from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ActionUncertainty:
    action: int
    admissibility_probability: float
    total_variance: float
    semantic_variance: float
    physical_variance: float


@dataclass(frozen=True)
class InterventionValue:
    kind: str
    target: int
    before_loss: float
    expected_after_loss: float
    expected_reduction: float


def _normalize_graphs(graphs: Sequence[Mapping[int, Sequence[int]]]) -> list[dict[int, tuple[int, ...]]]:
    if not graphs:
        raise ValueError("At least one semantic graph is required")
    return [{int(a): tuple(sorted(map(int, preds))) for a, preds in graph.items()} for graph in graphs]


def _normalize_weights(n: int, weights: Sequence[float] | None) -> np.ndarray:
    if weights is None:
        return np.full(n, 1.0 / n, dtype=float)
    w = np.asarray(weights, dtype=float)
    if w.shape != (n,):
        raise ValueError((w.shape, n))
    if np.any(w < 0) or float(w.sum()) <= 0:
        raise ValueError("Semantic weights must be nonnegative with positive sum")
    return w / w.sum()


def _validate_q(q_complete: Sequence[float]) -> np.ndarray:
    q = np.asarray(q_complete, dtype=float)
    if q.ndim != 1:
        raise ValueError("Completion belief must be a 1-D vector")
    if np.any(q < -1e-12) or np.any(q > 1 + 1e-12):
        raise ValueError("Completion probabilities must lie in [0,1]")
    return np.clip(q, 0.0, 1.0)


def action_probability(q_complete: Sequence[float], action: int, graph: Mapping[int, Sequence[int]]) -> float:
    q = _validate_q(q_complete)
    a = int(action)
    if not 0 <= a < len(q):
        raise ValueError(a)
    p = 1.0 - float(q[a])
    for pred in graph.get(a, ()):  # type: ignore[arg-type]
        p *= float(q[int(pred)])
    return float(p)


def _joint_authorization_probability(q: np.ndarray, action: int, graph_a: Mapping[int, Sequence[int]], graph_b: Mapping[int, Sequence[int]]) -> float:
    preds = set(map(int, graph_a.get(action, ()))) | set(map(int, graph_b.get(action, ())))
    p = 1.0 - float(q[action])
    for pred in preds:
        p *= float(q[pred])
    return float(p)


def action_uncertainty(q_complete: Sequence[float], action: int, graphs: Sequence[Mapping[int, Sequence[int]]], weights: Sequence[float] | None = None) -> ActionUncertainty:
    q = _validate_q(q_complete)
    gs = _normalize_graphs(graphs)
    w = _normalize_weights(len(gs), weights)
    a = int(action)
    pg = np.asarray([action_probability(q, a, g) for g in gs], dtype=float)
    p = float(w @ pg)
    total = float(p * (1.0 - p))
    second = 0.0
    for i, gi in enumerate(gs):
        for j, gj in enumerate(gs):
            second += float(w[i]) * float(w[j]) * _joint_authorization_probability(q, a, gi, gj)
    semantic = float(np.clip(p - second, 0.0, total + 1e-12))
    physical = float(np.clip(total - semantic, 0.0, total + 1e-12))
    return ActionUncertainty(a, p, total, semantic, physical)


def global_uncertainty(q_complete: Sequence[float], graphs: Sequence[Mapping[int, Sequence[int]]], weights: Sequence[float] | None = None, actions: Iterable[int] | None = None, normalize: bool = True) -> dict[str, float]:
    q = _validate_q(q_complete)
    acts = list(range(len(q))) if actions is None else [int(a) for a in actions]
    rows = [action_uncertainty(q, a, graphs, weights) for a in acts]
    denom = float(len(rows)) if normalize and rows else 1.0
    return {
        "total": sum(r.total_variance for r in rows) / denom,
        "semantic": sum(r.semantic_variance for r in rows) / denom,
        "physical": sum(r.physical_variance for r in rows) / denom,
    }


def expected_after_perfect_physical_query(q_complete: Sequence[float], predicate: int, graphs: Sequence[Mapping[int, Sequence[int]]], weights: Sequence[float] | None = None, actions: Iterable[int] | None = None, normalize: bool = True) -> float:
    q = _validate_q(q_complete)
    j = int(predicate)
    pj = float(q[j])
    if pj <= 1e-15 or pj >= 1.0 - 1e-15:
        return global_uncertainty(q, graphs, weights, actions, normalize)["total"]
    q1 = q.copy(); q1[j] = 1.0
    q0 = q.copy(); q0[j] = 0.0
    l1 = global_uncertainty(q1, graphs, weights, actions, normalize)["total"]
    l0 = global_uncertainty(q0, graphs, weights, actions, normalize)["total"]
    return float(pj * l1 + (1.0 - pj) * l0)


def semantic_alternative_groups(graphs: Sequence[Mapping[int, Sequence[int]]], action: int) -> dict[tuple[int, ...], list[int]]:
    gs = _normalize_graphs(graphs)
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for idx, graph in enumerate(gs):
        groups[tuple(graph.get(int(action), ()))].append(idx)
    return dict(groups)


def expected_after_semantic_review(q_complete: Sequence[float], action: int, graphs: Sequence[Mapping[int, Sequence[int]]], weights: Sequence[float] | None = None, actions: Iterable[int] | None = None, normalize: bool = True) -> float:
    q = _validate_q(q_complete)
    gs = _normalize_graphs(graphs)
    w = _normalize_weights(len(gs), weights)
    groups = semantic_alternative_groups(gs, int(action))
    if len(groups) <= 1:
        return global_uncertainty(q, gs, w, actions, normalize)["total"]
    expected = 0.0
    for idxs in groups.values():
        mass = float(w[idxs].sum())
        if mass <= 0:
            continue
        subgraphs = [gs[i] for i in idxs]
        subweights = (w[idxs] / mass).tolist()
        expected += mass * global_uncertainty(q, subgraphs, subweights, actions, normalize)["total"]
    return float(expected)


def physical_intervention_value(q_complete, predicate, graphs, weights=None, actions=None, normalize=True) -> InterventionValue:
    before = global_uncertainty(q_complete, graphs, weights, actions, normalize)["total"]
    after = expected_after_perfect_physical_query(q_complete, predicate, graphs, weights, actions, normalize)
    return InterventionValue("PHYSICAL_QUERY", int(predicate), before, after, float(before - after))


def semantic_intervention_value(q_complete, action, graphs, weights=None, actions=None, normalize=True) -> InterventionValue:
    before = global_uncertainty(q_complete, graphs, weights, actions, normalize)["total"]
    after = expected_after_semantic_review(q_complete, action, graphs, weights, actions, normalize)
    return InterventionValue("SEMANTIC_REVIEW", int(action), before, after, float(before - after))


def best_physical_intervention(q_complete, candidates, graphs, weights=None, actions=None, normalize=True) -> InterventionValue | None:
    vals = [physical_intervention_value(q_complete, j, graphs, weights, actions, normalize) for j in candidates]
    vals = [v for v in vals if v.expected_reduction > 1e-15]
    return max(vals, key=lambda v: (v.expected_reduction, -v.target)) if vals else None


def best_semantic_intervention(q_complete, candidates, graphs, weights=None, actions=None, normalize=True) -> InterventionValue | None:
    vals = [semantic_intervention_value(q_complete, a, graphs, weights, actions, normalize) for a in candidates]
    vals = [v for v in vals if v.expected_reduction > 1e-15]
    return max(vals, key=lambda v: (v.expected_reduction, -v.target)) if vals else None
