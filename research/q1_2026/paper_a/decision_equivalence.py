from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class SelectionResult:
    selected: tuple[int, ...]
    objective: int


def _normalize_violation_sets(violation_sets: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    rows = tuple(tuple(sorted(set(map(int, r)))) for r in violation_sets)
    if any(len(r) == 0 for r in rows):
        raise ValueError("a blocked state with no violated prerequisite cannot be preserved")
    return rows


def solve_hitting_set_bruteforce(
    n_prerequisites: int, violation_sets: Sequence[Sequence[int]]
) -> SelectionResult:
    rows = _normalize_violation_sets(violation_sets)
    for k in range(n_prerequisites + 1):
        for comb in combinations(range(n_prerequisites), k):
            chosen = set(comb)
            if all(chosen.intersection(r) for r in rows):
                return SelectionResult(tuple(comb), k)
    raise RuntimeError("infeasible")


def solve_hitting_set_milp(
    n_prerequisites: int, violation_sets: Sequence[Sequence[int]]
) -> SelectionResult:
    rows = _normalize_violation_sets(violation_sets)
    A = np.zeros((len(rows), n_prerequisites), dtype=float)
    for i, r in enumerate(rows):
        for j in r:
            if not 0 <= j < n_prerequisites:
                raise ValueError("prerequisite index out of range")
            A[i, j] = 1.0
    res = milp(
        c=np.ones(n_prerequisites),
        integrality=np.ones(n_prerequisites, dtype=int),
        bounds=Bounds(np.zeros(n_prerequisites), np.ones(n_prerequisites)),
        constraints=LinearConstraint(A, np.ones(len(rows)), np.full(len(rows), np.inf)),
    )
    if not res.success or res.x is None:
        raise RuntimeError(f"MILP failed: {res.message}")
    selected = tuple(int(i) for i, x in enumerate(res.x) if x >= 0.5)
    return SelectionResult(selected, len(selected))


def classify_optimum_family(
    n_prerequisites: int, violation_sets: Sequence[Sequence[int]]
) -> dict[str, tuple[int, ...]]:
    rows = _normalize_violation_sets(violation_sets)
    optimum = solve_hitting_set_bruteforce(n_prerequisites, rows).objective
    optima = []
    for comb in combinations(range(n_prerequisites), optimum):
        chosen = set(comb)
        if all(chosen.intersection(r) for r in rows):
            optima.append(set(comb))
    all_idx = set(range(n_prerequisites))
    mandatory = set.intersection(*optima) if optima else set()
    optional = set.union(*optima) - mandatory if optima else set()
    redundant = all_idx - (mandatory | optional)
    return {
        "mandatory": tuple(sorted(mandatory)),
        "optional_optimal": tuple(sorted(optional)),
        "redundant": tuple(sorted(redundant)),
    }
