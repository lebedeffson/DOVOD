from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

State = tuple[int, ...]
Model = tuple[int, ...]


@dataclass(frozen=True)
class NonidentifiabilityWitness:
    candidate: int
    required_model: Model
    free_model: Model
    counterfactual_state: State


@dataclass(frozen=True)
class PredicateIdentifiability:
    predicate: int
    role: str
    models_with: int
    models_without: int


def allows(state: State, model: Model) -> bool:
    return all(int(state[p]) == 1 for p in model)


def passive_positive_nonidentifiability_witness(
    positive_states: Sequence[State],
    base_prerequisites: Sequence[int],
    candidate: int,
) -> NonidentifiabilityWitness:
    base = tuple(sorted(set(map(int, base_prerequisites))))
    candidate = int(candidate)
    if candidate not in base:
        raise ValueError("candidate must be a base prerequisite")
    if not positive_states:
        raise ValueError("at least one positive state is required")
    n = len(positive_states[0])
    if any(len(s) != n for s in positive_states):
        raise ValueError("state lengths differ")
    if any(not allows(s, base) for s in positive_states):
        raise ValueError("all supplied observations must be successful under required model")
    free = tuple(p for p in base if p != candidate)
    cf = [1] * n
    cf[candidate] = 0
    counterfactual = tuple(cf)
    assert not allows(counterfactual, base)
    assert allows(counterfactual, free)
    return NonidentifiabilityWitness(candidate, base, free, counterfactual)


def enumerate_consistent_models(
    n_predicates: int,
    positive_states: Sequence[State],
    negative_states: Sequence[State],
    *,
    max_size: int | None = None,
) -> tuple[Model, ...]:
    if n_predicates < 0:
        raise ValueError("n_predicates must be non-negative")
    if any(len(s) != n_predicates for s in tuple(positive_states) + tuple(negative_states)):
        raise ValueError("state length mismatch")
    cap = n_predicates if max_size is None else min(int(max_size), n_predicates)
    out: list[Model] = []
    for k in range(cap + 1):
        for comb in combinations(range(n_predicates), k):
            if all(allows(s, comb) for s in positive_states) and all(
                not allows(s, comb) for s in negative_states
            ):
                out.append(tuple(comb))
    return tuple(out)


def classify_predicates(
    n_predicates: int, version_space: Sequence[Model]
) -> tuple[PredicateIdentifiability, ...]:
    if not version_space:
        raise ValueError("empty version space")
    rows = []
    for p in range(n_predicates):
        yes = sum(p in m for m in version_space)
        no = len(version_space) - yes
        role = "mandatory" if no == 0 else ("excluded" if yes == 0 else "ambiguous")
        rows.append(PredicateIdentifiability(p, role, yes, no))
    return tuple(rows)


def disagreement_states(m1: Model, m2: Model, states: Iterable[State]) -> tuple[State, ...]:
    return tuple(s for s in states if allows(s, m1) != allows(s, m2))
