from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class StageState:
    original_index: int
    fingerprint: str
    state_literals: tuple[str, ...]


def canonical_state_literals(state_literals: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(map(str, state_literals))))


def semantic_state_fingerprint(state_literals: Iterable[str]) -> str:
    """Object-sensitive fingerprint of an exact symbolic state.

    Literal order and duplicate literals are ignored. Object names are retained;
    this is an exact symbolic-state identity check, not an isomorphism heuristic.
    """

    canonical = canonical_state_literals(state_literals)
    payload = "\n".join(canonical).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stable_bucket(key: str, modulus: int) -> int:
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def pilot_index_ranked_states(
    states: Sequence[Iterable[str]],
    *,
    domain: str,
    problem_name: str,
    limit: int,
) -> tuple[StageState, ...]:
    """Reproduce the selector used by the inspected AMLGym pilot exactly.

    The pilot runner ranked *indices* using
    ``stable_bucket('state|domain|problem|index', 2**31-1)`` and selected the
    first ``limit`` entries. Reproducing that selector is necessary to exclude
    the states that were actually inspected, rather than an imagined pilot set.
    """

    if limit < 0:
        raise ValueError("limit must be non-negative")
    ranked = sorted(
        enumerate(states),
        key=lambda item: _stable_bucket(
            f"state|{domain}|{problem_name}|{item[0]}", 2**31 - 1
        ),
    )
    out = []
    for index, state in ranked[: min(limit, len(ranked))]:
        literals = tuple(map(str, state))
        out.append(
            StageState(
                original_index=index,
                fingerprint=semantic_state_fingerprint(literals),
                state_literals=canonical_state_literals(literals),
            )
        )
    return tuple(out)


def _confirmatory_rank_key(domain: str, problem_name: str, fingerprint: str) -> bytes:
    payload = f"confirmatory-state-rank-v1|{domain}|{problem_name}|{fingerprint}".encode(
        "utf-8"
    )
    return hashlib.sha256(payload).digest()


def confirmatory_states_excluding_pilot(
    states: Sequence[Iterable[str]],
    *,
    domain: str,
    problem_name: str,
    states_per_problem: int,
    pilot_states_per_problem: int,
    excluded_fingerprints: Iterable[str] = (),
) -> tuple[StageState, ...]:
    """Select a new semantic-state window disjoint from the actual pilot.

    All semantic duplicates of any state selected by the inspected pilot are
    removed. Remaining states are deduplicated by exact symbolic fingerprint and
    ranked without labels or actions. The returned set therefore cannot contain
    a state semantically identical to an inspected pilot state.
    """

    if states_per_problem < 0 or pilot_states_per_problem < 0:
        raise ValueError("state limits must be non-negative")

    pilot = pilot_index_ranked_states(
        states,
        domain=domain,
        problem_name=problem_name,
        limit=pilot_states_per_problem,
    )
    excluded = {item.fingerprint for item in pilot}
    excluded.update(map(str, excluded_fingerprints))

    first_by_fingerprint: dict[str, StageState] = {}
    for index, state in enumerate(states):
        literals = tuple(map(str, state))
        fingerprint = semantic_state_fingerprint(literals)
        if fingerprint in excluded:
            continue
        first_by_fingerprint.setdefault(
            fingerprint,
            StageState(
                original_index=index,
                fingerprint=fingerprint,
                state_literals=canonical_state_literals(literals),
            ),
        )

    ranked = sorted(
        first_by_fingerprint.values(),
        key=lambda item: (
            _confirmatory_rank_key(domain, problem_name, item.fingerprint),
            item.original_index,
        ),
    )
    return tuple(ranked[: min(states_per_problem, len(ranked))])


def state_split(
    *,
    domain: str,
    problem_name: str,
    fingerprint: str,
    repair_end: int = 500,
    calibration_end: int = 750,
    modulus: int = 1000,
) -> str:
    """Assign an entire semantic state to repair/calibration/test.

    Problem name is included so the unit is a concrete symbolic state within a
    benchmark problem. Action identity is intentionally absent, preventing
    actions from the same state from crossing splits.
    """

    if not (0 <= repair_end <= calibration_end <= modulus):
        raise ValueError("invalid split boundaries")
    bucket = _stable_bucket(
        f"confirmatory-state-split-v1|{domain}|{problem_name}|{fingerprint}", modulus
    )
    if bucket < repair_end:
        return "repair"
    if bucket < calibration_end:
        return "calibration"
    return "test"


def assert_stage_disjoint(
    pilot: Sequence[StageState],
    confirmatory: Sequence[StageState],
) -> None:
    pilot_fingerprints = {item.fingerprint for item in pilot}
    confirmatory_fingerprints = {item.fingerprint for item in confirmatory}
    overlap = pilot_fingerprints.intersection(confirmatory_fingerprints)
    if overlap:
        raise AssertionError(
            f"pilot/confirmatory semantic overlap: {len(overlap)} states"
        )
