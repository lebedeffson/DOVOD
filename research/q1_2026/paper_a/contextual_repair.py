from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

State = tuple[int, ...]
Context = tuple[tuple[int, int], ...]


@dataclass(frozen=True, order=True)
class RepairEdit:
    """One finite-vocabulary contextual repair edit."""
    kind: str
    context: Context
    prerequisite: int = -1
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in {"exception", "guard"}:
            raise ValueError("kind must be 'exception' or 'guard'")
        if self.kind == "exception" and self.prerequisite < 0:
            raise ValueError("exception requires a prerequisite index")
        if self.kind == "guard" and self.prerequisite != -1:
            raise ValueError("guard must not name a prerequisite")
        if self.weight <= 0:
            raise ValueError("weight must be positive")
        if tuple(sorted(self.context)) != self.context:
            raise ValueError("context literals must be sorted")
        if len({p for p, _ in self.context}) != len(self.context):
            raise ValueError("context cannot repeat a predicate")
        if any(v not in (0, 1) for _, v in self.context):
            raise ValueError("context values must be binary")


@dataclass(frozen=True)
class RepairSample:
    state: State
    allow: int

    def __post_init__(self) -> None:
        if self.allow not in (0, 1):
            raise ValueError("allow label must be binary")


def context_matches(state: State, context: Context) -> bool:
    return all(0 <= p < len(state) and int(state[p]) == int(v) for p, v in context)


def build_frozen_vocabulary(
    states: Sequence[State],
    base_prerequisites: Sequence[int],
    *,
    context_indices: Sequence[int] | None = None,
    max_context_width: int = 1,
    include_exceptions: bool = True,
    include_guards: bool = True,
) -> tuple[RepairEdit, ...]:
    """Construct a candidate edit vocabulary without using repair labels."""
    states = tuple(tuple(map(int, s)) for s in states)
    if not states:
        raise ValueError("states must be non-empty")
    n = len(states[0])
    if any(len(s) != n or any(v not in (0, 1) for v in s) for s in states):
        raise ValueError("states must have one common binary width")
    base = tuple(sorted(set(map(int, base_prerequisites))))
    if any(not 0 <= p < n for p in base):
        raise ValueError("base prerequisite index out of range")
    ctx_idx = tuple(range(n)) if context_indices is None else tuple(sorted(set(map(int, context_indices))))
    if any(not 0 <= p < n for p in ctx_idx):
        raise ValueError("context index out of range")
    width = int(max_context_width)
    if width < 0:
        raise ValueError("max_context_width must be non-negative")

    contexts: set[Context] = set()
    if width == 0:
        contexts.add(tuple())
    for k in range(1, min(width, len(ctx_idx)) + 1):
        for inds in combinations(ctx_idx, k):
            contexts.update({tuple((p, int(s[p])) for p in inds) for s in states})

    edits: set[RepairEdit] = set()
    for context in sorted(contexts):
        if include_exceptions:
            for p in base:
                edits.add(RepairEdit("exception", context, prerequisite=p))
        if include_guards:
            edits.add(RepairEdit("guard", context))
    return tuple(sorted(edits))


def _matching_exception_indices(state: State, prerequisite: int, vocabulary: Sequence[RepairEdit]) -> tuple[int, ...]:
    return tuple(
        i for i, edit in enumerate(vocabulary)
        if edit.kind == "exception" and edit.prerequisite == prerequisite and context_matches(state, edit.context)
    )


def _matching_guard_indices(state: State, vocabulary: Sequence[RepairEdit]) -> tuple[int, ...]:
    return tuple(i for i, edit in enumerate(vocabulary) if edit.kind == "guard" and context_matches(state, edit.context))


def repaired_allows(state: State, base_prerequisites: Sequence[int], selected_edits: Iterable[RepairEdit]) -> bool:
    selected = tuple(selected_edits)
    missing = [p for p in base_prerequisites if int(state[int(p)]) != 1]
    for p in missing:
        if not any(e.kind == "exception" and e.prerequisite == int(p) and context_matches(state, e.context) for e in selected):
            return False
    if any(e.kind == "guard" and context_matches(state, e.context) for e in selected):
        return False
    return True


def _validate_inputs(
    base_prerequisites: Sequence[int], vocabulary: Sequence[RepairEdit], samples: Sequence[RepairSample]
) -> tuple[tuple[int, ...], tuple[RepairEdit, ...], tuple[RepairSample, ...]]:
    base = tuple(sorted(set(map(int, base_prerequisites))))
    vocab = tuple(vocabulary)
    samples = tuple(samples)
    if not samples:
        raise ValueError("samples must be non-empty")
    n = len(samples[0].state)
    if any(len(s.state) != n for s in samples):
        raise ValueError("sample state lengths differ")
    if any(not 0 <= p < n for p in base):
        raise ValueError("base prerequisite index out of range")
    for edit in vocab:
        if edit.kind == "exception" and edit.prerequisite not in base:
            raise ValueError("exception prerequisite is not in base model")
        if any(not 0 <= p < n for p, _ in edit.context):
            raise ValueError("edit context index out of range")
    return base, vocab, samples


@dataclass(frozen=True)
class SoftRepairResult:
    selected_indices: tuple[int, ...]
    selected_edits: tuple[RepairEdit, ...]
    objective: float
    predictions: tuple[int, ...]
    error_indices: tuple[int, ...]
    false_allow_errors: int
    false_block_errors: int


def solve_contextual_repair_soft_milp(
    base_prerequisites: Sequence[int],
    vocabulary: Sequence[RepairEdit],
    samples: Sequence[RepairSample],
    *,
    false_allow_weight: float = 1.0,
    false_block_weight: float = 1.0,
) -> SoftRepairResult:
    """Solve finite-vocabulary bidirectional repair with explicit label slacks."""
    if false_allow_weight <= 0 or false_block_weight <= 0:
        raise ValueError("error weights must be positive")
    base, vocab, samples = _validate_inputs(base_prerequisites, vocabulary, samples)
    m, ns = len(vocab), len(samples)
    aux: list[tuple[int, int, tuple[int, ...]]] = []
    for si, sample in enumerate(samples):
        if sample.allow == 0:
            for p in base:
                if int(sample.state[p]) != 1:
                    aux.append((si, p, _matching_exception_indices(sample.state, p, vocab)))

    err_offset = m
    z_offset = m + ns
    z_index = {(si, p): z_offset + k for k, (si, p, _) in enumerate(aux)}
    nvar = m + ns + len(aux)
    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    def add(coeffs: dict[int, float], lo: float, hi: float) -> None:
        row = np.zeros(nvar, dtype=float)
        for idx, val in coeffs.items():
            row[int(idx)] = float(val)
        rows.append(row)
        lower.append(float(lo))
        upper.append(float(hi))

    for si, sample in enumerate(samples):
        state = sample.state
        ei = err_offset + si
        if sample.allow == 1:
            for p in base:
                if int(state[p]) == 1:
                    continue
                covers = _matching_exception_indices(state, p, vocab)
                if covers:
                    coeffs = {i: 1.0 for i in covers}
                    coeffs[ei] = 1.0
                    add(coeffs, 1.0, np.inf)
                else:
                    add({ei: 1.0}, 1.0, 1.0)
            for gi in _matching_guard_indices(state, vocab):
                add({gi: 1.0, ei: -1.0}, -np.inf, 0.0)
        else:
            block_causes: dict[int, float] = {ei: 1.0}
            for p in base:
                if int(state[p]) == 1:
                    continue
                covers = _matching_exception_indices(state, p, vocab)
                zi = z_index[(si, p)]
                block_causes[zi] = 1.0
                if not covers:
                    add({zi: 1.0}, 1.0, 1.0)
                else:
                    for edit_idx in covers:
                        add({zi: 1.0, edit_idx: 1.0}, -np.inf, 1.0)
                    coeffs = {zi: 1.0}
                    coeffs.update({edit_idx: 1.0 for edit_idx in covers})
                    add(coeffs, 1.0, np.inf)
            for gi in _matching_guard_indices(state, vocab):
                block_causes[gi] = block_causes.get(gi, 0.0) + 1.0
            add(block_causes, 1.0, np.inf)

    A = np.vstack(rows) if rows else np.zeros((0, nvar), dtype=float)
    c = np.zeros(nvar, dtype=float)
    c[:m] = [e.weight for e in vocab]
    for si, sample in enumerate(samples):
        c[err_offset + si] = false_block_weight if sample.allow == 1 else false_allow_weight
    res = milp(
        c=c,
        integrality=np.ones(nvar, dtype=int),
        bounds=Bounds(np.zeros(nvar), np.ones(nvar)),
        constraints=LinearConstraint(A, np.asarray(lower), np.asarray(upper)),
    )
    if not res.success or res.x is None:
        raise RuntimeError(f"soft repair MILP failed: {res.message}")

    selected_indices = tuple(i for i, x in enumerate(res.x[:m]) if x >= 0.5)
    selected_edits = tuple(vocab[i] for i in selected_indices)
    predictions = tuple(int(repaired_allows(s.state, base, selected_edits)) for s in samples)
    error_indices = tuple(i for i, (pred, sample) in enumerate(zip(predictions, samples)) if pred != sample.allow)
    false_allow = sum(pred == 1 and samples[i].allow == 0 for i, pred in enumerate(predictions))
    false_block = sum(pred == 0 and samples[i].allow == 1 for i, pred in enumerate(predictions))
    exact_objective = float(sum(e.weight for e in selected_edits))
    exact_objective += false_allow_weight * false_allow + false_block_weight * false_block
    if abs(float(res.fun) - exact_objective) > 1e-7:
        raise AssertionError((res.fun, exact_objective))
    return SoftRepairResult(
        selected_indices=selected_indices,
        selected_edits=selected_edits,
        objective=exact_objective,
        predictions=predictions,
        error_indices=error_indices,
        false_allow_errors=int(false_allow),
        false_block_errors=int(false_block),
    )
