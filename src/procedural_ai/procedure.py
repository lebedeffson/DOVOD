from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

Graph = dict[int, list[int]]
Record = tuple[str, str, list[tuple[str, list[int]]]]


def action_set(state: Sequence[int], graph: Graph) -> list[int]:
    """Return actions admissible under a prerequisite graph.

    A component/action is admissible when it is not already complete and all
    encoded prerequisites are complete. This is a model-level decision rule,
    not a mechanical-safety certification.
    """
    return [
        action
        for action in sorted(graph)
        if int(state[action]) != 1
        and all(int(state[pred]) == 1 for pred in graph[action])
    ]


def transitioned_actions(before: Sequence[int], after: Sequence[int]) -> list[int]:
    """Actions/components that transitioned into the completed state."""
    return [
        i for i, (a, b) in enumerate(zip(before, after))
        if int(a) != 1 and int(b) == 1
    ]


def next_action_events(records: Iterable[Record], *, split: str | None = None):
    """Extract observed next-action events from state trajectories."""
    events = []
    for record_split, recording, rows in records:
        if split is not None and record_split != split:
            continue
        for k in range(len(rows) - 1):
            state, nxt = rows[k][1], rows[k + 1][1]
            actual = transitioned_actions(state, nxt)
            if actual:
                events.append((recording, rows[k][0], list(state), actual))
    return events


def infer_unary_prerequisites(
    records: Iterable[Record],
    *,
    n_components: int,
    selected_recordings: Sequence[str] | None = None,
    split: str = "train",
    support: float = 1.0,
) -> Graph:
    """Infer unary, unconditional, same-state prerequisite candidates.

    For each observed transition into completion of an action, a predicate is
    retained when it was complete in at least ``support`` fraction of those
    pre-action states. The method is one-sided: later counterexamples can
    falsify a candidate, while absence of counterexamples does not prove
    physical necessity.
    """
    selected = set(selected_recordings) if selected_recordings is not None else None
    pre_states: dict[int, list[list[int]]] = defaultdict(list)
    for record_split, recording, rows in records:
        if record_split != split:
            continue
        if selected is not None and recording not in selected:
            continue
        for k in range(len(rows) - 1):
            state, nxt = rows[k][1], rows[k + 1][1]
            for action in transitioned_actions(state, nxt):
                pre_states[action].append(list(state))

    graph: Graph = {a: [] for a in range(n_components)}
    for action in range(n_components):
        states = pre_states[action]
        if not states:
            continue
        for pred in range(n_components):
            if pred == action:
                continue
            observed_support = sum(int(s[pred]) == 1 for s in states) / len(states)
            if observed_support >= support:
                graph[action].append(pred)
    return graph
