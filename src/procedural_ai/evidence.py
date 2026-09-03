from __future__ import annotations

from collections.abc import Iterable, Sequence

from .procedure import Graph, transitioned_actions


def prune_with_counterexample(graph: Graph, rows: Sequence[tuple[str, list[int]]]) -> Graph:
    """Remove prerequisite candidates contradicted by an observed successful action."""
    out = {action: list(preds) for action, preds in graph.items()}
    for k in range(len(rows) - 1):
        state, nxt = rows[k][1], rows[k + 1][1]
        for action in transitioned_actions(state, nxt):
            out[action] = [pred for pred in out[action] if int(state[pred]) == 1]
    return out


def edge_set(graph: Graph) -> set[tuple[int, int]]:
    return {(pred, action) for action, preds in graph.items() for pred in preds}


def carrier_survival_fraction(carrier_counts: Iterable[int], deletions: int) -> float:
    """Fraction of observed refutations guaranteed to survive arbitrary carrier loss."""
    counts = [int(x) for x in carrier_counts]
    if not counts:
        return 0.0
    return sum(c > deletions for c in counts) / len(counts)
