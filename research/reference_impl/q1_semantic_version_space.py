from __future__ import annotations

import itertools
from pathlib import Path

from hardening_loro_graph import action_set
from q1_decision_equivalent_graph import N

ROOT = Path(__file__).resolve().parent
VALUES = (-1, 0, 1)


def assemble_graphs(base: dict[int, list[int]], covers: dict[int, list[tuple[int, ...]]]) -> list[dict[int, list[int]]]:
    """Enumerate the complete minimum decision-equivalent version space."""
    actions = list(range(N))
    choices = [covers[a] for a in actions]
    graphs: list[dict[int, list[int]]] = []
    seen: set[tuple[tuple[int, tuple[int, ...]], ...]] = set()
    for selected in itertools.product(*choices):
        graph = {a: list(selected[a]) for a in actions}
        key = tuple((a, tuple(graph[a])) for a in actions)
        if key not in seen:
            seen.add(key)
            graphs.append(graph)
    return graphs


def conservative_union_graph(graphs: list[dict[int, list[int]]]) -> dict[int, list[int]]:
    """Conjunctive graph whose action set equals the intersection over versions."""
    return {a: sorted({p for graph in graphs for p in graph.get(a, [])}) for a in range(N)}


def semantic_sets(state: list[int], graphs: list[dict[int, list[int]]]) -> dict:
    sets = [frozenset(action_set(state, graph)) for graph in graphs]
    possible = set().union(*sets) if sets else set()
    certain = set.intersection(*(set(s) for s in sets)) if sets else set()
    unique = {tuple(sorted(s)) for s in sets}
    return {
        "possible": possible,
        "certain": certain,
        "ambiguous": possible - certain,
        "unique_action_sets": len(unique),
        "graph_disagreement": len(unique) > 1,
    }


def action_specific_alternatives(graphs: list[dict[int, list[int]]], action: int) -> tuple[tuple[int, ...], ...]:
    """Distinct prerequisite alternatives for one action across the version space."""
    return tuple(sorted({tuple(sorted(g.get(int(action), []))) for g in graphs}))
