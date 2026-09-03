from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .procedure import action_set, Graph


@dataclass(frozen=True)
class SemanticDecision:
    certain_actions: tuple[int, ...]
    possible_actions: tuple[int, ...]
    ambiguous_actions: tuple[int, ...]


class SemanticVersionSpace:
    """Finite family of procedure graphs consistent with frozen evidence."""

    def __init__(self, graphs: list[Graph]):
        if not graphs:
            raise ValueError("At least one graph is required")
        self.graphs = [
            {int(a): sorted(map(int, preds)) for a, preds in graph.items()}
            for graph in graphs
        ]
        canonical = json.dumps(self.graphs, sort_keys=True, separators=(",", ":"))
        self.digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def evaluate(self, state: list[int]) -> SemanticDecision:
        sets = [set(action_set(state, graph)) for graph in self.graphs]
        possible = set().union(*sets)
        certain = set.intersection(*sets)
        return SemanticDecision(
            certain_actions=tuple(sorted(certain)),
            possible_actions=tuple(sorted(possible)),
            ambiguous_actions=tuple(sorted(possible - certain)),
        )

    def alternatives(self, action: int) -> tuple[tuple[int, ...], ...]:
        return tuple(sorted({tuple(graph.get(action, [])) for graph in self.graphs}))

    def robust_prerequisites(self, action: int) -> tuple[int, ...]:
        return tuple(sorted({p for graph in self.graphs for p in graph.get(action, [])}))
