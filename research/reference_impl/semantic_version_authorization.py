from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from hardening_loro_graph import action_set


@dataclass(frozen=True)
class SemanticDecision:
    certain_actions: tuple[int, ...]
    possible_actions: tuple[int, ...]
    ambiguous_actions: tuple[int, ...]


class SemanticVersionSpace:
    """Runtime semantics over an equivalence class of prerequisite graphs."""

    def __init__(self, graphs: list[dict[int, list[int]]]):
        if not graphs:
            raise ValueError("At least one graph is required")
        self.graphs = [{int(a): sorted(map(int, preds)) for a, preds in graph.items()} for graph in graphs]
        canonical = json.dumps(self.graphs, sort_keys=True, separators=(",", ":"))
        self.digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def evaluate(self, state: list[int]) -> SemanticDecision:
        sets = [set(action_set(state, graph)) for graph in self.graphs]
        possible = set().union(*sets)
        certain = set.intersection(*sets)
        return SemanticDecision(tuple(sorted(certain)), tuple(sorted(possible)), tuple(sorted(possible - certain)))

    def status(self, state: list[int], action: int) -> str:
        d = self.evaluate(state)
        if action in d.certain_actions: return "CERTAIN"
        if action in d.ambiguous_actions: return "SEMANTIC_REVIEW"
        return "BLOCKED"

    def alternatives(self, action: int) -> tuple[tuple[int, ...], ...]:
        return tuple(sorted({tuple(graph.get(action, [])) for graph in self.graphs}))

    def robust_prerequisites(self, action: int) -> tuple[int, ...]:
        return tuple(sorted({p for graph in self.graphs for p in graph.get(action, [])}))

    def issue_certificate(self, state: list[int], action: int) -> dict:
        return {
            "schema": "tinyapv-semantic-authorization-v1",
            "semantics_digest": self.digest,
            "action": int(action),
            "status": self.status(state, action),
            "robust_prerequisites": list(self.robust_prerequisites(action)),
            "alternative_prerequisite_sets": [list(x) for x in self.alternatives(action)],
        }

    def verify_certificate(self, state: list[int], certificate: dict) -> tuple[bool, str]:
        if certificate.get("schema") != "tinyapv-semantic-authorization-v1": return False, "schema"
        if certificate.get("semantics_digest") != self.digest: return False, "semantics_digest"
        action = int(certificate.get("action", -1))
        if not 0 <= action < len(state): return False, "action"
        expected = self.issue_certificate(state, action)
        for key in ("status", "robust_prerequisites", "alternative_prerequisite_sets"):
            if certificate.get(key) != expected[key]: return False, key
        if expected["status"] == "CERTAIN":
            if int(state[action]) == 1: return False, "already_complete"
            if not all(int(state[p]) == 1 for p in expected["robust_prerequisites"]): return False, "premise_state"
        return True, "ok"
