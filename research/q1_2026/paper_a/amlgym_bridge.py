from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from .contextual_repair import (
    RepairEdit,
    RepairSample,
    SoftRepairResult,
    build_frozen_vocabulary,
    repaired_allows,
    solve_contextual_repair_soft_milp,
)

_LITERAL = re.compile(r"^\(([^()\s]+)(?:\s+([^()]+?))?\)$")


@dataclass(frozen=True)
class DecisionObservation:
    """One upstream-vs-reference action-applicability observation."""

    state_literals: tuple[str, ...]
    action_label: str
    base_allow: int
    truth_allow: int

    def __post_init__(self) -> None:
        if self.base_allow not in (0, 1) or self.truth_allow not in (0, 1):
            raise ValueError("base_allow and truth_allow must be binary")


@dataclass(frozen=True)
class OperatorRepair:
    operator: str
    feature_names: tuple[str, ...]
    vocabulary: tuple[RepairEdit, ...]
    fit: SoftRepairResult


@dataclass(frozen=True)
class GlobalDecisionBaseline:
    """Operator-local non-contextual comparator selected on repair data only."""

    operator: str
    policy: str
    objective: float

    def __post_init__(self) -> None:
        if self.policy not in {"identity", "always_allow", "always_block"}:
            raise ValueError("unknown global baseline policy")


def parse_action_label(label: str) -> tuple[str, tuple[str, ...]]:
    text = label.strip()
    if not (text.startswith("(") and text.endswith(")")):
        raise ValueError(f"invalid action label: {label!r}")
    parts = text[1:-1].split()
    if not parts:
        raise ValueError(f"empty action label: {label!r}")
    return parts[0], tuple(parts[1:])


def _normalize_literal(literal: str, action_args: Sequence[str]) -> str | None:
    text = literal.strip()
    if text.startswith("(not "):
        return None
    match = _LITERAL.match(text)
    if not match:
        return None
    predicate = match.group(1)
    args = tuple((match.group(2) or "").split())
    action_map = {obj: f"?{index}" for index, obj in enumerate(action_args)}
    if args and not any(obj in action_map for obj in args):
        return None
    mapped = tuple(action_map.get(obj, "*") for obj in args)
    return f"{predicate}({','.join(mapped)})" if mapped else f"{predicate}()"


def action_local_features(state_literals: Iterable[str], action_label: str) -> frozenset[str]:
    _, args = parse_action_label(action_label)
    features = set()
    for literal in state_literals:
        feature = _normalize_literal(str(literal), args)
        if feature is not None:
            features.add(feature)
    return frozenset(features)


def select_unlabeled_features(
    observations: Sequence[DecisionObservation],
    *,
    max_features: int = 10,
    min_frequency: float = 0.05,
    max_frequency: float = 0.95,
) -> tuple[str, ...]:
    """Freeze context features without using reference applicability labels."""

    if max_features < 0:
        raise ValueError("max_features must be non-negative")
    n = len(observations)
    if n == 0 or max_features == 0:
        return tuple()
    counts: dict[str, int] = {}
    for observation in observations:
        for feature in action_local_features(observation.state_literals, observation.action_label):
            counts[feature] = counts.get(feature, 0) + 1
    candidates = []
    for feature, count in counts.items():
        frequency = count / n
        if min_frequency <= frequency <= max_frequency:
            candidates.append((abs(frequency - 0.5), feature))
    candidates.sort()
    return tuple(feature for _, feature in candidates[:max_features])


def encode_observation(
    observation: DecisionObservation,
    feature_names: Sequence[str],
) -> tuple[int, ...]:
    present = action_local_features(observation.state_literals, observation.action_label)
    return (int(observation.base_allow),) + tuple(int(name in present) for name in feature_names)


def _weighted_vocabulary(
    vocabulary: Sequence[RepairEdit],
    edit_penalty: float,
) -> tuple[RepairEdit, ...]:
    if edit_penalty <= 0:
        raise ValueError("edit_penalty must be positive")
    return tuple(
        RepairEdit(edit.kind, edit.context, prerequisite=edit.prerequisite, weight=float(edit_penalty))
        for edit in vocabulary
    )


def fit_operator_repair(
    operator: str,
    observations: Sequence[DecisionObservation],
    *,
    max_features: int = 10,
    max_context_width: int = 1,
    edit_penalty: float = 0.25,
    false_allow_weight: float = 1.0,
    false_block_weight: float = 1.0,
) -> OperatorRepair:
    observations = tuple(observations)
    if not observations:
        raise ValueError("observations must be non-empty")
    if any(parse_action_label(o.action_label)[0] != operator for o in observations):
        raise ValueError("all observations must belong to operator")

    features = select_unlabeled_features(observations, max_features=max_features)
    states = tuple(encode_observation(o, features) for o in observations)
    raw_vocabulary = build_frozen_vocabulary(
        states,
        base_prerequisites=(0,),
        context_indices=tuple(range(1, len(features) + 1)),
        max_context_width=max_context_width,
        include_exceptions=True,
        include_guards=True,
    )
    vocabulary = _weighted_vocabulary(raw_vocabulary, edit_penalty)
    samples = tuple(RepairSample(state, o.truth_allow) for state, o in zip(states, observations))
    fit = solve_contextual_repair_soft_milp(
        (0,),
        vocabulary,
        samples,
        false_allow_weight=false_allow_weight,
        false_block_weight=false_block_weight,
    )
    return OperatorRepair(operator, features, vocabulary, fit)


def predict_operator_repair(model: OperatorRepair, observation: DecisionObservation) -> int:
    state = encode_observation(observation, model.feature_names)
    return int(repaired_allows(state, (0,), model.fit.selected_edits))


def fit_global_decision_baseline(
    operator: str,
    observations: Sequence[DecisionObservation],
    *,
    override_penalty: float = 0.25,
    false_allow_weight: float = 1.0,
    false_block_weight: float = 1.0,
) -> GlobalDecisionBaseline:
    """Fit identity/always-allow/always-block on the repair split only."""

    observations = tuple(observations)
    if not observations:
        raise ValueError("observations must be non-empty")
    if override_penalty < 0:
        raise ValueError("override_penalty must be non-negative")
    if false_allow_weight < 0 or false_block_weight < 0:
        raise ValueError("error weights must be non-negative")
    if any(parse_action_label(o.action_label)[0] != operator for o in observations):
        raise ValueError("all observations must belong to operator")

    policies = {
        "identity": tuple(o.base_allow for o in observations),
        "always_allow": tuple(1 for _ in observations),
        "always_block": tuple(0 for _ in observations),
    }
    priority = {"identity": 0, "always_block": 1, "always_allow": 2}
    scored = []
    for policy, predictions in policies.items():
        false_allows = sum(
            p == 1 and o.truth_allow == 0 for p, o in zip(predictions, observations)
        )
        false_blocks = sum(
            p == 0 and o.truth_allow == 1 for p, o in zip(predictions, observations)
        )
        objective = false_allow_weight * false_allows + false_block_weight * false_blocks
        if policy != "identity":
            objective += float(override_penalty)
        scored.append((float(objective), priority[policy], policy))
    objective, _, policy = min(scored)
    return GlobalDecisionBaseline(operator=operator, policy=policy, objective=objective)


def predict_global_decision_baseline(
    model: GlobalDecisionBaseline,
    observation: DecisionObservation,
) -> int:
    operator, _ = parse_action_label(observation.action_label)
    if operator != model.operator:
        raise ValueError("observation belongs to a different operator")
    if model.policy == "identity":
        return int(observation.base_allow)
    if model.policy == "always_allow":
        return 1
    return 0


def decision_metrics(
    observations: Sequence[DecisionObservation],
    predictions: Sequence[int],
) -> dict[str, float | int | None]:
    observations = tuple(observations)
    predictions = tuple(map(int, predictions))
    if len(observations) != len(predictions):
        raise ValueError("observations/predictions length mismatch")
    if not observations:
        return {
            "n": 0,
            "risk": 0.0,
            "class_balanced_risk": None,
            "false_allow_rate": 0.0,
            "false_block_rate": 0.0,
            "false_allows": 0,
            "false_blocks": 0,
            "positives": 0,
            "negatives": 0,
        }

    false_allows = sum(
        prediction == 1 and observation.truth_allow == 0
        for prediction, observation in zip(predictions, observations)
    )
    false_blocks = sum(
        prediction == 0 and observation.truth_allow == 1
        for prediction, observation in zip(predictions, observations)
    )
    negatives = sum(observation.truth_allow == 0 for observation in observations)
    positives = sum(observation.truth_allow == 1 for observation in observations)
    false_allow_rate = false_allows / negatives if negatives else 0.0
    false_block_rate = false_blocks / positives if positives else 0.0
    balanced = (
        0.5 * (false_allow_rate + false_block_rate)
        if negatives and positives
        else None
    )
    return {
        "n": len(observations),
        "risk": (false_allows + false_blocks) / len(observations),
        "class_balanced_risk": balanced,
        "false_allow_rate": false_allow_rate,
        "false_block_rate": false_block_rate,
        "false_allows": false_allows,
        "false_blocks": false_blocks,
        "positives": positives,
        "negatives": negatives,
    }


def stable_bucket(key: str, modulus: int = 1000) -> int:
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus
