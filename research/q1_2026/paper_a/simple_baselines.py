from __future__ import annotations

from dataclasses import dataclass
import hashlib
from random import Random
from typing import Sequence

from .amlgym_bridge import (
    DecisionObservation,
    action_local_features,
    parse_action_label,
    select_unlabeled_features,
)


@dataclass(frozen=True)
class OneEditPolicy:
    operator: str
    mode: str
    feature: str | None
    objective: float

    def __post_init__(self) -> None:
        allowed = {
            "identity",
            "exception_if_present",
            "exception_if_absent",
            "guard_if_present",
            "guard_if_absent",
        }
        if self.mode not in allowed:
            raise ValueError(f"unknown one-edit mode: {self.mode}")
        if self.mode == "identity" and self.feature is not None:
            raise ValueError("identity policy cannot carry a feature")
        if self.mode != "identity" and not self.feature:
            raise ValueError("non-identity policy requires a feature")


def _condition(observation: DecisionObservation, feature: str, *, present: bool) -> bool:
    active = feature in action_local_features(observation.state_literals, observation.action_label)
    return active if present else not active


def predict_one_edit(policy: OneEditPolicy, observation: DecisionObservation) -> int:
    operator, _ = parse_action_label(observation.action_label)
    if operator != policy.operator:
        raise ValueError("observation belongs to a different operator")
    base = int(observation.base_allow)
    if policy.mode == "identity":
        return base
    assert policy.feature is not None
    if policy.mode == "exception_if_present":
        return 1 if base == 0 and _condition(observation, policy.feature, present=True) else base
    if policy.mode == "exception_if_absent":
        return 1 if base == 0 and _condition(observation, policy.feature, present=False) else base
    if policy.mode == "guard_if_present":
        return 0 if base == 1 and _condition(observation, policy.feature, present=True) else base
    return 0 if base == 1 and _condition(observation, policy.feature, present=False) else base


def _weighted_error(
    observations: Sequence[DecisionObservation],
    predictions: Sequence[int],
    *,
    false_allow_weight: float,
    false_block_weight: float,
) -> float:
    score = 0.0
    for observation, prediction in zip(observations, predictions):
        p = int(prediction)
        if p == 1 and observation.truth_allow == 0:
            score += float(false_allow_weight)
        elif p == 0 and observation.truth_allow == 1:
            score += float(false_block_weight)
    return score


def candidate_one_edit_policies(
    operator: str,
    observations: Sequence[DecisionObservation],
    *,
    max_features: int = 8,
) -> tuple[OneEditPolicy, ...]:
    observations = tuple(observations)
    if not observations:
        raise ValueError("observations must be non-empty")
    if any(parse_action_label(o.action_label)[0] != operator for o in observations):
        raise ValueError("all observations must belong to operator")
    features = select_unlabeled_features(observations, max_features=max_features)
    policies = [OneEditPolicy(operator, "identity", None, 0.0)]
    for feature in features:
        for mode in (
            "exception_if_present",
            "exception_if_absent",
            "guard_if_present",
            "guard_if_absent",
        ):
            policies.append(OneEditPolicy(operator, mode, feature, 0.0))
    return tuple(policies)


def fit_frequency_one_edit(
    operator: str,
    observations: Sequence[DecisionObservation],
    *,
    max_features: int = 8,
    edit_penalty: float = 0.25,
    false_allow_weight: float = 1.0,
    false_block_weight: float = 1.0,
) -> OneEditPolicy:
    """Fit a deliberately simple contingency-count heuristic on repair data.

    The baseline may make at most one action-local exception/guard. Candidate
    triggers are frozen without labels; the selected trigger is the one with
    lowest weighted repair error plus one edit penalty. This uses only
    one-feature frequency/error counts and no MILP or context conjunctions.
    """

    observations = tuple(observations)
    candidates = candidate_one_edit_policies(operator, observations, max_features=max_features)
    scored = []
    for candidate in candidates:
        predictions = tuple(predict_one_edit(candidate, o) for o in observations)
        objective = _weighted_error(
            observations,
            predictions,
            false_allow_weight=false_allow_weight,
            false_block_weight=false_block_weight,
        )
        if candidate.mode != "identity":
            objective += float(edit_penalty)
        priority = 0 if candidate.mode == "identity" else 1
        scored.append((float(objective), priority, candidate.mode, candidate.feature or "", candidate))
    objective, _, _, _, chosen = min(scored)
    return OneEditPolicy(chosen.operator, chosen.mode, chosen.feature, objective)


def random_one_edit(
    operator: str,
    observations: Sequence[DecisionObservation],
    *,
    seed_key: str,
    max_features: int = 8,
) -> OneEditPolicy:
    """Choose one non-identity edit from the same frozen one-feature class.

    Truth labels do not influence the random choice. The seed is derived from a
    stable text key so CI reruns are deterministic across Python processes.
    """

    observations = tuple(observations)
    candidates = tuple(
        p for p in candidate_one_edit_policies(operator, observations, max_features=max_features)
        if p.mode != "identity"
    )
    if not candidates:
        return OneEditPolicy(operator, "identity", None, 0.0)
    digest = hashlib.sha256(seed_key.encode("utf-8")).digest()
    rng = Random(int.from_bytes(digest[:8], "big"))
    chosen = candidates[rng.randrange(len(candidates))]
    return OneEditPolicy(chosen.operator, chosen.mode, chosen.feature, 0.0)
