from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .amlgym_bridge import (
    DecisionObservation,
    OperatorRepair,
    decision_metrics,
    parse_action_label,
    predict_operator_repair,
)


@dataclass(frozen=True)
class CalibrationGateDecision:
    operator: str
    deployed: bool
    reason: str
    base_metrics: dict[str, float | int | None]
    candidate_metrics: dict[str, float | int | None]
    selected_edit_count: int


def calibration_gate_from_predictions(
    operator: str,
    observations: Sequence[DecisionObservation],
    candidate_predictions: Sequence[int],
    *,
    selected_edit_count: int,
    atol: float = 1e-15,
    require_false_allow_nonincrease: bool = True,
) -> CalibrationGateDecision:
    """Select a candidate repair using calibration data only.

    A non-empty repair is deployed only when it strictly lowers calibration
    misclassification risk and, by default, does not increase the number of
    false allows.  Ties are rejected.  An empty repair that reproduces the
    upstream decisions is retained as a no-op for bookkeeping.
    """

    observations = tuple(observations)
    candidate_predictions = tuple(map(int, candidate_predictions))
    if selected_edit_count < 0:
        raise ValueError("selected_edit_count must be non-negative")
    if len(observations) != len(candidate_predictions):
        raise ValueError("observations/predictions length mismatch")
    if not observations:
        empty = decision_metrics((), ())
        return CalibrationGateDecision(
            operator=operator,
            deployed=False,
            reason="no_calibration_samples",
            base_metrics=empty,
            candidate_metrics=empty,
            selected_edit_count=selected_edit_count,
        )

    if any(parse_action_label(o.action_label)[0] != operator for o in observations):
        raise ValueError("all calibration observations must belong to operator")

    base_predictions = tuple(o.base_allow for o in observations)
    base = decision_metrics(observations, base_predictions)
    candidate = decision_metrics(observations, candidate_predictions)

    if selected_edit_count == 0 and candidate_predictions == base_predictions:
        return CalibrationGateDecision(
            operator=operator,
            deployed=True,
            reason="empty_noop",
            base_metrics=base,
            candidate_metrics=candidate,
            selected_edit_count=0,
        )

    strict_risk_gain = float(candidate["risk"]) < float(base["risk"]) - atol
    false_allow_safe = (
        not require_false_allow_nonincrease
        or int(candidate["false_allows"]) <= int(base["false_allows"])
    )

    if not strict_risk_gain:
        deployed = False
        reason = "no_strict_calibration_gain"
    elif not false_allow_safe:
        deployed = False
        reason = "false_allow_regression"
    else:
        deployed = True
        reason = "strict_calibration_improvement"

    return CalibrationGateDecision(
        operator=operator,
        deployed=deployed,
        reason=reason,
        base_metrics=base,
        candidate_metrics=candidate,
        selected_edit_count=selected_edit_count,
    )


def choose_operator_calibration_gates(
    rows: Sequence[tuple[str, DecisionObservation]],
    candidates: dict[str, OperatorRepair],
    *,
    require_false_allow_nonincrease: bool = True,
) -> tuple[dict[str, OperatorRepair], dict[str, CalibrationGateDecision]]:
    """Gate each operator independently on the calibration split."""

    rows = tuple(rows)
    deployed: dict[str, OperatorRepair] = {}
    decisions: dict[str, CalibrationGateDecision] = {}

    for operator, model in sorted(candidates.items()):
        calibration = tuple(
            obs
            for split, obs in rows
            if split == "calibration" and parse_action_label(obs.action_label)[0] == operator
        )
        predictions = tuple(predict_operator_repair(model, obs) for obs in calibration)
        decision = calibration_gate_from_predictions(
            operator,
            calibration,
            predictions,
            selected_edit_count=len(model.fit.selected_edits),
            require_false_allow_nonincrease=require_false_allow_nonincrease,
        )
        decisions[operator] = decision
        if decision.deployed:
            deployed[operator] = model

    return deployed, decisions


def gated_prediction(
    observation: DecisionObservation,
    deployed_repairs: dict[str, OperatorRepair],
) -> int:
    operator, _ = parse_action_label(observation.action_label)
    model = deployed_repairs.get(operator)
    if model is None:
        return int(observation.base_allow)
    return predict_operator_repair(model, observation)
