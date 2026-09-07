from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .amlgym_bridge import (
    DecisionObservation,
    encode_observation,
    parse_action_label,
    select_unlabeled_features,
    stable_bucket,
)
from .contextual_repair import RepairEdit, build_frozen_vocabulary, repaired_allows


@dataclass(frozen=True)
class SingleEditBaseline:
    operator: str
    selection: str
    feature_names: tuple[str, ...]
    vocabulary_size: int
    edit: RepairEdit | None
    repair_corrected_errors: int
    repair_introduced_errors: int

    @property
    def selected_edit_count(self) -> int:
        return int(self.edit is not None)


def _edit_key(edit: RepairEdit) -> str:
    context = ",".join(f"{idx}:{value}" for idx, value in edit.context)
    return f"{edit.kind}|{edit.prerequisite}|{context}"


def _candidate_space(
    observations: Sequence[DecisionObservation],
    *,
    max_features: int,
    max_context_width: int,
) -> tuple[tuple[str, ...], tuple[RepairEdit, ...]]:
    observations = tuple(observations)
    if not observations:
        raise ValueError("observations must be non-empty")
    features = select_unlabeled_features(observations, max_features=max_features)
    states = tuple(encode_observation(obs, features) for obs in observations)
    vocabulary = build_frozen_vocabulary(
        states,
        base_prerequisites=(0,),
        context_indices=tuple(range(1, len(features) + 1)),
        max_context_width=max_context_width,
        include_exceptions=True,
        include_guards=True,
    )
    return features, tuple(sorted(vocabulary, key=_edit_key))


def _predictions(
    observations: Sequence[DecisionObservation],
    feature_names: Sequence[str],
    edit: RepairEdit | None,
) -> tuple[int, ...]:
    if edit is None:
        return tuple(int(obs.base_allow) for obs in observations)
    return tuple(
        int(repaired_allows(encode_observation(obs, feature_names), (0,), (edit,)))
        for obs in observations
    )


def _error_change(
    observations: Sequence[DecisionObservation],
    predictions: Sequence[int],
) -> tuple[int, int]:
    corrected = 0
    introduced = 0
    for obs, pred in zip(observations, predictions):
        base_wrong = int(obs.base_allow) != int(obs.truth_allow)
        pred_wrong = int(pred) != int(obs.truth_allow)
        corrected += int(base_wrong and not pred_wrong)
        introduced += int((not base_wrong) and pred_wrong)
    return corrected, introduced


def fit_random_single_edit(
    operator: str,
    observations: Sequence[DecisionObservation],
    *,
    seed: int = 20260906,
    max_features: int = 8,
    max_context_width: int = 1,
) -> SingleEditBaseline:
    """Pick one label-free edit deterministically from edits that alter repair decisions."""
    observations = tuple(observations)
    if any(parse_action_label(obs.action_label)[0] != operator for obs in observations):
        raise ValueError("all observations must belong to operator")
    features, vocabulary = _candidate_space(
        observations, max_features=max_features, max_context_width=max_context_width
    )
    eligible = []
    for edit in vocabulary:
        preds = _predictions(observations, features, edit)
        if any(pred != int(obs.base_allow) for obs, pred in zip(observations, preds)):
            eligible.append(edit)
    if not eligible:
        return SingleEditBaseline(operator, "random_single_edit", features, len(vocabulary), None, 0, 0)
    index = stable_bucket(f"random-single-edit-v1|{seed}|{operator}", len(eligible))
    edit = eligible[index]
    corrected, introduced = _error_change(observations, _predictions(observations, features, edit))
    return SingleEditBaseline(
        operator, "random_single_edit", features, len(vocabulary), edit, corrected, introduced
    )


def fit_counterexample_frequency_single_edit(
    operator: str,
    observations: Sequence[DecisionObservation],
    *,
    max_features: int = 8,
    max_context_width: int = 1,
) -> SingleEditBaseline:
    """Choose at most one edit by repair-split counterexample frequency only.

    The score is corrected upstream errors minus newly introduced errors. Ties favor
    more corrections, fewer introductions, then a stable lexical edit key. If no
    edit has positive net gain the heuristic keeps the upstream decision unchanged.
    """
    observations = tuple(observations)
    if any(parse_action_label(obs.action_label)[0] != operator for obs in observations):
        raise ValueError("all observations must belong to operator")
    features, vocabulary = _candidate_space(
        observations, max_features=max_features, max_context_width=max_context_width
    )
    scored = []
    for edit in vocabulary:
        predictions = _predictions(observations, features, edit)
        corrected, introduced = _error_change(observations, predictions)
        net = corrected - introduced
        scored.append((-net, -corrected, introduced, _edit_key(edit), edit, corrected, introduced))
    if not scored:
        return SingleEditBaseline(
            operator, "counterexample_frequency_single_edit", features, 0, None, 0, 0
        )
    best = min(scored)
    net = -best[0]
    if net <= 0:
        return SingleEditBaseline(
            operator,
            "counterexample_frequency_single_edit",
            features,
            len(vocabulary),
            None,
            0,
            0,
        )
    edit, corrected, introduced = best[4], best[5], best[6]
    return SingleEditBaseline(
        operator,
        "counterexample_frequency_single_edit",
        features,
        len(vocabulary),
        edit,
        corrected,
        introduced,
    )


def predict_single_edit(model: SingleEditBaseline, observation: DecisionObservation) -> int:
    operator, _ = parse_action_label(observation.action_label)
    if operator != model.operator:
        raise ValueError("observation belongs to a different operator")
    return _predictions((observation,), model.feature_names, model.edit)[0]


def serialize_single_edit(model: SingleEditBaseline) -> dict:
    edit = model.edit
    return {
        "selection": model.selection,
        "feature_count": len(model.feature_names),
        "vocabulary_size": model.vocabulary_size,
        "selected_edit_count": model.selected_edit_count,
        "repair_corrected_errors": model.repair_corrected_errors,
        "repair_introduced_errors": model.repair_introduced_errors,
        "edit": None if edit is None else {
            "kind": edit.kind,
            "prerequisite": edit.prerequisite,
            "context": [list(x) for x in edit.context],
        },
    }
