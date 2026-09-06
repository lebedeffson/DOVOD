from __future__ import annotations

from paper_a.amlgym_bridge import DecisionObservation
from paper_a.deployment import calibration_gate_from_predictions
from paper_a.protocol import (
    assert_stage_disjoint,
    confirmatory_states_excluding_pilot,
    pilot_index_ranked_states,
    semantic_state_fingerprint,
    state_split,
)


def _obs(truth, base):
    return tuple(
        DecisionObservation(
            state_literals=(f"(s {i})",),
            action_label="(op x)",
            base_allow=b,
            truth_allow=y,
        )
        for i, (y, b) in enumerate(zip(truth, base))
    )


def test_gate_requires_strict_calibration_gain():
    obs = _obs((0, 1, 1, 0), (0, 0, 1, 1))
    gate = calibration_gate_from_predictions(
        "op", obs, (0, 1, 0, 1), selected_edit_count=1
    )
    assert gate.deployed is False
    assert gate.reason == "no_strict_calibration_gain"


def test_gate_accepts_strict_gain_without_false_allow_regression():
    obs = _obs((0, 0, 1, 1), (0, 1, 0, 1))
    gate = calibration_gate_from_predictions(
        "op", obs, (0, 0, 1, 1), selected_edit_count=1
    )
    assert gate.deployed is True
    assert gate.reason == "strict_calibration_improvement"
    assert gate.candidate_metrics["risk"] < gate.base_metrics["risk"]
    assert gate.candidate_metrics["false_allows"] <= gate.base_metrics["false_allows"]


def test_gate_rejects_lower_total_risk_if_false_allows_increase():
    obs = _obs((0, 1, 1, 1), (0, 0, 0, 1))
    gate = calibration_gate_from_predictions(
        "op", obs, (1, 1, 1, 1), selected_edit_count=1
    )
    assert gate.candidate_metrics["risk"] < gate.base_metrics["risk"]
    assert gate.candidate_metrics["false_allows"] > gate.base_metrics["false_allows"]
    assert gate.deployed is False
    assert gate.reason == "false_allow_regression"


def test_empty_exact_noop_is_bookkept_as_deployed():
    obs = _obs((0, 1), (0, 1))
    gate = calibration_gate_from_predictions(
        "op", obs, (0, 1), selected_edit_count=0
    )
    assert gate.deployed is True
    assert gate.reason == "empty_noop"


def test_semantic_fingerprint_ignores_literal_order_and_duplicates():
    a = semantic_state_fingerprint(("(p a)", "(q b)", "(p a)"))
    b = semantic_state_fingerprint(("(q b)", "(p a)"))
    assert a == b


def test_pilot_selector_reproduces_index_hash_and_keeps_actual_indices():
    states = [(f"(p {i})",) for i in range(30)]
    selected = pilot_index_ranked_states(
        states, domain="d", problem_name="p", limit=12
    )
    assert len(selected) == 12
    assert len({x.original_index for x in selected}) == 12
    assert selected == pilot_index_ranked_states(
        states, domain="d", problem_name="p", limit=12
    )


def test_confirmatory_excludes_actual_pilot_and_all_semantic_duplicates():
    unique = [(f"(p {i})",) for i in range(40)]
    states = unique + [tuple(reversed(s)) for s in unique[:5]]
    pilot = pilot_index_ranked_states(
        states, domain="d", problem_name="p", limit=12
    )
    confirmatory = confirmatory_states_excluding_pilot(
        states,
        domain="d",
        problem_name="p",
        states_per_problem=12,
        pilot_states_per_problem=12,
    )
    assert len(confirmatory) == 12
    assert_stage_disjoint(pilot, confirmatory)
    assert not ({x.fingerprint for x in pilot} & {x.fingerprint for x in confirmatory})
    assert len({x.fingerprint for x in confirmatory}) == len(confirmatory)


def test_confirmatory_selection_ignores_labels_and_actions_by_api():
    states = [(f"(p {i})",) for i in range(30)]
    a = confirmatory_states_excluding_pilot(
        states, domain="d", problem_name="p", states_per_problem=12,
        pilot_states_per_problem=12,
    )
    b = confirmatory_states_excluding_pilot(
        states, domain="d", problem_name="p", states_per_problem=12,
        pilot_states_per_problem=12,
    )
    assert a == b


def test_state_split_is_action_independent_and_problem_scoped():
    fingerprint = semantic_state_fingerprint(("(p a)", "(q b)"))
    a = state_split(domain="d", problem_name="p1", fingerprint=fingerprint)
    b = state_split(domain="d", problem_name="p1", fingerprint=fingerprint)
    assert a == b
    assert a in {"repair", "calibration", "test"}
