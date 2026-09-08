from __future__ import annotations

from benchmarks.run_paper_b_likelihood_misspecification import (
    evaluate_assumed_policy_metrics_under_true_model,
    transform_reliability,
)
from paper_b.count_dp import EvidenceCountDP
from paper_b.static_world import Query, World


def test_alpha_one_preserves_reliability():
    for r in (0.5, 0.6, 0.75, 0.9, 1.0):
        assert abs(transform_reliability(r, 1.0) - r) < 1e-12


def test_underconfidence_moves_toward_chance():
    assert transform_reliability(0.9, 0.5) == 0.7
    assert transform_reliability(0.7, 0.5) == 0.6


def test_overconfidence_is_clipped_to_valid_range():
    assert transform_reliability(0.95, 1.25) == 1.0
    assert 0.5 <= transform_reliability(0.6, 1.25) <= 1.0


def test_operational_metrics_decompose_query_and_terminal_cost():
    models = ((0,),)
    worlds = (
        World((0,), 0, 1.0, 1.0),
        World((1,), 0, 1.0, 1.0),
    )
    belief = (0.5, 0.5)
    queries = (Query("inspect-state", "state", 0, 0.1),)
    true_solver = EvidenceCountDP(belief, worlds, models, queries, horizon=1)
    assumed_solver = EvidenceCountDP(belief, worlds, models, queries, horizon=1)

    result = true_solver.solve()
    assert result.action == ("QUERY", 0)

    metrics = evaluate_assumed_policy_metrics_under_true_model(true_solver, assumed_solver)
    assert abs(metrics["expected_total_cost"] - 0.1) < 1e-12
    assert abs(metrics["expected_acquisition_cost"] - 0.1) < 1e-12
    assert abs(metrics["expected_query_count"] - 1.0) < 1e-12
    assert abs(metrics["expected_terminal_decision_loss"]) < 1e-12
    assert abs(metrics["wrong_decision_probability"]) < 1e-12
    assert abs(metrics["false_allow_probability"]) < 1e-12
    assert abs(metrics["false_block_probability"]) < 1e-12
