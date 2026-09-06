from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_b.count_dp import EvidenceCountDP
from paper_b.orientation import bayes_error_after_calibration_and_direct, calibration_risk_gain
from paper_b.pomcp import StaticWorldPOMCP
from paper_b.static_world import Query, cartesian_worlds

OUT = ROOT / "results" / "paper_b_orientation_planning.json"


def oriented_case(r: float, query_cost: float = 0.01):
    models = ((0,),)
    worlds = cartesian_worlds(
        state_bits=1,
        models=models,
        physical_reliabilities=(r,),
        semantic_reliabilities=(0.8,),
        physical_orientations=(-1, 1),
    )
    belief = (1.0 / len(worlds),) * len(worlds)
    queries = (
        Query("calibrate-physical", "calibrate_physical", 0, query_cost),
        Query("direct-state", "state", 0, query_cost),
    )
    return worlds, belief, models, queries


def main() -> None:
    closed_form = []
    for r in (0.6, 0.7, 0.8, 0.9, 0.95):
        worlds, belief, models, queries = oriented_case(r, 0.01)
        solver = EvidenceCountDP(
            belief,
            worlds,
            models,
            queries,
            horizon=2,
            false_allow=1.0,
            false_block=1.0,
        )
        exact = solver.solve()
        expected_two_query = 0.02 + bayes_error_after_calibration_and_direct(r)
        expected_optimal = min(0.5, expected_two_query)
        closed_form.append(
            {
                "r": r,
                "exact_value": exact.value,
                "exact_action": list(exact.action),
                "closed_form_two_query_value": expected_two_query,
                "closed_form_optimal_value": expected_optimal,
                "abs_error": abs(exact.value - expected_optimal),
                "risk_gain_before_cost": calibration_risk_gain(r),
                "root_action_values": {
                    f"{a[0]}:{a[1]}": v for a, v in solver.root_action_values().items()
                },
            }
        )

    pomcp_rows = []
    for seed in range(5):
        r = 0.86
        worlds, belief, models, queries = oriented_case(r, 0.015)
        solver = EvidenceCountDP(
            belief,
            worlds,
            models,
            queries,
            horizon=2,
            false_allow=1.0,
            false_block=1.0,
        )
        exact = solver.solve()
        action_values = solver.root_action_values()
        optimum = min(action_values.values())
        optimal_actions = {a for a, value in action_values.items() if value <= optimum + 1e-12}
        approx = StaticWorldPOMCP(
            belief,
            worlds,
            models,
            queries,
            horizon=2,
            false_allow=1.0,
            false_block=1.0,
            seed=seed,
        ).solve(simulations=20000)
        pomcp_rows.append(
            {
                "seed": seed,
                "simulations": approx.simulations,
                "exact_value": exact.value,
                "pomcp_value": approx.value,
                "absolute_value_error": abs(approx.value - exact.value),
                "exact_canonical_action": list(exact.action),
                "exact_optimal_actions": [list(a) for a in sorted(optimal_actions)],
                "pomcp_action": list(approx.action),
                "pomcp_action_is_exact_optimal": approx.action in optimal_actions,
                "root_action_visits": [
                    [list(action), visits] for action, visits in approx.root_action_visits
                ],
            }
        )

    report = {
        "schema": "dovod-paper-b-orientation-planning-v2",
        "closed_form_regression": closed_form,
        "closed_form_all_match": all(row["abs_error"] < 1e-12 for row in closed_form),
        "pomcp": {
            "rows": pomcp_rows,
            "mean_absolute_value_error": sum(r["absolute_value_error"] for r in pomcp_rows)
            / len(pomcp_rows),
            "max_absolute_value_error": max(r["absolute_value_error"] for r in pomcp_rows),
            "exact_optimal_action_rate": sum(
                r["pomcp_action_is_exact_optimal"] for r in pomcp_rows
            )
            / len(pomcp_rows),
        },
        "claim_boundary": (
            "Orientation is part of the same hidden-world observation model used by the exact planner. "
            "The POMCP-style solver is an independently implemented approximate tree-search baseline in this package, not an external-library cross-check. "
            "Direct-first and calibration-first can be exactly tied because static evidence likelihoods commute; action scoring therefore uses membership in the exact optimal-action set."
        ),
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
