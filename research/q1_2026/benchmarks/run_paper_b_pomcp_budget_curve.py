from __future__ import annotations

"""POMCP compute-budget curve on acquisition-active exact-solvable cases.

The unconditional random-prior validation family used previously is weak for testing
information acquisition because its exact root action is often DECIDE.  This analysis
therefore uses a mechanism-based, POMCP-independent case rule: balanced hidden-world
priors are generated and the first cases whose exact nominal-cost root action is
QUERY are retained.

All simulation budgets and POMCP seeds are fixed in advance and all are reported.
For each POMCP action we evaluate *true root-action regret* with the exact Bellman
root action-value table.  This separates Monte-Carlo value-estimation error from the
practical cost of choosing a non-optimal root action.
"""

import json
import statistics
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_paper_b_scalability_robustness import (
    MODELS,
    cost_sensitive_queries,
    make_balanced_worlds,
    uniform_belief,
)
from paper_b.count_dp import EvidenceCountDP
from paper_b.pomcp import StaticWorldPOMCP

OUT = ROOT / "results" / "paper_b_pomcp_budget_curve.json"
BUDGETS = (250, 500, 1000, 2500, 5000, 10000, 20000)
POMCP_SEEDS = (0, 1, 2, 3, 4)
N_CASES = 12
HORIZON = 3
MAX_CANDIDATE_SEEDS = 120


def exact_problem(candidate_seed: int):
    seed = 20261200 + int(candidate_seed)
    worlds = make_balanced_worlds(seed)
    belief = uniform_belief(worlds)
    queries = cost_sensitive_queries(1.0, 1.0)
    solver = EvidenceCountDP(belief, worlds, MODELS, queries, horizon=HORIZON)
    result = solver.solve()
    values = solver.root_action_values()
    optimum = min(values.values())
    optimal_actions = {a for a, value in values.items() if value <= optimum + 1e-12}
    ordered_values = sorted(float(v) for v in values.values())
    second_best_gap = float(ordered_values[1] - ordered_values[0]) if len(ordered_values) > 1 else 0.0
    return seed, worlds, belief, queries, result, optimal_actions, values, second_best_gap


def select_cases() -> tuple[list[tuple], int]:
    selected = []
    candidate_seed = 0
    while len(selected) < N_CASES and candidate_seed < MAX_CANDIDATE_SEEDS:
        problem = exact_problem(candidate_seed)
        candidate_seed += 1
        exact = problem[4]
        # Selection depends only on the exact problem, never on POMCP output.
        if exact.action[0] == "QUERY":
            selected.append(problem)
    if len(selected) < N_CASES:
        raise RuntimeError(
            f"only {len(selected)} acquisition-active cases found in {candidate_seed} candidates"
        )
    return selected, candidate_seed


def run_case(case_id: int, problem: tuple) -> dict:
    seed, worlds, belief, queries, exact, optimal_actions, exact_action_values, second_best_gap = problem
    rows = []
    optimum = float(min(exact_action_values.values()))
    for budget in BUDGETS:
        for pomcp_seed in POMCP_SEEDS:
            fixed_seed = 910000 + 100 * int(case_id) + int(pomcp_seed)
            t0 = perf_counter()
            approx = StaticWorldPOMCP(
                belief,
                worlds,
                MODELS,
                queries,
                horizon=HORIZON,
                seed=fixed_seed,
            ).solve(simulations=budget)
            seconds = perf_counter() - t0
            chosen_true_value = float(exact_action_values[approx.action])
            rows.append(
                {
                    "simulations": int(budget),
                    "pomcp_seed": int(pomcp_seed),
                    "pomcp_action": list(approx.action),
                    "action_is_exact_optimal": bool(approx.action in optimal_actions),
                    "action_family_matches_exact": bool(approx.action[0] == exact.action[0]),
                    "true_root_action_regret": float(chosen_true_value - optimum),
                    "absolute_pomcp_value_error": float(abs(approx.value - exact.value)),
                    "seconds": float(seconds),
                    "root_visits": int(approx.root_visits),
                }
            )

    return {
        "case": int(case_id),
        "source_seed": int(seed),
        "worlds": len(worlds),
        "queries": len(queries),
        "horizon": HORIZON,
        "exact_value": float(exact.value),
        "exact_canonical_action": list(exact.action),
        "exact_optimal_actions": [list(a) for a in sorted(optimal_actions)],
        "exact_second_best_gap": float(second_best_gap),
        "rows": rows,
    }


def summarize(cases: list[dict]) -> list[dict]:
    summary = []
    for budget in BUDGETS:
        rows = [
            row
            for case in cases
            for row in case["rows"]
            if row["simulations"] == budget
        ]
        estimate_errors = [float(r["absolute_pomcp_value_error"]) for r in rows]
        regrets = [float(r["true_root_action_regret"]) for r in rows]
        seconds = [float(r["seconds"]) for r in rows]
        optimal = [bool(r["action_is_exact_optimal"]) for r in rows]
        family = [bool(r["action_family_matches_exact"]) for r in rows]
        case_success = []
        for case in cases:
            case_rows = [r for r in case["rows"] if r["simulations"] == budget]
            case_success.append(sum(bool(r["action_is_exact_optimal"]) for r in case_rows) / len(case_rows))
        summary.append(
            {
                "simulations": int(budget),
                "cases": len(cases),
                "runs": len(rows),
                "exact_optimal_action_rate": float(sum(optimal) / len(optimal)),
                "query_vs_decide_family_accuracy": float(sum(family) / len(family)),
                "cases_with_majority_exact_action": int(sum(rate > 0.5 for rate in case_success)),
                "mean_true_root_action_regret": float(statistics.fmean(regrets)),
                "median_true_root_action_regret": float(statistics.median(regrets)),
                "max_true_root_action_regret": float(max(regrets)),
                "zero_regret_rate": float(sum(r <= 1e-12 for r in regrets) / len(regrets)),
                "mean_absolute_pomcp_value_error": float(statistics.fmean(estimate_errors)),
                "median_absolute_pomcp_value_error": float(statistics.median(estimate_errors)),
                "mean_seconds": float(statistics.fmean(seconds)),
                "median_seconds": float(statistics.median(seconds)),
            }
        )
    return summary


def main() -> None:
    selected, candidates_examined = select_cases()
    cases = [run_case(case_id, problem) for case_id, problem in enumerate(selected)]
    report = {
        "schema": "dovod-paper-b-pomcp-budget-curve-v3",
        "selection_rule": (
            "first 12 balanced-prior candidate seeds whose exact nominal-cost root action is QUERY"
        ),
        "candidate_seeds_examined": int(candidates_examined),
        "budgets": list(BUDGETS),
        "pomcp_seeds": list(POMCP_SEEDS),
        "cases": cases,
        "summary": summarize(cases),
        "prior_negative_diagnostic": (
            "An earlier unconditional 12-case budget run had DECIDE as the exact root action in every case. "
            "It remains a negative diagnostic showing that unconditional random-prior cases are not discriminating enough for acquisition-budget analysis."
        ),
        "analysis_contract": (
            "All seven budgets and five POMCP seeds were fixed before this run and every run is reported. "
            "Case inclusion depends only on balanced hidden-world construction and the exact root action being QUERY. "
            "True action regret is evaluated from exact Bellman root action values, not from POMCP's own value estimate."
        ),
        "claim_boundary": (
            "This controlled analysis measures solver behavior on acquisition-active, exact-solvable static hidden-world tasks. "
            "It can reveal approximation failures and near-ties, but does not establish an optimal simulation budget for other tasks, "
            "does not validate POMCP under likelihood misspecification, and does not imply failure of POMCP implementations outside this package."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "candidate_seeds_examined": report["candidate_seeds_examined"],
                "summary": report["summary"],
                "claim_boundary": report["claim_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
