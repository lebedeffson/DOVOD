from __future__ import annotations

"""POMCP compute-budget curve on acquisition-active exact-solvable cases.

The first diagnostic version used the unconditional random-prior cases from the
existing exact-vs-POMCP check.  Their exact root action was DECIDE in all 12 cases,
which makes action recovery too easy and therefore weak evidence for an acquisition
planner.  This version keeps that negative diagnostic in the project history and
uses a mechanism-based selection rule already used by the cost-robustness study:
construct balanced hidden-world priors and retain the first cases whose exact
nominal-cost root action is QUERY.

Selection inspects only the exact root action, never POMCP performance.  Simulation
budgets are fixed in advance and every budget is reported, so the curve is a
compute/accuracy diagnostic rather than a tuning loop.
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
    return seed, worlds, belief, queries, result, optimal_actions


def select_cases() -> tuple[list[tuple], int]:
    selected = []
    candidate_seed = 0
    while len(selected) < N_CASES and candidate_seed < MAX_CANDIDATE_SEEDS:
        problem = exact_problem(candidate_seed)
        candidate_seed += 1
        exact = problem[4]
        # Mechanism-based selection: acquisition must actually be useful at root.
        if exact.action[0] == "QUERY":
            selected.append(problem)
    if len(selected) < N_CASES:
        raise RuntimeError(
            f"only {len(selected)} acquisition-active cases found in {candidate_seed} candidates"
        )
    return selected, candidate_seed


def run_case(case_id: int, problem: tuple) -> dict:
    seed, worlds, belief, queries, exact, optimal_actions = problem
    rows = []
    fixed_seed = 910000 + int(case_id)
    for budget in BUDGETS:
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
        rows.append(
            {
                "simulations": int(budget),
                "pomcp_action": list(approx.action),
                "action_is_exact_optimal": bool(approx.action in optimal_actions),
                "absolute_value_error": float(abs(approx.value - exact.value)),
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
        "rows": rows,
    }


def summarize(cases: list[dict]) -> list[dict]:
    summary = []
    for budget in BUDGETS:
        rows = [next(r for r in case["rows"] if r["simulations"] == budget) for case in cases]
        errors = [float(r["absolute_value_error"]) for r in rows]
        seconds = [float(r["seconds"]) for r in rows]
        actions = [bool(r["action_is_exact_optimal"]) for r in rows]
        summary.append(
            {
                "simulations": int(budget),
                "cases": len(rows),
                "exact_optimal_action_rate": float(sum(actions) / len(actions)),
                "exact_optimal_action_cases": int(sum(actions)),
                "mean_absolute_value_error": float(statistics.fmean(errors)),
                "median_absolute_value_error": float(statistics.median(errors)),
                "max_absolute_value_error": float(max(errors)),
                "mean_seconds": float(statistics.fmean(seconds)),
                "median_seconds": float(statistics.median(seconds)),
            }
        )
    return summary


def main() -> None:
    selected, candidates_examined = select_cases()
    cases = [run_case(case_id, problem) for case_id, problem in enumerate(selected)]
    report = {
        "schema": "dovod-paper-b-pomcp-budget-curve-v2",
        "selection_rule": (
            "first 12 balanced-prior candidate seeds whose exact nominal-cost root action is QUERY"
        ),
        "candidate_seeds_examined": int(candidates_examined),
        "budgets": list(BUDGETS),
        "cases": cases,
        "summary": summarize(cases),
        "prior_negative_diagnostic": (
            "An earlier unconditional 12-case budget run had DECIDE as the exact root action in every case. "
            "It is retained in workflow history as evidence that the unconditional random-prior family is not discriminating enough for acquisition-budget analysis."
        ),
        "analysis_contract": (
            "All seven simulation budgets were fixed before the acquisition-active run and are reported; no budget is selected from POMCP performance. "
            "Case inclusion depends only on balanced hidden-world construction and the exact nominal-cost root action being QUERY, matching the pre-existing cost-robustness selection logic."
        ),
        "claim_boundary": (
            "This controlled curve measures compute/accuracy behavior of the package POMCP implementation on acquisition-active exact-solvable static hidden-world tasks. "
            "It does not establish an optimal simulation budget for other tasks and does not validate POMCP under model misspecification or dynamic hidden states."
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
