from __future__ import annotations

"""POMCP compute-budget curve on the frozen exact-solvable static-world cases.

This is a diagnostic compute/accuracy analysis, not a tuning loop.  The simulation
budgets are fixed in advance and every budget is reported.  Each case reuses the
same POMCP random seed across budgets, so lower-budget trajectories are prefixes of
higher-budget trajectories for that case.  Exact Bellman planning supplies the
reference optimal-action set.
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
    make_worlds,
    query_pool,
    uniform_belief,
)
from paper_b.count_dp import EvidenceCountDP
from paper_b.pomcp import StaticWorldPOMCP

OUT = ROOT / "results" / "paper_b_pomcp_budget_curve.json"
BUDGETS = (250, 500, 1000, 2500, 5000, 10000, 20000)
N_CASES = 12
HORIZON = 3
Q_COUNT = 9


def run_case(case_id: int) -> dict:
    worlds = make_worlds(20260907 + int(case_id))
    belief = uniform_belief(worlds)
    queries = query_pool()[:Q_COUNT]

    exact_solver = EvidenceCountDP(belief, worlds, MODELS, queries, horizon=HORIZON)
    exact = exact_solver.solve()
    action_values = exact_solver.root_action_values()
    optimum = min(action_values.values())
    optimal_actions = {a for a, value in action_values.items() if value <= optimum + 1e-12}

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
    cases = [run_case(case_id) for case_id in range(N_CASES)]
    report = {
        "schema": "dovod-paper-b-pomcp-budget-curve-v1",
        "budgets": list(BUDGETS),
        "cases": cases,
        "summary": summarize(cases),
        "analysis_contract": (
            "All seven simulation budgets were fixed before this run and are reported; no budget is selected from test performance. "
            "The 12 hidden-world cases, query count, horizon, and exact reference construction match the existing exact-vs-POMCP validation family."
        ),
        "claim_boundary": (
            "This controlled curve measures compute/accuracy behavior of the package POMCP implementation on exact-solvable static hidden-world tasks. "
            "It does not establish an optimal simulation budget for other tasks and does not validate POMCP under model misspecification or dynamic hidden states."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "claim_boundary": report["claim_boundary"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
