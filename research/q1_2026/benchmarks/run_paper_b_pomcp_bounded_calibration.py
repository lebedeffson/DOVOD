from __future__ import annotations

"""Bounded POMCP exploration calibration with held-out confirmation.

This phase was opened after the acquisition-active budget diagnostic showed that the
package default exploration constant (1.25) did not recover the exact root action on
hard near-tie query problems.  The issue is treated as a protocol-visible solver
calibration question rather than hidden parameter chasing.

Protocol:
- construct the first 24 balanced-prior cases whose exact root action is QUERY;
- freeze the first 12 as development and the next 12 as confirmation;
- evaluate the predeclared exploration constants on development only;
- select by lowest mean TRUE root-action regret, then highest exact-action rate,
  then the smaller exploration constant;
- unlock confirmation once for the selected constant;
- report every development candidate and the held-out confirmation result.

The exact Bellman root action-value table supplies true root-action regret.  This is
controlled solver calibration, not evidence for dynamic-world or misspecified-model
performance.
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

OUT = ROOT / "results" / "paper_b_pomcp_bounded_calibration.json"
EXPLORATION_CANDIDATES = (1.25, 0.50, 0.20, 0.10, 0.05)
POMCP_SEEDS = (0, 1, 2)
SIMULATIONS = 15000
HORIZON = 3
N_DEVELOPMENT = 12
N_CONFIRMATION = 12
MAX_CANDIDATE_SEEDS = 240


def exact_problem(candidate_seed: int) -> dict:
    seed = 20262000 + int(candidate_seed)
    worlds = make_balanced_worlds(seed)
    belief = uniform_belief(worlds)
    queries = cost_sensitive_queries(1.0, 1.0)
    solver = EvidenceCountDP(belief, worlds, MODELS, queries, horizon=HORIZON)
    result = solver.solve()
    values = solver.root_action_values()
    optimum = min(values.values())
    return {
        "seed": seed,
        "worlds": worlds,
        "belief": belief,
        "queries": queries,
        "exact": result,
        "exact_values": values,
        "optimal_actions": {a for a, v in values.items() if v <= optimum + 1e-12},
        "optimum": float(optimum),
    }


def select_cases() -> tuple[list[dict], int]:
    selected = []
    candidate_seed = 0
    target = N_DEVELOPMENT + N_CONFIRMATION
    while len(selected) < target and candidate_seed < MAX_CANDIDATE_SEEDS:
        problem = exact_problem(candidate_seed)
        candidate_seed += 1
        if problem["exact"].action[0] == "QUERY":
            selected.append(problem)
    if len(selected) < target:
        raise RuntimeError(f"only {len(selected)} acquisition-active cases found")
    return selected, candidate_seed


def evaluate(split_cases: list[dict], exploration: float, split_name: str) -> dict:
    rows = []
    for case_id, problem in enumerate(split_cases):
        for pomcp_seed in POMCP_SEEDS:
            seed = 930000 + 1000 * (0 if split_name == "development" else 1) + 100 * case_id + pomcp_seed
            t0 = perf_counter()
            approx = StaticWorldPOMCP(
                problem["belief"],
                problem["worlds"],
                MODELS,
                problem["queries"],
                horizon=HORIZON,
                exploration=exploration,
                seed=seed,
            ).solve(simulations=SIMULATIONS)
            seconds = perf_counter() - t0
            true_value = float(problem["exact_values"][approx.action])
            regret = float(true_value - problem["optimum"])
            rows.append(
                {
                    "case": case_id,
                    "source_seed": int(problem["seed"]),
                    "pomcp_seed": pomcp_seed,
                    "action": list(approx.action),
                    "exact_action": list(problem["exact"].action),
                    "action_is_exact_optimal": bool(approx.action in problem["optimal_actions"]),
                    "query_vs_decide_family_accuracy": bool(approx.action[0] == problem["exact"].action[0]),
                    "true_root_action_regret": regret,
                    "seconds": float(seconds),
                }
            )

    regrets = [r["true_root_action_regret"] for r in rows]
    exact = [r["action_is_exact_optimal"] for r in rows]
    family = [r["query_vs_decide_family_accuracy"] for r in rows]
    return {
        "split": split_name,
        "exploration": float(exploration),
        "simulations": SIMULATIONS,
        "cases": len(split_cases),
        "runs": len(rows),
        "mean_true_root_action_regret": float(statistics.fmean(regrets)),
        "median_true_root_action_regret": float(statistics.median(regrets)),
        "max_true_root_action_regret": float(max(regrets)),
        "zero_regret_rate": float(sum(x <= 1e-12 for x in regrets) / len(regrets)),
        "exact_optimal_action_rate": float(sum(exact) / len(exact)),
        "query_vs_decide_family_accuracy": float(sum(family) / len(family)),
        "mean_seconds": float(statistics.fmean(r["seconds"] for r in rows)),
        "rows": rows,
    }


def main() -> None:
    cases, candidates_examined = select_cases()
    development = cases[:N_DEVELOPMENT]
    confirmation = cases[N_DEVELOPMENT:]

    development_results = [evaluate(development, c, "development") for c in EXPLORATION_CANDIDATES]
    selected = min(
        development_results,
        key=lambda row: (
            row["mean_true_root_action_regret"],
            -row["exact_optimal_action_rate"],
            row["exploration"],
        ),
    )
    selected_c = float(selected["exploration"])
    confirmation_result = evaluate(confirmation, selected_c, "confirmation")

    default_dev = next(r for r in development_results if r["exploration"] == 1.25)
    report = {
        "schema": "dovod-paper-b-pomcp-bounded-calibration-v1",
        "candidate_exploration_constants": list(EXPLORATION_CANDIDATES),
        "pomcp_seeds": list(POMCP_SEEDS),
        "simulations": SIMULATIONS,
        "selection_rule": (
            "minimum development mean true root-action regret; tie-break by higher exact-action rate and then smaller exploration constant"
        ),
        "case_rule": (
            "first 24 balanced-prior candidate seeds whose exact nominal-cost root action is QUERY; first 12 development, next 12 confirmation"
        ),
        "candidate_seeds_examined": int(candidates_examined),
        "development": development_results,
        "selected_exploration": selected_c,
        "default_development": {k: v for k, v in default_dev.items() if k != "rows"},
        "selected_development": {k: v for k, v in selected.items() if k != "rows"},
        "confirmation": confirmation_result,
        "analysis_contract": (
            "The candidate set, development/confirmation split, simulation budget, POMCP seeds, selection metric, tie-break, and case rule are fixed in this script. "
            "Confirmation performance is not used to revise the selected exploration constant."
        ),
        "claim_boundary": (
            "The result calibrates one implementation hyperparameter on controlled exact-solvable static cases. "
            "It is not a general POMCP hyperparameter recommendation and does not replace validation under different reward scales, horizons, or model misspecification."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_exploration": selected_c,
                "default_development": report["default_development"],
                "selected_development": report["selected_development"],
                "confirmation": {k: v for k, v in confirmation_result.items() if k != "rows"},
                "claim_boundary": report["claim_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
