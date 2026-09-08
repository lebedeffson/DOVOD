from __future__ import annotations

"""Likelihood-misspecification stress test for exact information acquisition.

The environment is kept fixed while the planner is given a systematically
miscalibrated reliability model.  We evaluate the *entire assumed-model policy*
under the true observation model, not only its first action.

For each hidden world, assumed source reliability is transformed as

    r_assumed = clip(0.5 + alpha * (r_true - 0.5), 0.5, 1.0).

alpha < 1 means the planner underestimates source informativeness; alpha > 1
means it is overconfident.  The levels are fixed in advance and all are reported.
Cases are the first 12 balanced-prior exact problems whose true optimal root action
is QUERY.  Case selection never inspects misspecification performance.
"""

import json
import statistics
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

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
from paper_b.static_world import World

OUT = ROOT / "results" / "paper_b_likelihood_misspecification.json"
ALPHAS = (0.50, 0.75, 1.00, 1.25)
N_CASES = 12
HORIZON = 3
MAX_CANDIDATE_SEEDS = 120


def transform_reliability(r: float, alpha: float) -> float:
    return float(min(1.0, max(0.5, 0.5 + float(alpha) * (float(r) - 0.5))))


def assumed_worlds(worlds: tuple[World, ...], alpha: float) -> tuple[World, ...]:
    return tuple(
        World(
            state=w.state,
            model=w.model,
            physical_reliability=transform_reliability(w.physical_reliability, alpha),
            semantic_reliability=transform_reliability(w.semantic_reliability, alpha),
            physical_orientation=w.physical_orientation,
            semantic_orientation=w.semantic_orientation,
        )
        for w in worlds
    )


def exact_problem(candidate_seed: int):
    seed = 20261200 + int(candidate_seed)
    worlds = make_balanced_worlds(seed)
    belief = uniform_belief(worlds)
    queries = cost_sensitive_queries(1.0, 1.0)
    true_solver = EvidenceCountDP(belief, worlds, MODELS, queries, horizon=HORIZON)
    true_result = true_solver.solve()
    return seed, worlds, belief, queries, true_solver, true_result


def select_cases() -> tuple[list[tuple], int]:
    selected = []
    candidate_seed = 0
    while len(selected) < N_CASES and candidate_seed < MAX_CANDIDATE_SEEDS:
        problem = exact_problem(candidate_seed)
        candidate_seed += 1
        if problem[5].action[0] == "QUERY":
            selected.append(problem)
    if len(selected) < N_CASES:
        raise RuntimeError(
            f"only {len(selected)} acquisition-active cases found in {candidate_seed} candidates"
        )
    return selected, candidate_seed


def evaluate_assumed_policy_under_true_model(
    true_solver: EvidenceCountDP,
    assumed_solver: EvidenceCountDP,
) -> float:
    """Expected true cost of the policy induced by assumed_solver."""

    @lru_cache(maxsize=None)
    def value(counts: tuple[int, ...]) -> float:
        true_belief = np.asarray(true_solver.belief_from_counts(counts), dtype=float)
        _, action = assumed_solver.V(counts)
        if action[0] == "DECIDE":
            return float(true_solver._loss[int(action[1])] @ true_belief)

        qi = int(action[1])
        total = float(true_solver.queries[qi].cost)
        for obs in (0, 1):
            po = float(true_solver._lik[qi, obs] @ true_belief)
            if po > 1e-14:
                total += po * value(true_solver._inc(counts, qi, obs))
        return float(total)

    return float(value(true_solver.zero_counts))


def run_case(case_id: int, problem: tuple) -> dict:
    seed, worlds, belief, queries, true_solver, true_result = problem
    rows = []
    for alpha in ALPHAS:
        aw = assumed_worlds(tuple(worlds), alpha)
        assumed_solver = EvidenceCountDP(belief, aw, MODELS, queries, horizon=HORIZON)
        assumed_result = assumed_solver.solve()
        true_policy_value = evaluate_assumed_policy_under_true_model(true_solver, assumed_solver)
        rows.append(
            {
                "alpha": float(alpha),
                "assumed_root_action": list(assumed_result.action),
                "root_action_matches_true_optimum": bool(assumed_result.action == true_result.action),
                "root_action_family_matches": bool(assumed_result.action[0] == true_result.action[0]),
                "assumed_model_value": float(assumed_result.value),
                "true_value_of_assumed_policy": float(true_policy_value),
                "full_policy_regret_under_true_model": float(true_policy_value - true_result.value),
            }
        )

    return {
        "case": int(case_id),
        "source_seed": int(seed),
        "true_optimal_value": float(true_result.value),
        "true_root_action": list(true_result.action),
        "rows": rows,
    }


def summarize(cases: list[dict]) -> list[dict]:
    out = []
    for alpha in ALPHAS:
        rows = [next(r for r in c["rows"] if r["alpha"] == alpha) for c in cases]
        regrets = [float(r["full_policy_regret_under_true_model"]) for r in rows]
        root = [bool(r["root_action_matches_true_optimum"]) for r in rows]
        family = [bool(r["root_action_family_matches"]) for r in rows]
        out.append(
            {
                "alpha": float(alpha),
                "cases": len(rows),
                "root_exact_action_rate": float(sum(root) / len(root)),
                "root_query_vs_decide_family_accuracy": float(sum(family) / len(family)),
                "zero_policy_regret_rate": float(sum(abs(x) <= 1e-12 for x in regrets) / len(regrets)),
                "mean_full_policy_regret": float(statistics.fmean(regrets)),
                "median_full_policy_regret": float(statistics.median(regrets)),
                "max_full_policy_regret": float(max(regrets)),
            }
        )
    return out


def main() -> None:
    selected, candidates_examined = select_cases()
    cases = [run_case(i, problem) for i, problem in enumerate(selected)]
    summary = summarize(cases)

    # Exact-model policy evaluation is a required internal sanity check.
    exact_row = next(row for row in summary if row["alpha"] == 1.0)
    if abs(exact_row["max_full_policy_regret"]) > 1e-10:
        raise AssertionError(f"alpha=1 policy evaluator is inconsistent: {exact_row}")

    report = {
        "schema": "dovod-paper-b-likelihood-misspecification-v1",
        "alphas": list(ALPHAS),
        "selection_rule": (
            "first 12 balanced-prior candidate seeds whose true exact nominal-cost root action is QUERY"
        ),
        "candidate_seeds_examined": int(candidates_examined),
        "cases": cases,
        "summary": summary,
        "analysis_contract": (
            "All four reliability scaling factors were fixed before the run and all are reported. "
            "The hidden worlds, prior, costs, orientations, and terminal losses are unchanged; only the planner's assumed source reliabilities are transformed. "
            "Performance is the expected cost of the entire assumed-model policy evaluated under the true observation model."
        ),
        "claim_boundary": (
            "This is a controlled calibration-misspecification stress test for static worlds. "
            "It does not cover wrong state-transition models, wrong query costs, adversarial orientation errors, or dynamic hidden states."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "claim_boundary": report["claim_boundary"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
