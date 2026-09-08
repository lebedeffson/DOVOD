from __future__ import annotations

"""Likelihood-misspecification stress test for exact information acquisition.

The environment is kept fixed while the planner is given a systematically
miscalibrated reliability model. We evaluate the *entire assumed-model policy*
under the true observation model, not only its first action.

For each hidden world, assumed source reliability is transformed as

    r_assumed = clip(0.5 + alpha * (r_true - 0.5), 0.5, 1.0).

alpha < 1 means the planner underestimates source informativeness; alpha > 1
means it is overconfident. The levels are fixed in advance and all are reported.
Cases are the first 12 balanced-prior exact problems whose true optimal root action
is QUERY. Case selection never inspects misspecification performance.

In addition to expected total cost, the evaluator decomposes each induced policy
into expected query count, acquisition cost, terminal decision loss, false-allow
probability, and false-block probability under the true model. This makes the
failure mechanism operationally interpretable instead of reporting regret alone.
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


def evaluate_assumed_policy_metrics_under_true_model(
    true_solver: EvidenceCountDP,
    assumed_solver: EvidenceCountDP,
) -> dict[str, float]:
    """Evaluate the entire assumed-model policy in the true environment.

    Returns expected totals from the root. False-allow / false-block quantities
    are probabilities of ending in the corresponding wrong decision, while
    terminal_decision_loss keeps the asymmetric loss weighting used by Bellman.
    """

    @lru_cache(maxsize=None)
    def metrics(counts: tuple[int, ...]) -> tuple[float, float, float, float, float, float]:
        true_belief = np.asarray(true_solver.belief_from_counts(counts), dtype=float)
        _, action = assumed_solver.V(counts)

        if action[0] == "DECIDE":
            decision = int(action[1])
            losses = true_solver._loss[decision]
            terminal_loss = float(losses @ true_belief)
            error_probability = float((losses > 0).astype(float) @ true_belief)
            false_allow_probability = error_probability if decision == 1 else 0.0
            false_block_probability = error_probability if decision == 0 else 0.0
            return (
                terminal_loss,
                0.0,
                0.0,
                terminal_loss,
                false_allow_probability,
                false_block_probability,
            )

        qi = int(action[1])
        immediate_cost = float(true_solver.queries[qi].cost)
        total_cost = immediate_cost
        acquisition_cost = immediate_cost
        query_count = 1.0
        terminal_loss = 0.0
        false_allow_probability = 0.0
        false_block_probability = 0.0

        for obs in (0, 1):
            po = float(true_solver._lik[qi, obs] @ true_belief)
            if po <= 1e-14:
                continue
            nxt = true_solver._inc(counts, qi, obs)
            child = metrics(nxt)
            total_cost += po * child[0]
            acquisition_cost += po * child[1]
            query_count += po * child[2]
            terminal_loss += po * child[3]
            false_allow_probability += po * child[4]
            false_block_probability += po * child[5]

        return (
            float(total_cost),
            float(acquisition_cost),
            float(query_count),
            float(terminal_loss),
            float(false_allow_probability),
            float(false_block_probability),
        )

    values = metrics(true_solver.zero_counts)
    report = {
        "expected_total_cost": float(values[0]),
        "expected_acquisition_cost": float(values[1]),
        "expected_query_count": float(values[2]),
        "expected_terminal_decision_loss": float(values[3]),
        "false_allow_probability": float(values[4]),
        "false_block_probability": float(values[5]),
        "wrong_decision_probability": float(values[4] + values[5]),
    }
    if abs(
        report["expected_total_cost"]
        - report["expected_acquisition_cost"]
        - report["expected_terminal_decision_loss"]
    ) > 1e-9:
        raise AssertionError(f"policy cost decomposition is inconsistent: {report}")
    return report


def evaluate_assumed_policy_under_true_model(
    true_solver: EvidenceCountDP,
    assumed_solver: EvidenceCountDP,
) -> float:
    """Backward-compatible scalar wrapper for expected true policy cost."""
    return float(
        evaluate_assumed_policy_metrics_under_true_model(true_solver, assumed_solver)[
            "expected_total_cost"
        ]
    )


def run_case(case_id: int, problem: tuple) -> dict:
    seed, worlds, belief, queries, true_solver, true_result = problem
    rows = []
    for alpha in ALPHAS:
        aw = assumed_worlds(tuple(worlds), alpha)
        assumed_solver = EvidenceCountDP(belief, aw, MODELS, queries, horizon=HORIZON)
        assumed_result = assumed_solver.solve()
        operational = evaluate_assumed_policy_metrics_under_true_model(true_solver, assumed_solver)
        true_policy_value = float(operational["expected_total_cost"])
        rows.append(
            {
                "alpha": float(alpha),
                "assumed_root_action": list(assumed_result.action),
                "root_action_matches_true_optimum": bool(assumed_result.action == true_result.action),
                "root_action_family_matches": bool(assumed_result.action[0] == true_result.action[0]),
                "assumed_model_value": float(assumed_result.value),
                "true_value_of_assumed_policy": true_policy_value,
                "full_policy_regret_under_true_model": float(true_policy_value - true_result.value),
                "operational_metrics_under_true_model": operational,
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
        op = [r["operational_metrics_under_true_model"] for r in rows]
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
                "mean_expected_query_count": float(statistics.fmean(x["expected_query_count"] for x in op)),
                "mean_expected_acquisition_cost": float(statistics.fmean(x["expected_acquisition_cost"] for x in op)),
                "mean_expected_terminal_decision_loss": float(
                    statistics.fmean(x["expected_terminal_decision_loss"] for x in op)
                ),
                "mean_wrong_decision_probability": float(
                    statistics.fmean(x["wrong_decision_probability"] for x in op)
                ),
                "mean_false_allow_probability": float(
                    statistics.fmean(x["false_allow_probability"] for x in op)
                ),
                "mean_false_block_probability": float(
                    statistics.fmean(x["false_block_probability"] for x in op)
                ),
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
        "schema": "dovod-paper-b-likelihood-misspecification-v2",
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
            "Performance is the expected cost of the entire assumed-model policy evaluated under the true observation model. "
            "Operational decomposition reports expected query count/acquisition cost plus terminal false-allow and false-block probabilities under that same true model."
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
