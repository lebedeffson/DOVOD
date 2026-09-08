from __future__ import annotations

"""Deterministic query-screening approximation for the exact count-DP.

POMCP is intentionally retained as a stochastic baseline, including its negative
acquisition-active result.  This benchmark asks a different practical question:
can a cheap one-step screen reduce the query vocabulary and then hand the smaller
problem back to the exact evidence-count DP?

The evaluated cases are selected only because the *full* exact h=3 planner chooses
QUERY at the root.  Query screening uses horizon-one action values and never sees
horizon-three outcomes while ranking.  The screen sizes K={1,2,4,6} are fixed and
all are reported; no K is chosen retrospectively from the results.

The method is a deterministic engineering approximation, not an exactness theorem:
a query that is weak myopically can still be useful through multi-query synergy and
can therefore be screened out.
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
from paper_b.static_world import Query, World

OUT = ROOT / "results" / "paper_b_screened_exact.json"
SCREEN_SIZES = (1, 2, 4, 6)
N_CASES = 12
HORIZON = 3
MAX_CANDIDATE_SEEDS = 120


def one_step_query_order(
    belief,
    worlds: tuple[World, ...],
    models,
    queries: tuple[Query, ...],
) -> tuple[tuple[int, ...], dict[int, float], float]:
    """Rank query indices by exact horizon-one expected cost, low is better."""
    t0 = perf_counter()
    solver = EvidenceCountDP(belief, worlds, models, queries, horizon=1)
    values = solver.root_action_values()
    query_values = {int(qi): float(values[("QUERY", qi)]) for qi in range(len(queries))}
    order = tuple(sorted(query_values, key=lambda qi: (query_values[qi], qi)))
    return order, query_values, float(perf_counter() - t0)


def exact_problem(candidate_seed: int):
    seed = 20261200 + int(candidate_seed)
    worlds = make_balanced_worlds(seed)
    belief = uniform_belief(worlds)
    queries = cost_sensitive_queries(1.0, 1.0)
    solver = EvidenceCountDP(belief, worlds, MODELS, queries, horizon=HORIZON)
    result = solver.solve()
    return seed, worlds, belief, queries, solver, result


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


def map_reduced_action(action, selected_indices: tuple[int, ...]):
    if action[0] == "DECIDE":
        return ("DECIDE", int(action[1]))
    return ("QUERY", int(selected_indices[int(action[1])]))


def run_case(case_id: int, problem: tuple) -> dict:
    seed, worlds, belief, queries, full_solver, full_result = problem
    order, one_step_values, screening_seconds = one_step_query_order(
        belief, tuple(worlds), MODELS, tuple(queries)
    )
    full_action = (str(full_result.action[0]), int(full_result.action[1]))

    rows = []
    for k in SCREEN_SIZES:
        selected_indices = tuple(order[:k])
        reduced_queries = tuple(queries[i] for i in selected_indices)
        reduced_solver = EvidenceCountDP(
            belief, worlds, MODELS, reduced_queries, horizon=HORIZON
        )
        reduced_result = reduced_solver.solve()
        mapped_action = map_reduced_action(reduced_result.action, selected_indices)
        regret = float(reduced_result.value - full_result.value)
        if regret < -1e-10:
            raise AssertionError(
                f"screened policy beat the full query vocabulary unexpectedly: {regret}"
            )
        rows.append(
            {
                "screen_size": int(k),
                "selected_query_indices": list(selected_indices),
                "selected_query_names": [queries[i].name for i in selected_indices],
                "mapped_root_action": list(mapped_action),
                "root_action_matches_full_exact": bool(mapped_action == full_action),
                "root_action_family_matches": bool(mapped_action[0] == full_action[0]),
                "full_exact_value": float(full_result.value),
                "screened_exact_value": float(reduced_result.value),
                "full_policy_regret_from_screening": max(0.0, regret),
                "full_exact_states": int(full_result.states),
                "screened_exact_states": int(reduced_result.states),
                "state_reduction_factor": float(full_result.states / max(1, reduced_result.states)),
                "full_exact_seconds": float(full_result.seconds),
                "screening_seconds": float(screening_seconds),
                "screened_exact_seconds": float(reduced_result.seconds),
                "screen_plus_solve_seconds": float(screening_seconds + reduced_result.seconds),
            }
        )

    return {
        "case": int(case_id),
        "source_seed": int(seed),
        "queries": len(queries),
        "horizon": HORIZON,
        "full_exact_root_action": list(full_action),
        "full_exact_root_query_name": (
            queries[full_action[1]].name if full_action[0] == "QUERY" else None
        ),
        "one_step_query_order": list(order),
        "one_step_query_values": {str(i): float(one_step_values[i]) for i in order},
        "rows": rows,
    }


def summarize(cases: list[dict]) -> list[dict]:
    summary = []
    for k in SCREEN_SIZES:
        rows = [next(r for r in case["rows"] if r["screen_size"] == k) for case in cases]
        regrets = [float(r["full_policy_regret_from_screening"]) for r in rows]
        state_factors = [float(r["state_reduction_factor"]) for r in rows]
        speedups = [
            float(r["full_exact_seconds"] / r["screen_plus_solve_seconds"])
            for r in rows
            if r["screen_plus_solve_seconds"] > 0
        ]
        summary.append(
            {
                "screen_size": int(k),
                "cases": len(rows),
                "root_exact_action_rate": float(
                    sum(r["root_action_matches_full_exact"] for r in rows) / len(rows)
                ),
                "root_query_vs_decide_family_accuracy": float(
                    sum(r["root_action_family_matches"] for r in rows) / len(rows)
                ),
                "zero_full_policy_regret_rate": float(
                    sum(abs(x) <= 1e-12 for x in regrets) / len(regrets)
                ),
                "mean_full_policy_regret": float(statistics.fmean(regrets)),
                "median_full_policy_regret": float(statistics.median(regrets)),
                "max_full_policy_regret": float(max(regrets)),
                "mean_state_reduction_factor": float(statistics.fmean(state_factors)),
                "median_state_reduction_factor": float(statistics.median(state_factors)),
                "mean_wall_clock_speedup_including_screen": float(statistics.fmean(speedups)),
            }
        )
    return summary


def main() -> None:
    selected, candidates_examined = select_cases()
    cases = [run_case(i, problem) for i, problem in enumerate(selected)]
    summary = summarize(cases)
    report = {
        "schema": "dovod-paper-b-screened-exact-v1",
        "screen_sizes": list(SCREEN_SIZES),
        "selection_rule": (
            "first 12 balanced-prior candidate seeds whose full exact nominal-cost h=3 root action is QUERY"
        ),
        "candidate_seeds_examined": int(candidates_examined),
        "cases": cases,
        "summary": summary,
        "analysis_contract": (
            "One-step query ranking is computed before each reduced h=3 solve. Screen sizes 1,2,4,6 are fixed and all are reported. "
            "Policy regret compares each reduced-query exact policy directly with the full 10-query exact policy on the same hidden worlds."
        ),
        "claim_boundary": (
            "Query screening is a deterministic engineering heuristic, not an exact compression result. "
            "It can miss multi-query synergies because ranking uses only one-step values. Wall-clock speedups are environment-specific; state reductions and policy regret are the primary evidence."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "claim_boundary": report["claim_boundary"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
