from __future__ import annotations

"""Held-out confirmation for source-stratified exact query screening.

Development observation: on the first 12 acquisition-active balanced cases, a top-6
one-step screen was much stronger than POMCP but mostly selected physical queries;
its remaining failure exposed a semantic-root case.  Before inspecting the next 12
acquisition-active cases, we therefore freeze a source-typed variant: keep the best
three physical-family queries and the best three semantic-family queries according
to the same horizon-one score, then run exact h=3 count-DP on those six queries.

The first 12 acquisition-active cases are development only.  The next 12 are the
confirmation set.  We report both unstratified top-6 and frozen 3+3 stratified
screening on confirmation; no rule is selected from confirmation outcomes.
"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_paper_b_scalability_robustness import (
    MODELS,
    cost_sensitive_queries,
    make_balanced_worlds,
    uniform_belief,
)
from benchmarks.run_paper_b_screened_exact import one_step_query_order
from paper_b.count_dp import EvidenceCountDP

OUT = ROOT / "results" / "paper_b_stratified_screen_confirmation.json"
HORIZON = 3
DEVELOPMENT_CASES = 12
CONFIRMATION_CASES = 12
SCREEN_SIZE = 6
PER_SOURCE_FAMILY = 3
MAX_CANDIDATE_SEEDS = 240


def source_family(kind: str) -> str:
    if kind in ("state", "calibrate_physical"):
        return "physical"
    if kind in ("model_feature", "calibrate_semantic"):
        return "semantic"
    raise ValueError(f"unknown query kind: {kind}")


def select_acquisition_active_cases() -> tuple[list[tuple], int]:
    selected = []
    candidate_seed = 0
    target = DEVELOPMENT_CASES + CONFIRMATION_CASES
    while len(selected) < target and candidate_seed < MAX_CANDIDATE_SEEDS:
        seed = 20261200 + candidate_seed
        candidate_seed += 1
        worlds = make_balanced_worlds(seed)
        belief = uniform_belief(worlds)
        queries = cost_sensitive_queries(1.0, 1.0)
        solver = EvidenceCountDP(belief, worlds, MODELS, queries, horizon=HORIZON)
        result = solver.solve()
        if result.action[0] == "QUERY":
            selected.append((seed, worlds, belief, queries, solver, result))
    if len(selected) < target:
        raise RuntimeError(f"only {len(selected)} acquisition-active cases found")
    return selected, candidate_seed


def stratified_indices(order: tuple[int, ...], queries) -> tuple[int, ...]:
    out = []
    for family in ("physical", "semantic"):
        family_order = [i for i in order if source_family(queries[i].kind) == family]
        out.extend(family_order[:PER_SOURCE_FAMILY])
    return tuple(out)


def map_action(action, selected_indices: tuple[int, ...]):
    if action[0] == "DECIDE":
        return ("DECIDE", int(action[1]))
    return ("QUERY", int(selected_indices[int(action[1])]))


def solve_screen(problem: tuple, selected_indices: tuple[int, ...]) -> dict:
    _, worlds, belief, queries, full_solver, full_result = problem
    reduced_queries = tuple(queries[i] for i in selected_indices)
    reduced_solver = EvidenceCountDP(
        belief, worlds, MODELS, reduced_queries, horizon=HORIZON
    )
    reduced = reduced_solver.solve()
    mapped = map_action(reduced.action, selected_indices)
    full_action = (str(full_result.action[0]), int(full_result.action[1]))
    regret = max(0.0, float(reduced.value - full_result.value))
    return {
        "selected_query_indices": list(selected_indices),
        "selected_query_names": [queries[i].name for i in selected_indices],
        "selected_source_families": [source_family(queries[i].kind) for i in selected_indices],
        "mapped_root_action": list(mapped),
        "full_root_action": list(full_action),
        "root_action_matches": bool(mapped == full_action),
        "root_action_family_matches": bool(mapped[0] == full_action[0]),
        "root_source_family_matches": bool(
            mapped[0] == "QUERY"
            and full_action[0] == "QUERY"
            and source_family(queries[mapped[1]].kind) == source_family(queries[full_action[1]].kind)
        ),
        "full_policy_regret": regret,
        "full_states": int(full_result.states),
        "screened_states": int(reduced.states),
        "state_reduction_factor": float(full_result.states / max(1, reduced.states)),
    }


def run_panel(cases: list[tuple], split: str) -> list[dict]:
    rows = []
    for local_id, problem in enumerate(cases):
        seed, worlds, belief, queries, full_solver, full_result = problem
        order, _, _ = one_step_query_order(belief, tuple(worlds), MODELS, tuple(queries))
        unstrat = tuple(order[:SCREEN_SIZE])
        strat = stratified_indices(order, queries)
        if len(strat) != SCREEN_SIZE:
            raise AssertionError(f"stratified screen did not select {SCREEN_SIZE} queries")
        rows.append(
            {
                "split": split,
                "case": int(local_id),
                "source_seed": int(seed),
                "full_root_query_name": queries[int(full_result.action[1])].name,
                "full_root_source_family": source_family(queries[int(full_result.action[1])].kind),
                "unstratified_top6": solve_screen(problem, unstrat),
                "stratified_3plus3": solve_screen(problem, strat),
            }
        )
    return rows


def summarize(rows: list[dict], key: str) -> dict:
    xs = [r[key] for r in rows]
    regrets = [float(x["full_policy_regret"]) for x in xs]
    return {
        "cases": len(xs),
        "root_exact_action_rate": float(sum(x["root_action_matches"] for x in xs) / len(xs)),
        "root_query_vs_decide_family_accuracy": float(
            sum(x["root_action_family_matches"] for x in xs) / len(xs)
        ),
        "root_source_family_accuracy": float(
            sum(x["root_source_family_matches"] for x in xs) / len(xs)
        ),
        "zero_full_policy_regret_rate": float(sum(abs(x) <= 1e-12 for x in regrets) / len(xs)),
        "mean_full_policy_regret": float(statistics.fmean(regrets)),
        "median_full_policy_regret": float(statistics.median(regrets)),
        "max_full_policy_regret": float(max(regrets)),
        "mean_state_reduction_factor": float(
            statistics.fmean(x["state_reduction_factor"] for x in xs)
        ),
    }


def main() -> None:
    selected, candidates_examined = select_acquisition_active_cases()
    dev = selected[:DEVELOPMENT_CASES]
    confirm = selected[DEVELOPMENT_CASES:]
    dev_rows = run_panel(dev, "development")
    confirmation_rows = run_panel(confirm, "confirmation")

    report = {
        "schema": "dovod-paper-b-stratified-screen-confirmation-v1",
        "candidate_seeds_examined": int(candidates_examined),
        "development_cases": DEVELOPMENT_CASES,
        "confirmation_cases": CONFIRMATION_CASES,
        "frozen_screen_rule": "rank at h=1, then keep top 3 physical-family and top 3 semantic-family queries; exact h=3 on the six-query subset",
        "development_summary": {
            "unstratified_top6": summarize(dev_rows, "unstratified_top6"),
            "stratified_3plus3": summarize(dev_rows, "stratified_3plus3"),
        },
        "confirmation_summary": {
            "unstratified_top6": summarize(confirmation_rows, "unstratified_top6"),
            "stratified_3plus3": summarize(confirmation_rows, "stratified_3plus3"),
        },
        "development_rows": dev_rows,
        "confirmation_rows": confirmation_rows,
        "analysis_contract": (
            "The source-stratified 3+3 rule was fixed after the first 12 development cases and before inspecting the next 12 acquisition-active cases. "
            "Confirmation reports both the original top-6 screen and the frozen stratified variant; no confirmation-based tuning or model selection is performed."
        ),
        "claim_boundary": (
            "This is a controlled held-out confirmation over balanced static worlds, not a real-source deployment study. "
            "Both six-query screens retain only a subset of the full query vocabulary and can have nonzero policy regret."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "development_summary": report["development_summary"],
        "confirmation_summary": report["confirmation_summary"],
        "claim_boundary": report["claim_boundary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
