from __future__ import annotations

"""Fresh source-diverse confirmation of shallow exact screening.

The h=1 top-6 screen was strong on physical-root cases but a targeted stress showed
that semantically optimal queries can rank only 7th-10th myopically.  This follow-up
freezes a single refinement before inspecting a new seed block: rank queries by
exact horizon-two root query value, keep the best six, and run exact horizon-three
count-DP on that reduced vocabulary.

For transparency the old h=1 top-6 screen is evaluated on the same new cases as a
reference.  Cases are the first eight physical-root and first eight semantic-root
full-exact h=3 problems from seeds starting at 20266000.  Selection uses only the
full-exact root source family and never screening success.
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

OUT = ROOT / "results" / "paper_b_h2_screen_confirmation.json"
HORIZON = 3
SCREEN_SIZE = 6
PER_FAMILY = 8
SEED_BASE = 20266000
MAX_CANDIDATES = 600


def source_family(kind: str) -> str:
    if kind in ("state", "calibrate_physical"):
        return "physical"
    if kind in ("model_feature", "calibrate_semantic"):
        return "semantic"
    raise ValueError(f"unknown query kind {kind}")


def shallow_query_order(belief, worlds, models, queries, depth: int):
    t0 = perf_counter()
    solver = EvidenceCountDP(belief, worlds, models, queries, horizon=int(depth))
    values = solver.root_action_values()
    query_values = {i: float(values[("QUERY", i)]) for i in range(len(queries))}
    order = tuple(sorted(query_values, key=lambda i: (query_values[i], i)))
    seconds = float(perf_counter() - t0)
    return order, query_values, solver, seconds


def exact_problem(seed: int):
    worlds = make_balanced_worlds(seed)
    belief = uniform_belief(worlds)
    queries = cost_sensitive_queries(1.0, 1.0)
    solver = EvidenceCountDP(belief, worlds, MODELS, queries, horizon=HORIZON)
    result = solver.solve()
    return worlds, belief, queries, solver, result


def collect_cases() -> tuple[dict[str, list[tuple]], int]:
    selected = {"physical": [], "semantic": []}
    examined = 0
    while examined < MAX_CANDIDATES and any(len(v) < PER_FAMILY for v in selected.values()):
        seed = SEED_BASE + examined
        examined += 1
        problem = exact_problem(seed)
        result = problem[4]
        if result.action[0] != "QUERY":
            continue
        queries = problem[2]
        fam = source_family(queries[int(result.action[1])].kind)
        if len(selected[fam]) < PER_FAMILY:
            selected[fam].append((seed, *problem))
    if any(len(v) < PER_FAMILY for v in selected.values()):
        raise RuntimeError(
            f"insufficient cases after {examined}: physical={len(selected['physical'])}, semantic={len(selected['semantic'])}"
        )
    return selected, examined


def map_action(action, selected: tuple[int, ...]):
    if action[0] == "DECIDE":
        return ("DECIDE", int(action[1]))
    return ("QUERY", int(selected[int(action[1])]))


def evaluate_screen(problem: tuple, depth: int) -> dict:
    seed, worlds, belief, queries, full_solver, full = problem
    order, query_values, shallow_solver, screening_seconds = shallow_query_order(
        belief, tuple(worlds), MODELS, tuple(queries), depth
    )
    selected = tuple(order[:SCREEN_SIZE])
    reduced_queries = tuple(queries[i] for i in selected)
    reduced_solver = EvidenceCountDP(
        belief, worlds, MODELS, reduced_queries, horizon=HORIZON
    )
    reduced = reduced_solver.solve()
    mapped = map_action(reduced.action, selected)
    full_action = (str(full.action[0]), int(full.action[1]))
    regret = max(0.0, float(reduced.value - full.value))

    # root_action_values populates the shallow solver's value cache through the
    # child states needed for its query evaluations. This count is a deterministic
    # proxy for screening work, not a claim about unique states across both solvers.
    screen_states = int(shallow_solver.stats["states"])
    total_proxy_states = screen_states + int(reduced.states)
    return {
        "screen_depth": int(depth),
        "selected_query_indices": list(selected),
        "selected_query_names": [queries[i].name for i in selected],
        "selected_source_families": [source_family(queries[i].kind) for i in selected],
        "full_root_query_in_screen": bool(full_action[1] in selected),
        "rank_of_full_root_query": int(order.index(full_action[1]) + 1),
        "shallow_value_of_full_root_query": float(query_values[full_action[1]]),
        "mapped_root_action": list(mapped),
        "root_exact_action_match": bool(mapped == full_action),
        "root_query_vs_decide_family_match": bool(mapped[0] == full_action[0]),
        "full_policy_regret": regret,
        "zero_policy_regret": bool(regret <= 1e-12),
        "screen_states": screen_states,
        "screened_h3_states": int(reduced.states),
        "screen_plus_solve_state_proxy": int(total_proxy_states),
        "full_h3_states": int(full.states),
        "state_work_reduction_factor": float(full.states / max(1, total_proxy_states)),
        "screening_seconds": float(screening_seconds),
        "reduced_h3_seconds": float(reduced.seconds),
        "screen_plus_solve_seconds": float(screening_seconds + reduced.seconds),
        "full_h3_seconds": float(full.seconds),
    }


def run_case(case_id: int, family: str, problem: tuple) -> dict:
    seed, worlds, belief, queries, full_solver, full = problem
    full_action = (str(full.action[0]), int(full.action[1]))
    return {
        "case": int(case_id),
        "source_seed": int(seed),
        "target_root_source_family": family,
        "full_root_action": list(full_action),
        "full_root_query_name": queries[full_action[1]].name,
        "h1_top6": evaluate_screen(problem, 1),
        "h2_top6": evaluate_screen(problem, 2),
    }


def summarize(cases: list[dict], key: str) -> dict:
    xs = [c[key] for c in cases]
    regrets = [float(x["full_policy_regret"]) for x in xs]
    return {
        "cases": len(xs),
        "full_root_query_in_screen_rate": float(sum(x["full_root_query_in_screen"] for x in xs) / len(xs)),
        "root_exact_action_rate": float(sum(x["root_exact_action_match"] for x in xs) / len(xs)),
        "root_query_vs_decide_family_accuracy": float(
            sum(x["root_query_vs_decide_family_match"] for x in xs) / len(xs)
        ),
        "zero_full_policy_regret_rate": float(sum(x["zero_policy_regret"] for x in xs) / len(xs)),
        "mean_full_policy_regret": float(statistics.fmean(regrets)),
        "median_full_policy_regret": float(statistics.median(regrets)),
        "max_full_policy_regret": float(max(regrets)),
        "mean_state_work_reduction_factor": float(
            statistics.fmean(x["state_work_reduction_factor"] for x in xs)
        ),
        "mean_wall_clock_speedup_including_screen": float(
            statistics.fmean(
                x["full_h3_seconds"] / x["screen_plus_solve_seconds"]
                for x in xs
                if x["screen_plus_solve_seconds"] > 0
            )
        ),
    }


def main() -> None:
    selected, examined = collect_cases()
    panels = {}
    for family in ("physical", "semantic"):
        cases = [run_case(i, family, packed) for i, packed in enumerate(selected[family])]
        panels[family] = {
            "h1_top6_summary": summarize(cases, "h1_top6"),
            "h2_top6_summary": summarize(cases, "h2_top6"),
            "cases": cases,
        }

    report = {
        "schema": "dovod-paper-b-h2-screen-confirmation-v1",
        "fresh_seed_base": SEED_BASE,
        "candidate_seeds_examined": int(examined),
        "cases_per_source_family": PER_FAMILY,
        "screen_size": SCREEN_SIZE,
        "panels": panels,
        "analysis_contract": (
            "The h=2 ranking refinement and K=6 are fixed before evaluating this fresh seed block. "
            "Cases are selected only by full-exact h=3 root source family; both h=1 and h=2 screens are reported on exactly the same selected cases."
        ),
        "claim_boundary": (
            "This is targeted source-diverse confirmation, not a prevalence estimate. The h=2 screen is still a heuristic: "
            "multi-query value at depth three can in principle depend on a query that ranks poorly at depth two. State-work factors sum state counts from separate solvers and are engineering proxies."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "candidate_seeds_examined": examined,
        "physical": {"h1": panels["physical"]["h1_top6_summary"], "h2": panels["physical"]["h2_top6_summary"]},
        "semantic": {"h1": panels["semantic"]["h1_top6_summary"], "h2": panels["semantic"]["h2_top6_summary"]},
        "claim_boundary": report["claim_boundary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
