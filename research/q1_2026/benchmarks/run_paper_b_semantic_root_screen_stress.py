from __future__ import annotations

"""Mechanism stress test for query screening across root source families.

The unconditional and held-out screening panels are dominated by physical-root
problems.  This targeted diagnostic therefore selects cases by the *full exact*
root source family only, never by screening success: the first eight physical-root
and first eight semantic-root h=3 cases from a fresh deterministic seed block.

For each case, queries are ranked by exact horizon-one query value.  Reduced exact
horizon-three planners retain the top K queries for K in {6,7,8}.  All K are fixed
and reported.  The goal is not to estimate prevalence; it is to expose whether the
screening heuristic fails specifically when semantic acquisition is truly optimal.
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

OUT = ROOT / "results" / "paper_b_semantic_root_screen_stress.json"
HORIZON = 3
SCREEN_SIZES = (6, 7, 8)
PER_FAMILY = 8
SEED_BASE = 20265000
MAX_CANDIDATES = 600


def source_family(kind: str) -> str:
    if kind in ("state", "calibrate_physical"):
        return "physical"
    if kind in ("model_feature", "calibrate_semantic"):
        return "semantic"
    raise ValueError(f"unknown query kind {kind}")


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
            f"insufficient source-family cases after {examined}: "
            f"physical={len(selected['physical'])}, semantic={len(selected['semantic'])}"
        )
    return selected, examined


def map_action(action, selected: tuple[int, ...]):
    if action[0] == "DECIDE":
        return ("DECIDE", int(action[1]))
    return ("QUERY", int(selected[int(action[1])]))


def run_case(case_id: int, family: str, packed: tuple) -> dict:
    seed, worlds, belief, queries, full_solver, full = packed
    order, one_step_values, screening_seconds = one_step_query_order(
        belief, tuple(worlds), MODELS, tuple(queries)
    )
    full_action = (str(full.action[0]), int(full.action[1]))
    rows = []
    for k in SCREEN_SIZES:
        selected = tuple(order[:k])
        reduced_queries = tuple(queries[i] for i in selected)
        reduced_solver = EvidenceCountDP(
            belief, worlds, MODELS, reduced_queries, horizon=HORIZON
        )
        reduced = reduced_solver.solve()
        mapped = map_action(reduced.action, selected)
        regret = max(0.0, float(reduced.value - full.value))
        rows.append({
            "screen_size": int(k),
            "selected_query_indices": list(selected),
            "selected_query_names": [queries[i].name for i in selected],
            "selected_source_families": [source_family(queries[i].kind) for i in selected],
            "full_root_query_in_screen": bool(full_action[1] in selected),
            "mapped_root_action": list(mapped),
            "root_exact_action_match": bool(mapped == full_action),
            "root_query_vs_decide_family_match": bool(mapped[0] == full_action[0]),
            "full_policy_regret": regret,
            "zero_policy_regret": bool(regret <= 1e-12),
            "state_reduction_factor": float(full.states / max(1, reduced.states)),
            "screening_seconds": float(screening_seconds),
            "reduced_seconds": float(reduced.seconds),
        })
    return {
        "case": int(case_id),
        "source_seed": int(seed),
        "target_root_source_family": family,
        "full_root_action": list(full_action),
        "full_root_query_name": queries[full_action[1]].name,
        "full_root_query_kind": queries[full_action[1]].kind,
        "one_step_rank_of_full_root_query": int(order.index(full_action[1]) + 1),
        "one_step_value_of_full_root_query": float(one_step_values[full_action[1]]),
        "rows": rows,
    }


def summarize(rows: list[dict], k: int) -> dict:
    xs = [next(r for r in case["rows"] if r["screen_size"] == k) for case in rows]
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
        "mean_state_reduction_factor": float(statistics.fmean(x["state_reduction_factor"] for x in xs)),
    }


def main() -> None:
    selected, examined = collect_cases()
    panels = {}
    for family in ("physical", "semantic"):
        case_rows = [run_case(i, family, packed) for i, packed in enumerate(selected[family])]
        panels[family] = {
            "summary": {str(k): summarize(case_rows, k) for k in SCREEN_SIZES},
            "cases": case_rows,
        }

    report = {
        "schema": "dovod-paper-b-semantic-root-screen-stress-v1",
        "seed_base": SEED_BASE,
        "candidate_seeds_examined": int(examined),
        "cases_per_source_family": PER_FAMILY,
        "screen_sizes": list(SCREEN_SIZES),
        "panels": panels,
        "analysis_contract": (
            "Cases are selected only by the full exact h=3 root source family from a fresh seed block. "
            "Screen sizes 6,7,8 and the h=1 ranking rule are fixed before execution and all are reported."
        ),
        "claim_boundary": (
            "This is a targeted source-family stress test and cannot estimate how common semantic-root cases are. "
            "Any reduced-query policy with nonzero regret remains an approximation; source-family selection itself uses the full exact oracle."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "candidate_seeds_examined": examined,
        "physical": panels["physical"]["summary"],
        "semantic": panels["semantic"]["summary"],
        "claim_boundary": report["claim_boundary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
