from __future__ import annotations

"""Structural typed-core screening: keep decision-bearing queries, drop calibrations.

The one-step and two-step value rankings can miss semantic model-feature queries that
matter only through deeper interaction.  A different deterministic screen uses the
known query semantics instead of shallow values: retain all four physical state
queries and all four semantic model-feature queries, while removing the two explicit
source-calibration queries.  The resulting eight-query vocabulary is solved exactly
at horizon three.

The rule is frozen before this run.  It is evaluated on (i) 32 unconditional fresh
balanced cases and (ii) a separate fresh source-diverse stress panel containing the
first eight physical-root and first eight semantic-root acquisition-active cases.
No case is selected according to screen performance.
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
from paper_b.count_dp import EvidenceCountDP

OUT = ROOT / "results" / "paper_b_structural_core_screen.json"
HORIZON = 3
UNCONDITIONAL_SEED_BASE = 20267000
UNCONDITIONAL_CASES = 32
STRESS_SEED_BASE = 20268000
PER_FAMILY = 8
MAX_CANDIDATES = 600


def source_family(kind: str) -> str:
    if kind in ("state", "calibrate_physical"):
        return "physical"
    if kind in ("model_feature", "calibrate_semantic"):
        return "semantic"
    raise ValueError(f"unknown query kind {kind}")


def structural_core_indices(queries) -> tuple[int, ...]:
    return tuple(i for i, q in enumerate(queries) if q.kind in ("state", "model_feature"))


def exact_problem(seed: int):
    worlds = make_balanced_worlds(seed)
    belief = uniform_belief(worlds)
    queries = cost_sensitive_queries(1.0, 1.0)
    solver = EvidenceCountDP(belief, worlds, MODELS, queries, horizon=HORIZON)
    result = solver.solve()
    return seed, worlds, belief, queries, solver, result


def map_action(action, selected: tuple[int, ...]):
    if action[0] == "DECIDE":
        return ("DECIDE", int(action[1]))
    return ("QUERY", int(selected[int(action[1])]))


def evaluate(problem: tuple) -> dict:
    seed, worlds, belief, queries, full_solver, full = problem
    selected = structural_core_indices(queries)
    reduced_queries = tuple(queries[i] for i in selected)
    reduced_solver = EvidenceCountDP(belief, worlds, MODELS, reduced_queries, horizon=HORIZON)
    reduced = reduced_solver.solve()
    mapped = map_action(reduced.action, selected)
    full_action = (str(full.action[0]), int(full.action[1]))
    regret = max(0.0, float(reduced.value - full.value))
    return {
        "seed": int(seed),
        "full_root_action": list(full_action),
        "full_root_query_name": queries[full_action[1]].name if full_action[0] == "QUERY" else None,
        "full_root_source_family": (
            source_family(queries[full_action[1]].kind) if full_action[0] == "QUERY" else None
        ),
        "selected_query_indices": list(selected),
        "selected_query_names": [queries[i].name for i in selected],
        "mapped_root_action": list(mapped),
        "root_exact_action_match": bool(mapped == full_action),
        "root_query_vs_decide_family_match": bool(mapped[0] == full_action[0]),
        "full_policy_regret": regret,
        "zero_policy_regret": bool(regret <= 1e-12),
        "full_states": int(full.states),
        "screened_states": int(reduced.states),
        "state_reduction_factor": float(full.states / max(1, reduced.states)),
        "full_seconds": float(full.seconds),
        "screened_seconds": float(reduced.seconds),
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"cases": 0}
    regrets = [float(r["full_policy_regret"]) for r in rows]
    return {
        "cases": len(rows),
        "root_exact_action_rate": float(sum(r["root_exact_action_match"] for r in rows) / len(rows)),
        "root_query_vs_decide_family_accuracy": float(
            sum(r["root_query_vs_decide_family_match"] for r in rows) / len(rows)
        ),
        "zero_full_policy_regret_rate": float(sum(r["zero_policy_regret"] for r in rows) / len(rows)),
        "mean_full_policy_regret": float(statistics.fmean(regrets)),
        "median_full_policy_regret": float(statistics.median(regrets)),
        "max_full_policy_regret": float(max(regrets)),
        "mean_state_reduction_factor": float(statistics.fmean(r["state_reduction_factor"] for r in rows)),
        "mean_wall_clock_speedup": float(statistics.fmean(
            r["full_seconds"] / r["screened_seconds"] for r in rows if r["screened_seconds"] > 0
        )),
    }


def collect_stress_cases() -> tuple[dict[str, list[tuple]], int]:
    selected = {"physical": [], "semantic": []}
    examined = 0
    while examined < MAX_CANDIDATES and any(len(v) < PER_FAMILY for v in selected.values()):
        problem = exact_problem(STRESS_SEED_BASE + examined)
        examined += 1
        queries = problem[3]
        result = problem[5]
        if result.action[0] != "QUERY":
            continue
        family = source_family(queries[int(result.action[1])].kind)
        if len(selected[family]) < PER_FAMILY:
            selected[family].append(problem)
    if any(len(v) < PER_FAMILY for v in selected.values()):
        raise RuntimeError(
            f"insufficient stress cases after {examined}: physical={len(selected['physical'])}, semantic={len(selected['semantic'])}"
        )
    return selected, examined


def main() -> None:
    unconditional = [evaluate(exact_problem(UNCONDITIONAL_SEED_BASE + i)) for i in range(UNCONDITIONAL_CASES)]
    stress_cases, examined = collect_stress_cases()
    stress = {
        family: [evaluate(problem) for problem in stress_cases[family]]
        for family in ("physical", "semantic")
    }

    report = {
        "schema": "dovod-paper-b-structural-core-screen-v1",
        "screen_rule": "retain every state and model_feature query; remove calibrate_physical and calibrate_semantic",
        "unconditional_seed_block": [UNCONDITIONAL_SEED_BASE, UNCONDITIONAL_SEED_BASE + UNCONDITIONAL_CASES - 1],
        "unconditional": {
            "summary": summarize(unconditional),
            "rows": unconditional,
        },
        "source_diverse_stress": {
            "seed_base": STRESS_SEED_BASE,
            "candidate_seeds_examined": int(examined),
            "physical": {"summary": summarize(stress["physical"]), "rows": stress["physical"]},
            "semantic": {"summary": summarize(stress["semantic"]), "rows": stress["semantic"]},
        },
        "analysis_contract": (
            "The eight-query structural rule, 32-case unconditional seed block, and 8+8 source-diverse stress sizes are fixed before execution. "
            "The stress panel is selected only by the full exact root source family, not screen performance."
        ),
        "claim_boundary": (
            "Removing calibration queries is not generally valid when calibration information has decision value. "
            "This benchmark tests that structural simplification only in the stated static-world family and reports any induced policy regret."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "unconditional": report["unconditional"]["summary"],
        "physical_stress": report["source_diverse_stress"]["physical"]["summary"],
        "semantic_stress": report["source_diverse_stress"]["semantic"]["summary"],
        "candidate_seeds_examined": examined,
        "claim_boundary": report["claim_boundary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
