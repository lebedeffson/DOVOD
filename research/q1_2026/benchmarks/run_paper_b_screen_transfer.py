from __future__ import annotations

"""Out-of-sample source-cost transfer test for the frozen top-6 query screen.

The screen size K=6 and one-step ranking rule were fixed by the earlier development
analysis.  This test uses a fresh deterministic seed block and three cost regimes
fixed before execution: nominal, physical evidence expensive, and semantic evidence
expensive.  Cases are not filtered on the horizon-three action or on screening
success.  For each case, the top six queries under the horizon-one exact action-value
ranking are retained and an exact horizon-three count-DP is solved on that reduced
vocabulary.

The report gives both unconditional results and the acquisition-active subset in each
cost regime.  This is a transfer diagnostic for the screening heuristic, not an
exactness theorem.
"""

import json
import statistics
import sys
from collections import Counter
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

OUT = ROOT / "results" / "paper_b_screen_transfer.json"
HORIZON = 3
SCREEN_SIZE = 6
N_SEEDS = 16
SEED_BASE = 20264000
COST_REGIMES = (
    ("nominal", 1.0, 1.0),
    ("physical_expensive", 5.0, 1.0),
    ("semantic_expensive", 1.0, 5.0),
)


def query_family(kind: str) -> str:
    if kind in ("state", "calibrate_physical"):
        return "physical"
    if kind in ("model_feature", "calibrate_semantic"):
        return "semantic"
    raise ValueError(f"unknown query kind {kind}")


def map_action(action, selected: tuple[int, ...]):
    if action[0] == "DECIDE":
        return ("DECIDE", int(action[1]))
    return ("QUERY", int(selected[int(action[1])]))


def run_case(seed: int, physical_cost: float, semantic_cost: float) -> dict:
    worlds = make_balanced_worlds(seed)
    belief = uniform_belief(worlds)
    queries = cost_sensitive_queries(physical_cost, semantic_cost)

    full_solver = EvidenceCountDP(belief, worlds, MODELS, queries, horizon=HORIZON)
    full = full_solver.solve()

    order, _, screening_seconds = one_step_query_order(
        belief, tuple(worlds), MODELS, tuple(queries)
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

    full_source = None
    if full_action[0] == "QUERY":
        full_source = query_family(queries[full_action[1]].kind)

    selected_families = [query_family(queries[i].kind) for i in selected]
    return {
        "seed": int(seed),
        "full_root_action": list(full_action),
        "full_root_query_name": queries[full_action[1]].name if full_action[0] == "QUERY" else None,
        "full_root_source_family": full_source,
        "selected_query_indices": list(selected),
        "selected_query_names": [queries[i].name for i in selected],
        "selected_source_families": selected_families,
        "selected_physical_count": int(sum(x == "physical" for x in selected_families)),
        "selected_semantic_count": int(sum(x == "semantic" for x in selected_families)),
        "mapped_root_action": list(mapped),
        "root_exact_action_match": bool(mapped == full_action),
        "root_query_vs_decide_family_match": bool(mapped[0] == full_action[0]),
        "full_policy_regret": regret,
        "zero_policy_regret": bool(regret <= 1e-12),
        "full_states": int(full.states),
        "screened_states": int(reduced.states),
        "state_reduction_factor": float(full.states / max(1, reduced.states)),
        "full_seconds": float(full.seconds),
        "screening_seconds": float(screening_seconds),
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
        "mean_state_reduction_factor": float(
            statistics.fmean(r["state_reduction_factor"] for r in rows)
        ),
        "full_root_source_families": dict(Counter(
            r["full_root_source_family"] if r["full_root_source_family"] is not None else "DECIDE"
            for r in rows
        )),
        "mean_selected_physical_queries": float(
            statistics.fmean(r["selected_physical_count"] for r in rows)
        ),
        "mean_selected_semantic_queries": float(
            statistics.fmean(r["selected_semantic_count"] for r in rows)
        ),
    }


def main() -> None:
    regimes = []
    all_rows = []
    for name, physical_cost, semantic_cost in COST_REGIMES:
        rows = [
            run_case(SEED_BASE + i, physical_cost, semantic_cost)
            for i in range(N_SEEDS)
        ]
        active = [r for r in rows if r["full_root_action"][0] == "QUERY"]
        regimes.append({
            "name": name,
            "physical_cost_multiplier": physical_cost,
            "semantic_cost_multiplier": semantic_cost,
            "summary_all_cases": summarize(rows),
            "summary_acquisition_active": summarize(active),
            "rows": rows,
        })
        all_rows.extend(rows)

    report = {
        "schema": "dovod-paper-b-screen-transfer-v1",
        "screen_rule": "top 6 queries by exact horizon-one root query value, then exact horizon-three count-DP on the reduced vocabulary",
        "fresh_seed_block": [SEED_BASE, SEED_BASE + N_SEEDS - 1],
        "cost_regimes": [
            {"name": n, "physical": p, "semantic": s} for n, p, s in COST_REGIMES
        ],
        "regimes": regimes,
        "summary_all_regimes": summarize(all_rows),
        "analysis_contract": (
            "K=6, the ranking rule, fresh seed block, three source-cost regimes, and 16 cases per regime are fixed before execution. "
            "No case is selected by full-horizon action or screening outcome. Acquisition-active subsets are reported only as post-stratified summaries of the fixed unconditional panels."
        ),
        "claim_boundary": (
            "This tests transfer across controlled source-cost regimes in static balanced worlds. "
            "It does not establish universal safe query elimination, and a zero-regret result in these panels would not imply exactness outside the tested model class."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary_all_regimes": report["summary_all_regimes"],
        "regimes": [
            {"name": r["name"], "all": r["summary_all_cases"], "active": r["summary_acquisition_active"]}
            for r in regimes
        ],
        "claim_boundary": report["claim_boundary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
