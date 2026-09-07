from __future__ import annotations

import json
import sys
from math import isfinite
from pathlib import Path
from random import Random
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_b.count_dp import EvidenceCountDP, evidence_count_state_count, ordered_history_state_count
from paper_b.pomcp import StaticWorldPOMCP
from paper_b.static_world import Query, World, observation_probability, terminal_loss

OUT = ROOT / "results" / "paper_b_scalability_robustness.json"

MODELS = (
    (0, 1, 2),
    (0, 3, 4),
    (1, 5, 6),
    (2, 7, 8),
)
STATE_BITS = 12


def make_worlds(seed: int, n_worlds: int = 512) -> tuple[World, ...]:
    rng = Random(seed)
    worlds = []
    reliabilities = (0.75, 0.85, 0.95)
    for _ in range(n_worlds):
        state = tuple(rng.randrange(2) for _ in range(STATE_BITS))
        model = rng.randrange(len(MODELS))
        rp = reliabilities[rng.randrange(len(reliabilities))]
        rs = reliabilities[rng.randrange(len(reliabilities))]
        op = -1 if rng.random() < 0.12 else 1
        os = -1 if rng.random() < 0.10 else 1
        worlds.append(World(state, model, rp, rs, op, os))
    return tuple(worlds)


def query_pool(semantic_multiplier: float = 1.0) -> tuple[Query, ...]:
    queries = [
        Query("calibrate-physical", "calibrate_physical", 0, 0.010),
        Query("calibrate-semantic", "calibrate_semantic", 0, 0.010 * semantic_multiplier),
    ]
    for i in range(STATE_BITS):
        queries.append(Query(f"state-{i}", "state", i, 0.012))
        queries.append(Query(f"model-feature-{i}", "model_feature", i, 0.014 * semantic_multiplier))
    return tuple(queries)


def uniform_belief(worlds: tuple[World, ...]) -> tuple[float, ...]:
    return (1.0 / len(worlds),) * len(worlds)


def myopic_root_action(belief, worlds, queries, false_allow=2.0, false_block=1.0):
    b = np.asarray(belief, dtype=float)
    loss = np.asarray(
        [[terminal_loss(d, w, MODELS, false_allow=false_allow, false_block=false_block) for w in worlds] for d in (0, 1)],
        dtype=float,
    )
    risks = loss @ b
    best_action = ("DECIDE", int(np.argmin(risks)))
    best_value = float(np.min(risks))
    for qi, q in enumerate(queries):
        value = float(q.cost)
        for obs in (0, 1):
            lik = np.asarray([observation_probability(q, w, MODELS, obs) for w in worlds], dtype=float)
            po = float(lik @ b)
            if po <= 1e-14:
                continue
            post = b * lik
            post /= post.sum()
            value += po * float(np.min(loss @ post))
        if value < best_value - 1e-12:
            best_value = value
            best_action = ("QUERY", qi)
    return best_action, best_value


def exact_vs_pomcp_validation() -> dict:
    rows = []
    for seed in range(12):
        worlds = make_worlds(20260907 + seed)
        belief = uniform_belief(worlds)
        queries = query_pool()[:9]
        exact_solver = EvidenceCountDP(belief, worlds, MODELS, queries, horizon=3)
        exact = exact_solver.solve()
        action_values = exact_solver.root_action_values()
        optimum = min(action_values.values())
        optimal_actions = {a for a, v in action_values.items() if v <= optimum + 1e-12}
        t0 = perf_counter()
        approx = StaticWorldPOMCP(belief, worlds, MODELS, queries, horizon=3, seed=9000 + seed).solve(simulations=15000)
        seconds = perf_counter() - t0
        rows.append({
            "seed": seed,
            "worlds": len(worlds),
            "queries": len(queries),
            "horizon": 3,
            "exact_value": exact.value,
            "exact_action": list(exact.action),
            "pomcp_action": list(approx.action),
            "pomcp_value": approx.value,
            "pomcp_action_is_exact_optimal": approx.action in optimal_actions,
            "absolute_value_error": abs(approx.value - exact.value),
            "pomcp_seconds": seconds,
        })
    return {
        "rows": rows,
        "cases": len(rows),
        "exact_optimal_action_rate": sum(r["pomcp_action_is_exact_optimal"] for r in rows) / len(rows),
        "mean_absolute_value_error": sum(r["absolute_value_error"] for r in rows) / len(rows),
        "max_absolute_value_error": max(r["absolute_value_error"] for r in rows),
        "mean_pomcp_seconds": sum(r["pomcp_seconds"] for r in rows) / len(rows),
    }


def large_query_stress() -> dict:
    rows = []
    for q_count in (12, 18, 24, 26):
        for seed in range(3):
            worlds = make_worlds(20261000 + 100 * q_count + seed)
            belief = uniform_belief(worlds)
            queries = query_pool()[:q_count]
            myopic_action, myopic_value = myopic_root_action(belief, worlds, queries)
            t0 = perf_counter()
            approx = StaticWorldPOMCP(belief, worlds, MODELS, queries, horizon=6, seed=7000 + seed).solve(simulations=10000)
            seconds = perf_counter() - t0
            rows.append({
                "queries": q_count,
                "seed": seed,
                "worlds": len(worlds),
                "horizon": 6,
                "count_dp_state_envelope": evidence_count_state_count(q_count, 6),
                "ordered_history_state_envelope": ordered_history_state_count(q_count, 6),
                "pomcp_seconds": seconds,
                "pomcp_action": list(approx.action),
                "myopic_action": list(myopic_action),
                "same_root_action_as_myopic": approx.action == myopic_action,
                "myopic_one_step_value": myopic_value,
            })
    return {"rows": rows}


def cost_grid_minimax_regret() -> dict:
    multipliers = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0)
    rows = []
    improvements = []
    for seed in range(8):
        worlds = make_worlds(20261100 + seed)
        belief = uniform_belief(worlds)
        q_values_by_cost = {}
        opt_by_cost = {}
        for c in multipliers:
            queries = query_pool(c)[:9]
            solver = EvidenceCountDP(belief, worlds, MODELS, queries, horizon=3)
            vals = solver.root_action_values()
            q_values_by_cost[c] = vals
            opt_by_cost[c] = min(vals.values())

        actions = set.intersection(*(set(v.keys()) for v in q_values_by_cost.values()))
        regrets = {
            action: max(q_values_by_cost[c][action] - opt_by_cost[c] for c in multipliers)
            for action in actions
        }
        robust_action = min(regrets, key=lambda a: (regrets[a], a))
        nominal_vals = q_values_by_cost[1.0]
        nominal_opt = min(nominal_vals.values())
        nominal_action = min((a for a, v in nominal_vals.items() if v <= nominal_opt + 1e-12), key=lambda a: a)
        nominal_worst = max(q_values_by_cost[c][nominal_action] - opt_by_cost[c] for c in multipliers)
        robust_worst = regrets[robust_action]
        improvements.append(nominal_worst - robust_worst)
        rows.append({
            "seed": seed,
            "cost_multipliers": list(multipliers),
            "nominal_action": list(nominal_action),
            "minimax_regret_action": list(robust_action),
            "nominal_worst_case_regret": nominal_worst,
            "minimax_worst_case_regret": robust_worst,
            "worst_case_regret_reduction": nominal_worst - robust_worst,
            "optimal_action_switches_across_grid": len({
                min((a for a, v in q_values_by_cost[c].items() if v <= opt_by_cost[c] + 1e-12), key=lambda a: a)
                for c in multipliers
            }) - 1,
        })
    return {
        "rows": rows,
        "cases": len(rows),
        "cases_where_minimax_reduces_worst_case_regret": sum(x > 1e-12 for x in improvements),
        "mean_worst_case_regret_reduction": sum(improvements) / len(improvements),
        "max_worst_case_regret_reduction": max(improvements),
    }


def main() -> None:
    report = {
        "schema": "dovod-paper-b-scalability-robustness-v1",
        "exact_vs_pomcp": exact_vs_pomcp_validation(),
        "large_query_stress": large_query_stress(),
        "cost_uncertainty": cost_grid_minimax_regret(),
        "claim_boundary": (
            "The exact-vs-POMCP block uses controlled static hidden worlds and validates approximate root-action recovery only within that class. "
            "The large-query block is an engineering scalability stress and has no exact optimality claim at Q>9. "
            "The minimax-regret block is a discrete cost-grid robustness diagnostic, not an estimate of real operational costs. "
            "Real procedural usefulness remains supported separately by the frozen MECCANO Bellman-vs-myopic evaluation."
        ),
    }
    assert all(isfinite(float(r["pomcp_seconds"])) for r in report["large_query_stress"]["rows"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
