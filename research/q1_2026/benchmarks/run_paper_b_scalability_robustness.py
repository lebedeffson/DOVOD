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
from paper_b.static_world import Query, World, admissible, observation_probability, terminal_loss

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


def make_balanced_worlds(seed: int, n_worlds: int = 512) -> tuple[World, ...]:
    """Controlled prior with equal admissible/non-admissible mass.

    Balancing prevents the trivial always-block prior that appeared in the first
    unconditional cost diagnostic. Selection depends only on ground-truth class,
    not on any planner outcome.
    """
    rng = Random(seed)
    target = n_worlds // 2
    buckets = {0: [], 1: []}
    reliabilities = (0.75, 0.85, 0.95)
    attempts = 0
    while min(len(buckets[0]), len(buckets[1])) < target:
        attempts += 1
        if attempts > 200000:
            raise RuntimeError("could not construct balanced hidden-world prior")
        state = tuple(rng.randrange(2) for _ in range(STATE_BITS))
        model = rng.randrange(len(MODELS))
        w = World(
            state,
            model,
            reliabilities[rng.randrange(len(reliabilities))],
            reliabilities[rng.randrange(len(reliabilities))],
            -1 if rng.random() < 0.12 else 1,
            -1 if rng.random() < 0.10 else 1,
        )
        y = int(admissible(w, MODELS))
        if len(buckets[y]) < target:
            buckets[y].append(w)
    worlds = buckets[0] + buckets[1]
    rng.shuffle(worlds)
    return tuple(worlds)


def cost_sensitive_queries(semantic_multiplier: float) -> tuple[Query, ...]:
    """Fixed query vocabulary for semantic-cost uncertainty stress."""
    qs = [
        Query("calibrate-semantic", "calibrate_semantic", 0, 0.025 * semantic_multiplier),
    ]
    for i in range(4):
        qs.append(Query(f"state-{i}", "state", i, 0.040))
        qs.append(Query(f"model-feature-{i}", "model_feature", i, 0.035 * semantic_multiplier))
    return tuple(qs[:9])


def unconditional_random_prior_cost_diagnostic() -> dict:
    """Preserve the negative result that motivated the targeted stress."""
    multipliers = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0)
    rows = []
    for seed in range(8):
        worlds = make_worlds(20261100 + seed)
        belief = uniform_belief(worlds)
        actions = []
        for c in multipliers:
            solver = EvidenceCountDP(belief, worlds, MODELS, query_pool(c)[:9], horizon=3)
            vals = solver.root_action_values()
            opt = min(vals.values())
            action = min((a for a, v in vals.items() if v <= opt + 1e-12), key=lambda a: a)
            actions.append(action)
        rows.append({
            "seed": seed,
            "distinct_optimal_root_actions": len(set(actions)),
            "all_immediate_decisions": all(a[0] == "DECIDE" for a in actions),
        })
    return {
        "cases": len(rows),
        "rows": rows,
        "all_cases_trivial_immediate_decision": all(r["all_immediate_decisions"] for r in rows),
        "interpretation": (
            "Unconditioned random priors were dominated by inadmissible worlds, so all tested costs led to immediate DECIDE. "
            "This negative diagnostic is retained and is not used as evidence of cost robustness."
        ),
    }


def acquisition_active_cost_minimax_regret() -> dict:
    """First-action minimax-regret stress on semantic-cost-active beliefs.

    Cases are selected by a predeclared mechanism criterion: at nominal semantic
    cost 1, the exact planner must choose a semantic query. No robust-policy
    performance is inspected during selection.
    """
    multipliers = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0)
    selected = []
    candidate_seed = 0
    while len(selected) < 8 and candidate_seed < 80:
        seed = 20261200 + candidate_seed
        candidate_seed += 1
        worlds = make_balanced_worlds(seed)
        belief = uniform_belief(worlds)
        nominal_queries = cost_sensitive_queries(1.0)
        nominal_solver = EvidenceCountDP(belief, worlds, MODELS, nominal_queries, horizon=3)
        vals = nominal_solver.root_action_values()
        opt = min(vals.values())
        action = min((a for a, v in vals.items() if v <= opt + 1e-12), key=lambda a: a)
        if action[0] != "QUERY":
            continue
        q = nominal_queries[int(action[1])]
        if q.kind not in ("model_feature", "calibrate_semantic"):
            continue
        selected.append((seed, worlds, belief))
    if len(selected) < 8:
        raise RuntimeError(f"only {len(selected)} semantic-acquisition-active cases found")

    rows = []
    improvements = []
    switch_cases = 0
    for case_id, (seed, worlds, belief) in enumerate(selected):
        q_values_by_cost = {}
        opt_by_cost = {}
        optimal_action_by_cost = {}
        for c in multipliers:
            queries = cost_sensitive_queries(c)
            solver = EvidenceCountDP(belief, worlds, MODELS, queries, horizon=3)
            vals = solver.root_action_values()
            q_values_by_cost[c] = vals
            opt = min(vals.values())
            opt_by_cost[c] = opt
            optimal_action_by_cost[c] = min((a for a, v in vals.items() if v <= opt + 1e-12), key=lambda a: a)

        actions = set.intersection(*(set(v.keys()) for v in q_values_by_cost.values()))
        regrets = {
            action: max(q_values_by_cost[c][action] - opt_by_cost[c] for c in multipliers)
            for action in actions
        }
        robust_action = min(regrets, key=lambda a: (regrets[a], a))
        nominal_action = optimal_action_by_cost[1.0]
        nominal_worst = max(q_values_by_cost[c][nominal_action] - opt_by_cost[c] for c in multipliers)
        robust_worst = regrets[robust_action]
        improvement = nominal_worst - robust_worst
        improvements.append(improvement)
        distinct = len(set(optimal_action_by_cost.values()))
        switch_cases += int(distinct > 1)
        rows.append({
            "case": case_id,
            "seed": seed,
            "worlds": len(worlds),
            "cost_multipliers": list(multipliers),
            "nominal_action": list(nominal_action),
            "minimax_regret_action": list(robust_action),
            "nominal_worst_case_first_action_regret": nominal_worst,
            "minimax_worst_case_first_action_regret": robust_worst,
            "worst_case_regret_reduction": improvement,
            "distinct_optimal_actions_across_cost_grid": distinct,
            "optimal_actions_by_cost": {str(c): list(optimal_action_by_cost[c]) for c in multipliers},
        })
    return {
        "selection_rule": "first eight balanced-prior seeds whose exact nominal-cost root action is a semantic query",
        "candidate_seeds_examined": candidate_seed,
        "cases": len(rows),
        "cases_with_optimal_action_switch_across_cost_grid": switch_cases,
        "cases_where_minimax_reduces_worst_case_first_action_regret": sum(x > 1e-12 for x in improvements),
        "mean_worst_case_first_action_regret_reduction": sum(improvements) / len(improvements),
        "max_worst_case_first_action_regret_reduction": max(improvements),
        "rows": rows,
    }


def main() -> None:
    report = {
        "schema": "dovod-paper-b-scalability-robustness-v1",
        "exact_vs_pomcp": exact_vs_pomcp_validation(),
        "large_query_stress": large_query_stress(),
        "cost_uncertainty": {
            "unconditional_negative_diagnostic": unconditional_random_prior_cost_diagnostic(),
            "acquisition_active_minimax_regret": acquisition_active_cost_minimax_regret(),
        },
        "claim_boundary": (
            "The exact-vs-POMCP block uses controlled static hidden worlds and validates approximate root-action recovery only within that class. "
            "The large-query block is an engineering scalability stress and has no exact optimality claim at Q>9. "
            "The minimax-regret block is a discrete cost-grid first-action robustness diagnostic on predeclared semantic-acquisition-active controlled beliefs, not an estimate of real operational costs. "
            "Real procedural usefulness remains supported separately by the frozen MECCANO Bellman-vs-myopic evaluation."
        ),
    }
    assert all(isfinite(float(r["pomcp_seconds"])) for r in report["large_query_stress"]["rows"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
