from __future__ import annotations

import json
import math
import sys
import time
from functools import lru_cache
from pathlib import Path
from random import Random

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_b.count_dp import EvidenceCountDP
from paper_b.pomcp import StaticWorldPOMCP
from paper_b.static_world import Query, World, admissible, observation_probability, terminal_loss

OUT = ROOT / "results" / "paper_b_noise_h10.json"
MODELS = ((0, 1), (0, 2, 3), (1, 4), (2, 5, 6))
STATE_BITS = 8


def make_balanced_worlds(seed: int, epsilon: float, n_worlds: int = 256) -> tuple[World, ...]:
    if not 0.0 <= epsilon <= 0.5:
        raise ValueError("epsilon must lie in [0,0.5]")
    if n_worlds % 2:
        raise ValueError("n_worlds must be even")
    rng = Random(seed)
    target = n_worlds // 2
    buckets = {0: [], 1: []}
    attempts = 0
    reliability = 1.0 - float(epsilon)
    while min(len(buckets[0]), len(buckets[1])) < target:
        attempts += 1
        if attempts > 500_000:
            raise RuntimeError("could not build balanced prior")
        state = tuple(rng.randrange(2) for _ in range(STATE_BITS))
        world = World(state, rng.randrange(len(MODELS)), reliability, reliability, 1, 1)
        y = int(admissible(world, MODELS))
        if len(buckets[y]) < target:
            buckets[y].append(world)
    worlds = buckets[0] + buckets[1]
    rng.shuffle(worlds)
    return tuple(worlds)


def uniform_belief(worlds):
    return (1.0 / len(worlds),) * len(worlds)


def query_pool(count: int = 9) -> tuple[Query, ...]:
    queries = []
    for i in range(STATE_BITS):
        queries.append(Query(f"state-{i}", "state", i, 0.025))
    for i in range(STATE_BITS):
        queries.append(Query(f"model-feature-{i}", "model_feature", i, 0.030))
    if count > len(queries):
        # A larger stress pool may repeat semantically valid sensors at different costs.
        # Repeated query types are intentional: EvidenceCountDP/POMCP permit repeated
        # noisy evidence, so these are independent sensor actions over the same facts.
        j = 0
        while len(queries) < count:
            idx = j % STATE_BITS
            kind = "state" if (j // STATE_BITS) % 2 == 0 else "model_feature"
            cost = 0.025 if kind == "state" else 0.030
            queries.append(Query(f"aux-{kind}-{j}", kind, idx, cost + 0.001 * (j % 3)))
            j += 1
    return tuple(queries[:count])


def receding_myopic_value(initial, worlds, queries, horizon, false_allow=2.0, false_block=1.0):
    """Evaluate a one-step-VoI policy over a multi-step execution horizon.

    At every posterior the policy greedily compares DECIDE with the expected
    terminal loss after exactly one additional observation. If it queries, the
    same rule is recomputed at the next posterior. The returned value is the
    full expected cost of executing that receding-myopic policy for up to
    ``horizon`` observations, making it directly comparable to the exact DP.
    """
    worlds = tuple(worlds)
    queries = tuple(queries)
    b0 = np.asarray(initial, dtype=float)
    b0 = b0 / b0.sum()
    lik = np.empty((len(queries), 2, len(worlds)), dtype=float)
    for qi, query in enumerate(queries):
        for obs in (0, 1):
            lik[qi, obs] = [observation_probability(query, w, MODELS, obs) for w in worlds]
    loss = np.asarray(
        [[terminal_loss(d, w, MODELS, false_allow=false_allow, false_block=false_block) for w in worlds] for d in (0, 1)],
        dtype=float,
    )

    @lru_cache(maxsize=None)
    def belief(counts):
        b = b0.copy()
        for qi in range(len(queries)):
            c0, c1 = counts[2 * qi], counts[2 * qi + 1]
            if c0:
                b *= np.power(lik[qi, 0], c0)
            if c1:
                b *= np.power(lik[qi, 1], c1)
        z = float(b.sum())
        if z <= 1e-300:
            return tuple(b0.tolist())
        return tuple((b / z).tolist())

    def inc(counts, qi, obs):
        xs = list(counts)
        xs[2 * qi + obs] += 1
        return tuple(xs)

    @lru_cache(maxsize=None)
    def greedy_action(counts):
        b = np.asarray(belief(counts), dtype=float)
        risks = loss @ b
        d = int(np.argmin(risks))
        best_value = float(risks[d])
        best_action = ("DECIDE", d)
        for qi, query in enumerate(queries):
            one_step = float(query.cost)
            for obs in (0, 1):
                po = float(lik[qi, obs] @ b)
                if po <= 1e-14:
                    continue
                bp = np.asarray(belief(inc(counts, qi, obs)), dtype=float)
                one_step += po * float(np.min(loss @ bp))
            action = ("QUERY", qi)
            if one_step < best_value - 1e-12 or (abs(one_step - best_value) <= 1e-12 and action < best_action):
                best_value, best_action = one_step, action
        return best_action

    @lru_cache(maxsize=None)
    def evaluate(counts):
        used = sum(counts)
        b = np.asarray(belief(counts), dtype=float)
        risks = loss @ b
        d = int(np.argmin(risks))
        if used >= horizon:
            return float(risks[d])
        action = greedy_action(counts)
        if action[0] == "DECIDE":
            return float(risks[int(action[1])])
        qi = int(action[1])
        value = float(queries[qi].cost)
        for obs in (0, 1):
            po = float(lik[qi, obs] @ b)
            if po > 1e-14:
                value += po * evaluate(inc(counts, qi, obs))
        return float(value)

    zero = (0,) * (2 * len(queries))
    return float(evaluate(zero)), greedy_action(zero), len(evaluate.cache_info().__dict__) if False else evaluate.cache_info().currsize


def run_noise_case(seed: int, epsilon: float, n_worlds: int = 256, horizon: int = 4, q_count: int = 9) -> dict:
    worlds = make_balanced_worlds(seed, epsilon, n_worlds=n_worlds)
    initial = uniform_belief(worlds)
    queries = query_pool(q_count)
    solver = EvidenceCountDP(initial, worlds, MODELS, queries, horizon=horizon)
    exact = solver.solve()
    myopic_value, myopic_action, myopic_states = receding_myopic_value(initial, worlds, queries, horizon)
    gap = float(myopic_value - exact.value)
    if gap < -1e-8:
        raise AssertionError(f"exact DP worse than receding myopic by {-gap}")
    return {
        "seed": seed,
        "epsilon": epsilon,
        "reliability": 1.0 - epsilon,
        "worlds": len(worlds),
        "queries": len(queries),
        "horizon": horizon,
        "exact_value": exact.value,
        "receding_myopic_value": myopic_value,
        "bellman_advantage": max(0.0, gap),
        "exact_action": list(exact.action),
        "receding_myopic_action": list(myopic_action),
        "root_action_disagreement": exact.action != myopic_action,
        "exact_decides_immediately": exact.action[0] == "DECIDE",
        "myopic_decides_immediately": myopic_action[0] == "DECIDE",
        "exact_states": exact.states,
        "myopic_policy_states": myopic_states,
        "exact_seconds": exact.seconds,
    }


def noisy_observation_sweep() -> dict:
    rows = []
    for epsilon in (0.0, 0.05, 0.10, 0.20):
        for seed_offset in range(4):
            rows.append(run_noise_case(20262000 + seed_offset, epsilon))
    summary = []
    for epsilon in (0.0, 0.05, 0.10, 0.20):
        group = [r for r in rows if r["epsilon"] == epsilon]
        summary.append(
            {
                "epsilon": epsilon,
                "cases": len(group),
                "mean_bellman_advantage": float(np.mean([r["bellman_advantage"] for r in group])),
                "max_bellman_advantage": max(r["bellman_advantage"] for r in group),
                "positive_advantage_fraction": sum(r["bellman_advantage"] > 1e-12 for r in group) / len(group),
                "root_action_disagreement_fraction": sum(r["root_action_disagreement"] for r in group) / len(group),
                "exact_immediate_decide_fraction": sum(r["exact_decides_immediately"] for r in group) / len(group),
                "myopic_immediate_decide_fraction": sum(r["myopic_decides_immediately"] for r in group) / len(group),
            }
        )
    return {"rows": rows, "summary": summary}


def h10_pomcp_stress() -> dict:
    rows = []
    for q_count in (9, 26):
        for seed_offset in range(3):
            worlds = make_balanced_worlds(20263000 + 100 * q_count + seed_offset, 0.10, n_worlds=512)
            initial = uniform_belief(worlds)
            queries = query_pool(q_count)
            per_horizon = {}
            for horizon in (6, 10):
                t0 = time.perf_counter()
                result = StaticWorldPOMCP(
                    initial,
                    worlds,
                    MODELS,
                    queries,
                    horizon=horizon,
                    seed=8000 + 100 * q_count + seed_offset,
                ).solve(simulations=20_000)
                per_horizon[str(horizon)] = {
                    "action": list(result.action),
                    "value": result.value,
                    "root_visits": result.root_visits,
                    "seconds": time.perf_counter() - t0,
                }
            h6, h10 = per_horizon["6"], per_horizon["10"]
            if not all(math.isfinite(float(per_horizon[h]["value"])) for h in ("6", "10")):
                raise AssertionError("non-finite POMCP value")
            rows.append(
                {
                    "queries": q_count,
                    "seed": seed_offset,
                    "worlds": len(worlds),
                    "epsilon": 0.10,
                    "simulations": 20_000,
                    "h6": h6,
                    "h10": h10,
                    "same_root_action_h6_h10": h6["action"] == h10["action"],
                    "absolute_value_shift_h6_h10": abs(float(h10["value"]) - float(h6["value"])),
                }
            )
    return {
        "rows": rows,
        "root_action_stability_fraction": sum(r["same_root_action_h6_h10"] for r in rows) / len(rows),
        "mean_absolute_value_shift_h6_h10": float(np.mean([r["absolute_value_shift_h6_h10"] for r in rows])),
    }


def main() -> None:
    report = {
        "schema": "dovod-paper-b-noise-h10-v1",
        "noisy_observation_sweep": noisy_observation_sweep(),
        "h10_pomcp_stress": h10_pomcp_stress(),
        "claim_boundary": (
            "Noise sweep uses controlled balanced hidden-world priors with known symmetric observation error epsilon and compares exact Bellman planning against the fully evaluated receding one-step-VoI policy. "
            "The h=10 block is an approximate-planning stress test at epsilon=0.10; h=6/h=10 stability is a robustness diagnostic, not an exact optimality certificate at h=10."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"schema": report["schema"], "noisy_summary": report["noisy_observation_sweep"]["summary"], "h10_summary": {k: v for k, v in report["h10_pomcp_stress"].items() if k != "rows"}, "claim_boundary": report["claim_boundary"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
