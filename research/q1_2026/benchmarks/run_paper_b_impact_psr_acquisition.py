from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path


def _frame_number(text: str) -> int:
    return int(Path(text).stem)


def _load_sequence(folder: Path) -> list[int]:
    path = folder / "PSR_labels.csv"
    by_frame: dict[int, list[int]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if not row:
                continue
            by_frame[_frame_number(row[0])].append(int(row[1]))
    return [ids[0] for _, ids in sorted(by_frame.items()) if len(ids) == 1]


def _load_split(root: Path, split: str) -> list[list[int]]:
    split_dir = root / "dataset" / "PSR" / "labels" / split
    return [_load_sequence(folder) for folder in sorted(split_dir.iterdir()) if folder.is_dir()]


def _step_properties(root: Path) -> dict[int, dict[str, object]]:
    path = root / "dataset" / "PSR" / "labels" / "procedure_info_IMPACT.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for row in rows:
        desc = str(row["description"])
        out[int(row["id"])] = {
            "component": int(row["state_idx"]),
            "operation": "install" if bool(row["install"]) else "remove",
            "quality": "incorrect" if desc.lower().startswith("incorrectly") else "normal",
            "description": desc,
        }
    return out


def _transition_counts(sequences: list[list[int]]) -> dict[int, Counter[int]]:
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    for seq in sequences:
        for a, b in zip(seq, seq[1:]):
            counts[a][b] += 1
    return counts


def _query_outcome(step: int, query: str, props):
    return props[step][query]


def _normalize(weights: dict[int, float]) -> dict[int, float]:
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("posterior mass must be positive")
    return {k: v / total for k, v in weights.items() if v > 0}


def _posterior_after(prior: dict[int, float], query: str, outcome, props) -> dict[int, float]:
    return _normalize({step: p for step, p in prior.items() if _query_outcome(step, query, props) == outcome})


def _map_decision(prior: dict[int, float]) -> int:
    best = max(prior.values())
    return min(step for step, p in prior.items() if abs(p - best) <= 1e-15)


def _stop_risk(prior: dict[int, float]) -> float:
    return 1.0 - max(prior.values())


def solve_policy(prior: dict[int, float], props, costs: dict[str, float], horizon: int):
    queries = tuple(sorted(costs))
    initial_key = tuple(sorted(prior.items()))

    @lru_cache(maxsize=None)
    def rec(posterior_key, remaining_queries, h):
        posterior = dict(posterior_key)
        best_value = _stop_risk(posterior)
        best_action = ("DECIDE", _map_decision(posterior))
        if h <= 0:
            return best_value, best_action
        for query in remaining_queries:
            outcome_mass = defaultdict(float)
            for step, p in posterior.items():
                outcome_mass[_query_outcome(step, query, props)] += p
            if len(outcome_mass) <= 1:
                continue
            value = float(costs[query])
            next_remaining = tuple(q for q in remaining_queries if q != query)
            for outcome, mass in outcome_mass.items():
                post = _posterior_after(posterior, query, outcome, props)
                sub_value, _ = rec(tuple(sorted(post.items())), next_remaining, h - 1)
                value += mass * sub_value
            candidate = ("QUERY", query)
            if value < best_value - 1e-12 or (abs(value - best_value) <= 1e-12 and candidate < best_action):
                best_value, best_action = value, candidate
        return best_value, best_action

    value, action = rec(initial_key, queries, horizon)

    def policy(posterior: dict[int, float], remaining: tuple[str, ...], h: int):
        return rec(tuple(sorted(posterior.items())), remaining, h)[1]

    return value, action, policy


def _realized_cost(true_step: int, prior: dict[int, float], props, costs, horizon: int) -> tuple[float, list[str], int]:
    _, _, policy = solve_policy(prior, props, costs, horizon)
    posterior = dict(prior)
    remaining = tuple(sorted(costs))
    paid = 0.0
    trace = []
    for h in range(horizon, -1, -1):
        action = policy(posterior, remaining, h)
        if action[0] == "DECIDE":
            pred = int(action[1])
            return paid + float(pred != true_step), trace, pred
        query = str(action[1])
        trace.append(query)
        paid += float(costs[query])
        outcome = _query_outcome(true_step, query, props)
        posterior = _posterior_after(posterior, query, outcome, props)
        remaining = tuple(q for q in remaining if q != query)
    pred = _map_decision(posterior)
    return paid + float(pred != true_step), trace, pred


def _bootstrap_ci(diffs: list[float], seed: int = 20260907, draws: int = 4000) -> tuple[float, float]:
    if not diffs:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    means = []
    n = len(diffs)
    for _ in range(draws):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * draws)], means[min(draws - 1, int(0.975 * draws))]


def _evaluate_costs(test_transitions, priors, props, costs, min_train_count):
    rows = []
    for current, true_next in test_transitions:
        counter = priors.get(current)
        if counter is None or sum(counter.values()) < min_train_count or len(counter) < 2 or true_next not in counter:
            continue
        prior = _normalize({step: float(count) for step, count in counter.items()})
        exact_value, exact_root, _ = solve_policy(prior, props, costs, horizon=3)
        myopic_value, myopic_root, _ = solve_policy(prior, props, costs, horizon=1)
        exact_cost, exact_trace, exact_pred = _realized_cost(true_next, prior, props, costs, horizon=3)
        myopic_cost, myopic_trace, myopic_pred = _realized_cost(true_next, prior, props, costs, horizon=1)
        rows.append({
            "current_step": current,
            "true_next_step": true_next,
            "current_description": props[current]["description"],
            "true_next_description": props[true_next]["description"],
            "candidate_next_steps": len(prior),
            "train_transition_support": int(sum(counter.values())),
            "exact_expected_value": exact_value,
            "myopic_expected_value": myopic_value,
            "exact_root_action": list(exact_root),
            "myopic_root_action": list(myopic_root),
            "root_actions_differ": exact_root != myopic_root,
            "exact_realized_cost": exact_cost,
            "myopic_realized_cost": myopic_cost,
            "realized_cost_reduction": myopic_cost - exact_cost,
            "exact_query_trace": exact_trace,
            "myopic_query_trace": myopic_trace,
            "exact_prediction": exact_pred,
            "myopic_prediction": myopic_pred,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impact-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-train-count", type=int, default=5)
    args = ap.parse_args()
    root = Path(args.impact_root)
    props = _step_properties(root)
    train_sequences = _load_split(root, "train")
    test_sequences = _load_split(root, "test")
    train_counts = _transition_counts(train_sequences)
    test_transitions = [(a, b) for seq in test_sequences for a, b in zip(seq, seq[1:])]
    base_costs = {"component": 0.04, "operation": 0.03, "quality": 0.04}
    rows = _evaluate_costs(test_transitions, train_counts, props, base_costs, args.min_train_count)
    diffs = [r["exact_realized_cost"] - r["myopic_realized_cost"] for r in rows]
    ci = _bootstrap_ci(diffs)
    positive = sorted((r for r in rows if r["realized_cost_reduction"] > 1e-15), key=lambda r: -r["realized_cost_reduction"])
    total_gain = sum(max(0.0, r["realized_cost_reduction"]) for r in rows)
    top_k = max(1, int(math.ceil(0.2 * len(rows)))) if rows else 0
    top_gain = sum(max(0.0, r["realized_cost_reduction"]) for r in sorted(rows, key=lambda r: -r["realized_cost_reduction"])[:top_k]) if rows else 0.0
    cost_scales = (0.5, 1.0, 2.0, 5.0, 10.0)
    sensitivity = []
    switch_example = None
    for row in rows:
        current = row["current_step"]
        true_next = row["true_next_step"]
        counter = train_counts[current]
        prior = _normalize({step: float(count) for step, count in counter.items()})
        actions = {}
        for scale in cost_scales:
            costs = {q: c * scale for q, c in base_costs.items()}
            _, root_action, _ = solve_policy(prior, props, costs, horizon=3)
            actions[str(scale)] = list(root_action)
        kinds = {tuple(a) for a in actions.values()}
        if len(kinds) > 1 and switch_example is None:
            switch_example = {
                "current_step": current,
                "current_description": props[current]["description"],
                "true_next_step": true_next,
                "true_next_description": props[true_next]["description"],
                "root_action_by_uniform_cost_scale": actions,
            }
        sensitivity.append({"current_step": current, "true_next_step": true_next, "distinct_root_actions": len(kinds)})

    report = {
        "schema": "dovod-paper-b-impact-psr-acquisition-v1",
        "external_dataset": {
            "name": "IMPACT v1.1 Procedure Step Recognition annotations",
            "source_repo": "Kratos-Wen/IMPACT",
            "source_commit": "4fed5faa5f05f7aece55712e458defa1f372b248",
            "source_subtree": "dataset/PSR/labels",
        },
        "adapter": {
            "target": "next singleton PSR step conditioned on current singleton PSR step",
            "training_prior": "empirical next-step distribution on official train split",
            "test": "official test split; transitions unseen under the train conditional support are reported as uncovered",
            "query_types": {
                "component": "reveals state_idx/component of the next official PSR step",
                "operation": "reveals install-vs-remove property of the next official PSR step",
                "quality": "reveals normal-vs-incorrect-installation property of the next official PSR step",
            },
            "base_normalized_query_costs": base_costs,
            "decision_loss": "0-1 next-step classification loss",
            "exact_horizon": 3,
            "myopic_horizon": 1,
            "excluded_concurrent_events": "frames with multiple simultaneous PSR step labels are excluded rather than arbitrarily ordered",
        },
        "coverage": {
            "train_sequences": len(train_sequences),
            "test_sequences": len(test_sequences),
            "raw_test_transitions": len(test_transitions),
            "evaluated_transitions": len(rows),
            "coverage_fraction": 0.0 if not test_transitions else len(rows) / len(test_transitions),
            "min_train_transition_support": args.min_train_count,
        },
        "comparison": {
            "exact_mean_realized_cost": None if not rows else sum(r["exact_realized_cost"] for r in rows) / len(rows),
            "myopic_mean_realized_cost": None if not rows else sum(r["myopic_realized_cost"] for r in rows) / len(rows),
            "relative_reduction": None if not rows else (sum(r["myopic_realized_cost"] for r in rows) - sum(r["exact_realized_cost"] for r in rows)) / max(1e-15, sum(r["myopic_realized_cost"] for r in rows)),
            "paired_exact_minus_myopic_bootstrap_ci95": list(ci),
            "root_action_disagreement_fraction": 0.0 if not rows else sum(r["root_actions_differ"] for r in rows) / len(rows),
            "transitions_exact_better": sum(r["realized_cost_reduction"] > 1e-15 for r in rows),
            "transitions_tied": sum(abs(r["realized_cost_reduction"]) <= 1e-15 for r in rows),
            "transitions_exact_worse": sum(r["realized_cost_reduction"] < -1e-15 for r in rows),
            "top_20pct_share_of_positive_gain": None if total_gain <= 1e-15 else top_gain / total_gain,
        },
        "cost_sensitivity": {
            "uniform_scales": list(cost_scales),
            "transitions_with_root_action_switch": sum(r["distinct_root_actions"] > 1 for r in sensitivity),
            "switch_fraction": 0.0 if not sensitivity else sum(r["distinct_root_actions"] > 1 for r in sensitivity) / len(sensitivity),
            "representative_switch": switch_example,
        },
        "largest_realized_improvements": positive[:10],
        "rows": rows,
        "claim_boundary": (
            "This is a secondary post-reviewer procedural benchmark on official IMPACT PSR annotations. "
            "The next-step prior is learned only from the official train split, and evaluated transitions come from the official test split. "
            "Query outcomes are deterministic properties of the annotated next step and query costs are normalized decision-loss units, not measured industrial costs. "
            "The experiment tests multi-step typed acquisition on a second real procedural transition distribution; it does not estimate sensor latency, money, or human effort."
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
