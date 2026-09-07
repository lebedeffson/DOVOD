from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path


def _read_segments(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    required = {"video", "start_frame", "end_frame", "action_id", "verb_id", "noun_id", "action_cls", "verb_cls", "noun_cls"}
    if not rows:
        return []
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"missing Assembly101 columns: {sorted(missing)}")
    out = []
    for r in rows:
        out.append({
            "video": str(r["video"]),
            "start_frame": int(r["start_frame"]),
            "end_frame": int(r["end_frame"]),
            "action_id": int(r["action_id"]),
            "verb_id": int(r["verb_id"]),
            "noun_id": int(r["noun_id"]),
            "action_cls": str(r["action_cls"]),
            "verb_cls": str(r["verb_cls"]),
            "noun_cls": str(r["noun_cls"]),
        })
    return out


def _sequences(rows: list[dict]) -> list[list[int]]:
    by_video: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_video[row["video"]].append(row)
    seqs = []
    for video in sorted(by_video):
        ordered = sorted(by_video[video], key=lambda r: (r["start_frame"], r["end_frame"], r["action_id"]))
        seq = [r["action_id"] for r in ordered]
        if len(seq) >= 2:
            seqs.append(seq)
    return seqs


def _properties(rows: list[dict]) -> dict[int, dict[str, object]]:
    props = {}
    for r in rows:
        aid = r["action_id"]
        cur = {"verb": r["verb_id"], "noun": r["noun_id"], "description": r["action_cls"]}
        if aid in props and props[aid] != cur:
            raise ValueError(f"action_id {aid} has inconsistent verb/noun mapping")
        props[aid] = cur
    return props


def _transition_counts(sequences: list[list[int]]) -> dict[int, Counter[int]]:
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    for seq in sequences:
        for a, b in zip(seq, seq[1:]):
            counts[a][b] += 1
    return counts


def _normalize(counter: Counter[int] | dict[int, float]) -> dict[int, float]:
    total = float(sum(counter.values()))
    if total <= 0:
        raise ValueError("empty prior")
    return {int(k): float(v) / total for k, v in counter.items() if v > 0}


def _stop_risk(prior: dict[int, float]) -> float:
    return 1.0 - max(prior.values())


def _map_decision(prior: dict[int, float]) -> int:
    best = max(prior.values())
    return min(k for k, p in prior.items() if abs(p - best) <= 1e-15)


def _query_outcome(action: int, query: str, props):
    return props[action][query]


def _posterior(prior, query, outcome, props):
    return _normalize({a: p for a, p in prior.items() if _query_outcome(a, query, props) == outcome})


def solve_policy(prior: dict[int, float], props, costs: dict[str, float], horizon: int):
    queries = tuple(sorted(costs))

    @lru_cache(maxsize=None)
    def rec(posterior_key, remaining, h):
        posterior = dict(posterior_key)
        best = _stop_risk(posterior)
        action = ("DECIDE", _map_decision(posterior))
        if h <= 0:
            return best, action
        for query in remaining:
            masses = defaultdict(float)
            for a, p in posterior.items():
                masses[_query_outcome(a, query, props)] += p
            if len(masses) <= 1:
                continue
            value = float(costs[query])
            next_remaining = tuple(q for q in remaining if q != query)
            for outcome, mass in masses.items():
                post = _posterior(posterior, query, outcome, props)
                sub, _ = rec(tuple(sorted(post.items())), next_remaining, h - 1)
                value += mass * sub
            candidate = ("QUERY", query)
            if value < best - 1e-12 or (abs(value - best) <= 1e-12 and candidate < action):
                best, action = value, candidate
        return float(best), action

    return rec(tuple(sorted(prior.items())), queries, int(horizon))


def _realized_cost(true_next, prior, props, costs, horizon):
    posterior = dict(prior)
    remaining = tuple(sorted(costs))
    paid = 0.0
    trace = []
    for h in range(horizon, -1, -1):
        _, action = solve_policy(posterior, props, {q: costs[q] for q in remaining}, h)
        if action[0] == "DECIDE":
            pred = int(action[1])
            return paid + float(pred != true_next), trace, pred
        query = str(action[1])
        trace.append(query)
        paid += float(costs[query])
        outcome = _query_outcome(true_next, query, props)
        posterior = _posterior(posterior, query, outcome, props)
        remaining = tuple(q for q in remaining if q != query)
    pred = _map_decision(posterior)
    return paid + float(pred != true_next), trace, pred


def _bootstrap_ci(diffs: list[float], seed: int = 20260907, draws: int = 4000):
    if not diffs:
        return [math.nan, math.nan]
    rng = random.Random(seed)
    means = []
    n = len(diffs)
    for _ in range(draws):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return [means[int(0.025 * draws)], means[min(draws - 1, int(0.975 * draws))]]


def evaluate(train_rows, test_rows, *, min_train_count: int = 5, costs=None) -> dict:
    costs = {"noun": 0.04, "verb": 0.03} if costs is None else dict(costs)
    props = _properties(train_rows + test_rows)
    train_sequences = _sequences(train_rows)
    test_sequences = _sequences(test_rows)
    counts = _transition_counts(train_sequences)
    test_transitions = [(a, b) for seq in test_sequences for a, b in zip(seq, seq[1:])]
    rows = []
    for current, true_next in test_transitions:
        counter = counts.get(current)
        if not counter or sum(counter.values()) < min_train_count or len(counter) < 2 or true_next not in counter:
            continue
        prior = _normalize(counter)
        exact_value, exact_root = solve_policy(prior, props, costs, horizon=2)
        myopic_value, myopic_root = solve_policy(prior, props, costs, horizon=1)
        exact_cost, exact_trace, exact_pred = _realized_cost(true_next, prior, props, costs, horizon=2)
        myopic_cost, myopic_trace, myopic_pred = _realized_cost(true_next, prior, props, costs, horizon=1)
        rows.append({
            "current_action": current,
            "true_next_action": true_next,
            "candidates": len(prior),
            "train_support": int(sum(counter.values())),
            "exact_expected_value": exact_value,
            "myopic_expected_value": myopic_value,
            "exact_root_action": list(exact_root),
            "myopic_root_action": list(myopic_root),
            "root_actions_differ": exact_root != myopic_root,
            "exact_realized_cost": exact_cost,
            "myopic_realized_cost": myopic_cost,
            "realized_cost_reduction": myopic_cost - exact_cost,
            "exact_trace": exact_trace,
            "myopic_trace": myopic_trace,
            "exact_prediction": exact_pred,
            "myopic_prediction": myopic_pred,
        })
    diffs = [r["exact_realized_cost"] - r["myopic_realized_cost"] for r in rows]
    return {
        "coverage": {
            "train_sequences": len(train_sequences),
            "test_sequences": len(test_sequences),
            "raw_test_transitions": len(test_transitions),
            "evaluated_transitions": len(rows),
            "coverage_fraction": 0.0 if not test_transitions else len(rows) / len(test_transitions),
            "min_train_transition_support": min_train_count,
        },
        "comparison": {
            "exact_mean_realized_cost": None if not rows else sum(r["exact_realized_cost"] for r in rows) / len(rows),
            "myopic_mean_realized_cost": None if not rows else sum(r["myopic_realized_cost"] for r in rows) / len(rows),
            "root_action_disagreement_fraction": 0.0 if not rows else sum(r["root_actions_differ"] for r in rows) / len(rows),
            "exact_better_tied_worse": [
                sum(r["realized_cost_reduction"] > 1e-15 for r in rows),
                sum(abs(r["realized_cost_reduction"]) <= 1e-15 for r in rows),
                sum(r["realized_cost_reduction"] < -1e-15 for r in rows),
            ],
            "paired_exact_minus_myopic_bootstrap_ci95": _bootstrap_ci(diffs),
        },
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations-dir", required=True, help="Directory containing Assembly101 fine-grained train.csv and test.csv")
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-train-count", type=int, default=5)
    args = ap.parse_args()
    root = Path(args.annotations_dir)
    train = _read_segments(root / "train.csv")
    test = _read_segments(root / "test.csv")
    result = evaluate(train, test, min_train_count=args.min_train_count)
    report = {
        "schema": "dovod-paper-b-assembly101-acquisition-v1",
        "external_dataset": {
            "name": "Assembly101 fine-grained action annotations",
            "annotation_schema_source": "assembly-101/assembly101-annotations",
            "required_files": ["train.csv", "test.csv"],
        },
        "adapter": {
            "target": "next fine-grained action conditioned on current action",
            "training_prior": "empirical next-action distribution from official train annotations",
            "test": "official test annotations",
            "query_types": {"verb": "reveals next-action verb id", "noun": "reveals next-action noun/object id"},
            "normalized_query_costs": {"verb": 0.03, "noun": 0.04},
            "exact_horizon": 2,
            "myopic_horizon": 1,
        },
        **result,
        "claim_boundary": (
            "Third-dataset workload-structure test using only Assembly101 action annotations. "
            "Verb/noun attributes are typed information channels, not physical-vs-semantic source labels, and costs are normalized decision-loss units rather than measured effort. "
            "Therefore this benchmark can support cross-dataset non-myopic workload claims but cannot by itself validate persistent source reliability/orientation."
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
