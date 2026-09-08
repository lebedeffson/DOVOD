from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

BASE_COSTS = {"noun": 0.04, "verb": 0.03}
SUPPORT_SENSITIVITY = (3, 5, 10)
COST_MULTIPLIERS = (0.5, 1.0, 2.0, 5.0)


def _read_segments(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    required = {
        "video",
        "start_frame",
        "end_frame",
        "action_id",
        "verb_id",
        "noun_id",
        "action_cls",
        "verb_cls",
        "noun_cls",
    }
    if not rows:
        return []
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"missing Assembly101 columns: {sorted(missing)}")
    out = []
    for r in rows:
        out.append(
            {
                "video": str(r["video"]),
                "start_frame": int(r["start_frame"]),
                "end_frame": int(r["end_frame"]),
                "action_id": int(r["action_id"]),
                "verb_id": int(r["verb_id"]),
                "noun_id": int(r["noun_id"]),
                "action_cls": str(r["action_cls"]),
                "verb_cls": str(r["verb_cls"]),
                "noun_cls": str(r["noun_cls"]),
            }
        )
    return out


def _sequences(rows: list[dict]) -> list[list[int]]:
    by_video: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_video[row["video"]].append(row)
    seqs = []
    for video in sorted(by_video):
        ordered = sorted(
            by_video[video],
            key=lambda r: (r["start_frame"], r["end_frame"], r["action_id"]),
        )
        seq = [r["action_id"] for r in ordered]
        if len(seq) >= 2:
            seqs.append(seq)
    return seqs


def _row_property(row: dict) -> dict[str, object]:
    return {
        "verb": int(row["verb_id"]),
        "noun": int(row["noun_id"]),
        "description": str(row["action_cls"]),
    }


def _properties(rows: list[dict]) -> dict[int, dict[str, object]]:
    """Build action metadata from one declared split only.

    The external evaluation calls this on the official training split. Test rows
    are never used to construct query outcomes or candidate metadata.
    """
    props: dict[int, dict[str, object]] = {}
    for row in rows:
        aid = int(row["action_id"])
        cur = _row_property(row)
        if aid in props and props[aid] != cur:
            raise ValueError(f"action_id {aid} has inconsistent metadata within split")
        props[aid] = cur
    return props


def _validate_test_metadata(train_props: dict[int, dict[str, object]], test_rows: list[dict]) -> dict:
    """Validate shared action IDs without importing test-only metadata.

    Only verb/noun IDs matter to the query model. Unknown test action IDs are
    counted but deliberately remain absent from ``train_props``.
    """
    test_ids: set[int] = set()
    unknown_ids: set[int] = set()
    shared_occurrences = 0
    unknown_occurrences = 0
    for row in test_rows:
        aid = int(row["action_id"])
        test_ids.add(aid)
        if aid not in train_props:
            unknown_ids.add(aid)
            unknown_occurrences += 1
            continue
        shared_occurrences += 1
        expected = train_props[aid]
        if int(row["verb_id"]) != int(expected["verb"]) or int(row["noun_id"]) != int(expected["noun"]):
            raise ValueError(f"action_id {aid} has train/test verb/noun metadata mismatch")
    return {
        "train_action_ids": len(train_props),
        "test_action_ids": len(test_ids),
        "shared_test_action_ids": len(test_ids.difference(unknown_ids)),
        "unknown_test_action_ids": len(unknown_ids),
        "unknown_test_action_id_values": sorted(unknown_ids),
        "shared_test_row_occurrences": shared_occurrences,
        "unknown_test_row_occurrences": unknown_occurrences,
        "metadata_source_for_policy": "official_train_only",
    }


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


def _weighted_bootstrap_ci(diff_counts: Counter[float], *, seed: int = 20260908, draws: int = 4000):
    n = int(sum(diff_counts.values()))
    if n <= 0 or draws <= 0:
        return None
    values = np.asarray(sorted(diff_counts), dtype=float)
    counts = np.asarray([diff_counts[float(v)] for v in values], dtype=float)
    probs = counts / counts.sum()
    rng = np.random.default_rng(seed)
    sampled = rng.multinomial(n, probs, size=int(draws))
    means = sampled @ values / n
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _weighted_mean(total: float, n: int):
    return None if n <= 0 else float(total / n)


def evaluate(
    train_rows,
    test_rows,
    *,
    min_train_count: int = 5,
    costs=None,
    include_rows: bool = True,
    bootstrap_draws: int = 4000,
) -> dict:
    costs = dict(BASE_COSTS if costs is None else costs)

    # Leakage barrier: query metadata is constructed from train only.
    props = _properties(train_rows)
    metadata = _validate_test_metadata(props, test_rows)

    train_sequences = _sequences(train_rows)
    test_sequences = _sequences(test_rows)
    counts = _transition_counts(train_sequences)
    test_transitions = [(a, b) for seq in test_sequences for a, b in zip(seq, seq[1:])]
    pair_counts = Counter(test_transitions)

    skip = Counter()
    rows = []
    evaluated = 0
    exact_total = 0.0
    myopic_total = 0.0
    disagreement_total = 0
    better = tied = worse = 0
    positive_reduction_total = 0.0
    negative_reduction_total = 0.0
    diff_counts: Counter[float] = Counter()

    for (current, true_next), frequency in sorted(pair_counts.items()):
        counter = counts.get(current)
        if not counter:
            skip["no_train_prior"] += frequency
            continue
        train_support = int(sum(counter.values()))
        if train_support < min_train_count:
            skip["below_train_support_threshold"] += frequency
            continue
        if len(counter) < 2:
            skip["single_train_candidate"] += frequency
            continue
        if true_next not in counter:
            skip["true_next_unseen_after_current_in_train"] += frequency
            continue
        if current not in props or true_next not in props or any(a not in props for a in counter):
            raise AssertionError("evaluated candidate metadata must come from train split")

        prior = _normalize(counter)
        exact_value, exact_root = solve_policy(prior, props, costs, horizon=2)
        myopic_value, myopic_root = solve_policy(prior, props, costs, horizon=1)
        exact_cost, exact_trace, exact_pred = _realized_cost(true_next, prior, props, costs, horizon=2)
        myopic_cost, myopic_trace, myopic_pred = _realized_cost(true_next, prior, props, costs, horizon=1)
        reduction = float(myopic_cost - exact_cost)
        diff = float(exact_cost - myopic_cost)

        evaluated += frequency
        exact_total += frequency * exact_cost
        myopic_total += frequency * myopic_cost
        disagreement_total += frequency * int(exact_root != myopic_root)
        diff_counts[round(diff, 12)] += frequency
        if reduction > 1e-15:
            better += frequency
            positive_reduction_total += frequency * reduction
        elif reduction < -1e-15:
            worse += frequency
            negative_reduction_total += frequency * (-reduction)
        else:
            tied += frequency

        if include_rows:
            rows.append(
                {
                    "current_action": current,
                    "true_next_action": true_next,
                    "test_frequency": frequency,
                    "candidates": len(prior),
                    "train_support": train_support,
                    "exact_expected_value": exact_value,
                    "myopic_expected_value": myopic_value,
                    "exact_root_action": list(exact_root),
                    "myopic_root_action": list(myopic_root),
                    "root_actions_differ": exact_root != myopic_root,
                    "exact_realized_cost": exact_cost,
                    "myopic_realized_cost": myopic_cost,
                    "realized_cost_reduction": reduction,
                    "exact_trace": exact_trace,
                    "myopic_trace": myopic_trace,
                    "exact_prediction": exact_pred,
                    "myopic_prediction": myopic_pred,
                }
            )

    raw_n = len(test_transitions)
    comparison = {
        "exact_mean_realized_cost": _weighted_mean(exact_total, evaluated),
        "myopic_mean_realized_cost": _weighted_mean(myopic_total, evaluated),
        "mean_realized_cost_reduction": _weighted_mean(myopic_total - exact_total, evaluated),
        "root_action_disagreement_fraction": 0.0 if evaluated == 0 else disagreement_total / evaluated,
        "exact_better_tied_worse": [better, tied, worse],
        "mean_reduction_when_exact_better": _weighted_mean(positive_reduction_total, better),
        "mean_increase_when_exact_worse": _weighted_mean(negative_reduction_total, worse),
        "paired_exact_minus_myopic_bootstrap_ci95": _weighted_bootstrap_ci(
            diff_counts,
            draws=bootstrap_draws,
        ),
    }
    return {
        "metadata_isolation": {
            **metadata,
            "evaluated_transitions_using_test_only_action_metadata": 0,
            "test_metadata_role": "consistency_check_only_for_action_ids_seen_in_train",
        },
        "coverage": {
            "train_sequences": len(train_sequences),
            "test_sequences": len(test_sequences),
            "raw_test_transitions": raw_n,
            "unique_test_transition_pairs": len(pair_counts),
            "evaluated_transitions": evaluated,
            "evaluated_unique_transition_pairs": len(rows) if include_rows else None,
            "coverage_fraction": 0.0 if raw_n == 0 else evaluated / raw_n,
            "min_train_transition_support": min_train_count,
            "skip_counts": dict(sorted(skip.items())),
        },
        "comparison": comparison,
        "rows": rows,
    }


def sensitivity(train_rows, test_rows) -> dict:
    support = {}
    for threshold in SUPPORT_SENSITIVITY:
        r = evaluate(
            train_rows,
            test_rows,
            min_train_count=threshold,
            costs=BASE_COSTS,
            include_rows=False,
            bootstrap_draws=0,
        )
        support[str(threshold)] = {
            "coverage": r["coverage"],
            "comparison": r["comparison"],
        }

    cost = {}
    for multiplier in COST_MULTIPLIERS:
        costs = {k: v * multiplier for k, v in BASE_COSTS.items()}
        r = evaluate(
            train_rows,
            test_rows,
            min_train_count=5,
            costs=costs,
            include_rows=False,
            bootstrap_draws=0,
        )
        cost[str(multiplier)] = {
            "normalized_query_costs": costs,
            "coverage": r["coverage"],
            "comparison": r["comparison"],
        }
    return {
        "support_thresholds": support,
        "joint_query_cost_multipliers": cost,
    }


def _file_receipt(path: Path) -> dict:
    data = path.read_bytes()
    return {
        "name": path.name,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--annotations-dir",
        required=True,
        help="Directory containing Assembly101 fine-grained train.csv and test.csv",
    )
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-train-count", type=int, default=5)
    ap.add_argument("--skip-sensitivity", action="store_true")
    args = ap.parse_args()

    root = Path(args.annotations_dir)
    train_path = root / "train.csv"
    test_path = root / "test.csv"
    train = _read_segments(train_path)
    test = _read_segments(test_path)
    result = evaluate(train, test, min_train_count=args.min_train_count)

    report = {
        "schema": "dovod-paper-b-assembly101-acquisition-v2-leakage-safe",
        "external_dataset": {
            "name": "Assembly101 fine-grained action annotations",
            "annotation_schema_source": "assembly-101/assembly101-annotations",
            "split_policy": "official train/test",
            "files": [_file_receipt(train_path), _file_receipt(test_path)],
        },
        "adapter": {
            "target": "next fine-grained action conditioned on current action",
            "training_prior": "empirical next-action distribution from official train annotations only",
            "query_metadata": "verb/noun mapping from official train annotations only",
            "test": "official test annotations used only for held-out transition evaluation and shared-ID metadata consistency checks",
            "query_types": {
                "verb": "reveals next-action verb id",
                "noun": "reveals next-action noun/object id",
            },
            "normalized_query_costs": BASE_COSTS,
            "exact_horizon": 2,
            "myopic_horizon": 1,
        },
        **result,
        "sensitivity": None if args.skip_sensitivity else sensitivity(train, test),
        "claim_boundary": (
            "Third procedural-distribution workload test using Assembly101 fine-grained action annotations. "
            "All priors, candidate sets, and verb/noun query metadata are constructed from the official train split only; "
            "test-only action metadata cannot enter policy construction. Verb/noun attributes are typed information channels, "
            "not physical-vs-semantic source labels, and costs are normalized decision-loss units rather than measured effort. "
            "The benchmark can support cross-dataset non-myopic workload claims but cannot by itself validate persistent source reliability/orientation."
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    public = {k: v for k, v in report.items() if k != "rows"}
    print(json.dumps(public, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
