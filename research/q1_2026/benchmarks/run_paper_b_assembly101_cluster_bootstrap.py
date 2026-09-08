from __future__ import annotations

"""Video-cluster robustness analysis for the Assembly101 acquisition benchmark.

The primary Assembly101 adapter evaluates many adjacent transitions from the same
held-out videos. Treating those transitions as iid in a bootstrap would overstate
effective sample size. This companion analysis keeps policy construction exactly
train-only, caches each eligible transition-pair policy once, then resamples whole
test videos with replacement. The primary cluster estimand is the transition-
weighted mean exact-minus-myopic realized cost.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from benchmarks import run_paper_b_assembly101_acquisition as base
except ModuleNotFoundError:  # direct script execution from benchmarks/
    import run_paper_b_assembly101_acquisition as base

DEFAULT_DRAWS = 10_000
DEFAULT_SEED = 20260908


def _video_sequences(rows: list[dict]) -> list[tuple[str, list[int]]]:
    by_video: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_video[str(row["video"])].append(row)
    out = []
    for video in sorted(by_video):
        ordered = sorted(
            by_video[video],
            key=lambda r: (r["start_frame"], r["end_frame"], r["action_id"]),
        )
        seq = [int(r["action_id"]) for r in ordered]
        if len(seq) >= 2:
            out.append((video, seq))
    return out


def _eligible_pair_metrics(train_rows, test_rows, *, min_train_count: int, costs: dict[str, float]):
    props = base._properties(train_rows)
    metadata = base._validate_test_metadata(props, test_rows)
    train_counts = base._transition_counts(base._sequences(train_rows))
    video_sequences = _video_sequences(test_rows)
    pair_frequency = Counter(
        (a, b)
        for _, seq in video_sequences
        for a, b in zip(seq, seq[1:])
    )

    skip = Counter()
    metrics = {}
    for pair, frequency in sorted(pair_frequency.items()):
        current, true_next = pair
        counter = train_counts.get(current)
        if not counter:
            skip["no_train_prior"] += frequency
            continue
        support = int(sum(counter.values()))
        if support < min_train_count:
            skip["below_train_support_threshold"] += frequency
            continue
        if len(counter) < 2:
            skip["single_train_candidate"] += frequency
            continue
        if true_next not in counter:
            skip["true_next_unseen_after_current_in_train"] += frequency
            continue
        if current not in props or true_next not in props or any(a not in props for a in counter):
            raise AssertionError("eligible pair metadata must be train-derived")

        prior = base._normalize(counter)
        exact_value, exact_root = base.solve_policy(prior, props, costs, horizon=2)
        myopic_value, myopic_root = base.solve_policy(prior, props, costs, horizon=1)
        exact_cost, _, _ = base._realized_cost(true_next, prior, props, costs, horizon=2)
        myopic_cost, _, _ = base._realized_cost(true_next, prior, props, costs, horizon=1)
        metrics[pair] = {
            "exact_expected_value": float(exact_value),
            "myopic_expected_value": float(myopic_value),
            "exact_root": exact_root,
            "myopic_root": myopic_root,
            "exact_cost": float(exact_cost),
            "myopic_cost": float(myopic_cost),
            "exact_minus_myopic": float(exact_cost - myopic_cost),
        }

    return metadata, video_sequences, pair_frequency, metrics, dict(sorted(skip.items()))


def _bootstrap_cluster_ratios(
    diff_totals: np.ndarray,
    transition_counts: np.ndarray,
    per_video_means: np.ndarray,
    *,
    seed: int,
    draws: int,
) -> dict:
    if len(diff_totals) == 0 or draws <= 0:
        return {
            "draws": int(draws),
            "transition_weighted_ci95": None,
            "equal_video_weighted_ci95": None,
        }
    rng = np.random.default_rng(seed)
    n_videos = len(diff_totals)
    transition_weighted = np.empty(draws, dtype=float)
    equal_video_weighted = np.empty(draws, dtype=float)
    for i in range(draws):
        idx = rng.integers(0, n_videos, size=n_videos)
        denom = float(transition_counts[idx].sum())
        transition_weighted[i] = float(diff_totals[idx].sum() / denom)
        equal_video_weighted[i] = float(per_video_means[idx].mean())
    return {
        "draws": int(draws),
        "seed": int(seed),
        "transition_weighted_ci95": [
            float(np.quantile(transition_weighted, 0.025)),
            float(np.quantile(transition_weighted, 0.975)),
        ],
        "equal_video_weighted_ci95": [
            float(np.quantile(equal_video_weighted, 0.025)),
            float(np.quantile(equal_video_weighted, 0.975)),
        ],
    }


def analyze(
    train_rows,
    test_rows,
    *,
    min_train_count: int = 5,
    costs=None,
    draws: int = DEFAULT_DRAWS,
    seed: int = DEFAULT_SEED,
) -> dict:
    costs = dict(base.BASE_COSTS if costs is None else costs)
    metadata, video_sequences, pair_frequency, metrics, skip = _eligible_pair_metrics(
        train_rows,
        test_rows,
        min_train_count=min_train_count,
        costs=costs,
    )

    per_video = []
    raw_transitions = 0
    evaluated_transitions = 0
    for video, seq in video_sequences:
        exact_total = 0.0
        myopic_total = 0.0
        root_disagreements = 0
        n = 0
        raw_n = max(0, len(seq) - 1)
        raw_transitions += raw_n
        for pair in zip(seq, seq[1:]):
            metric = metrics.get(pair)
            if metric is None:
                continue
            n += 1
            exact_total += metric["exact_cost"]
            myopic_total += metric["myopic_cost"]
            root_disagreements += int(metric["exact_root"] != metric["myopic_root"])
        if n:
            diff_total = exact_total - myopic_total
            per_video.append(
                {
                    "video": video,
                    "evaluated_transitions": n,
                    "exact_cost_total": exact_total,
                    "myopic_cost_total": myopic_total,
                    "exact_minus_myopic_total": diff_total,
                    "mean_exact_minus_myopic": diff_total / n,
                    "root_action_disagreements": root_disagreements,
                }
            )
            evaluated_transitions += n

    diffs = np.asarray([r["exact_minus_myopic_total"] for r in per_video], dtype=float)
    ns = np.asarray([r["evaluated_transitions"] for r in per_video], dtype=float)
    means = np.asarray([r["mean_exact_minus_myopic"] for r in per_video], dtype=float)
    cluster_bootstrap = _bootstrap_cluster_ratios(
        diffs,
        ns,
        means,
        seed=seed,
        draws=draws,
    )

    better = int(np.sum(means < -1e-15)) if len(means) else 0
    tied = int(np.sum(np.abs(means) <= 1e-15)) if len(means) else 0
    worse = int(np.sum(means > 1e-15)) if len(means) else 0
    transition_weighted = None if evaluated_transitions == 0 else float(diffs.sum() / ns.sum())
    equal_video_weighted = None if not per_video else float(means.mean())

    return {
        "schema": "dovod-paper-b-assembly101-video-cluster-v1",
        "policy_isolation": {
            **metadata,
            "prior_source": "official_train_only",
            "candidate_set_source": "official_train_only",
            "query_metadata_source": "official_train_only",
            "test_only_metadata_used_for_policy": False,
        },
        "coverage": {
            "test_videos": len(video_sequences),
            "evaluated_test_videos": len(per_video),
            "raw_test_transitions": raw_transitions,
            "evaluated_transitions": evaluated_transitions,
            "coverage_fraction": 0.0 if raw_transitions == 0 else evaluated_transitions / raw_transitions,
            "unique_test_transition_pairs": len(pair_frequency),
            "eligible_unique_transition_pairs": len(metrics),
            "min_train_transition_support": int(min_train_count),
            "skip_counts": skip,
        },
        "cluster_analysis": {
            "resampling_unit": "official_test_video",
            "primary_estimand": "transition-weighted mean exact-minus-myopic realized cost",
            "transition_weighted_exact_minus_myopic": transition_weighted,
            "equal_video_weighted_exact_minus_myopic": equal_video_weighted,
            "video_exact_better_tied_worse": [better, tied, worse],
            "bootstrap": cluster_bootstrap,
        },
        "per_video": per_video,
        "claim_boundary": (
            "Whole held-out Assembly101 videos are the resampling clusters, so adjacent transitions from one video are not treated as iid bootstrap units. "
            "The primary point estimate remains transition-weighted to match the main workload metric. This analysis strengthens uncertainty quantification only; "
            "it does not turn verb/noun channels into physical-vs-semantic source labels or validate persistent source orientation."
        ),
    }


def _receipt(path: Path) -> dict:
    data = path.read_bytes()
    return {"name": path.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-train-count", type=int, default=5)
    ap.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    args = ap.parse_args()

    root = Path(args.annotations_dir)
    train_path = root / "train.csv"
    test_path = root / "test.csv"
    report = analyze(
        base._read_segments(train_path),
        base._read_segments(test_path),
        min_train_count=args.min_train_count,
        draws=args.draws,
    )
    report["dataset_receipts"] = [_receipt(train_path), _receipt(test_path)]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    public = {k: v for k, v in report.items() if k != "per_video"}
    print(json.dumps(public, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
