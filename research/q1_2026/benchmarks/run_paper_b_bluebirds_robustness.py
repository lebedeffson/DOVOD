from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_paper_b_bluebirds import (
    GT_URL,
    LABELS_URL,
    accuracy,
    bootstrap_accuracy,
    fetch_text,
    parse_ground_truth,
    parse_worker_labels,
    policy_accuracy,
    stable_rank,
    task_split,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper_b_bluebirds_robustness.json"


def orientation_half(task: int) -> int:
    return stable_rank(f"bluebirds-orientation-half|{task}") % 2


def split_accuracy(worker_labels, truth, tasks, half: int) -> tuple[int, float]:
    selected = [t for t in tasks if t in worker_labels and orientation_half(t) == half]
    if not selected:
        return 0, float("nan")
    yt = [truth[t] for t in selected]
    yp = [worker_labels[t] for t in selected]
    return len(selected), accuracy(yt, yp)


def paired_bootstrap_difference(y_true, pred_a, pred_b, *, seed: int, n_boot: int = 10000) -> dict:
    truth = np.asarray(y_true, dtype=int)
    a = np.asarray(pred_a, dtype=int)
    b = np.asarray(pred_b, dtype=int)
    if not (len(truth) == len(a) == len(b)):
        raise ValueError("paired arrays must have equal length")
    delta = (truth == a).astype(float) - (truth == b).astype(float)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, len(delta), size=len(delta))
        vals[i] = np.mean(delta[idx])
    return {
        "mean_difference": float(np.mean(delta)),
        "ci95": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
        "n": int(len(delta)),
        "bootstrap_replicates": int(n_boot),
    }


def run(gt_text: str, labels_text: str) -> dict:
    truth = parse_ground_truth(gt_text)
    labels = parse_worker_labels(labels_text)
    tasks = sorted(truth)
    calibration = [t for t in tasks if task_split(t) == "calibration"]
    test = [t for t in tasks if task_split(t) == "test"]

    raw_reliability = {}
    naive_orientation = {}
    stable_orientation = {}
    rows = []
    for worker in sorted(labels):
        cal_tasks = [t for t in calibration if t in labels[worker]]
        test_tasks = [t for t in test if t in labels[worker]]
        if not cal_tasks or not test_tasks:
            continue
        cal_acc = accuracy([truth[t] for t in cal_tasks], [labels[worker][t] for t in cal_tasks])
        n0, a0 = split_accuracy(labels[worker], truth, calibration, 0)
        n1, a1 = split_accuracy(labels[worker], truth, calibration, 1)
        naive = -1 if cal_acc < 0.5 else 1
        replicated = bool(n0 and n1 and a0 < 0.5 and a1 < 0.5)
        stable = -1 if replicated else 1
        test_acc = accuracy([truth[t] for t in test_tasks], [labels[worker][t] for t in test_tasks])
        naive_orientation[worker] = naive
        stable_orientation[worker] = stable
        correct = sum(truth[t] == labels[worker][t] for t in cal_tasks)
        raw_reliability[worker] = (1.0 + correct) / (2.0 + len(cal_tasks))
        rows.append({
            "worker": worker,
            "calibration_accuracy": cal_acc,
            "half0_n": n0,
            "half0_accuracy": a0,
            "half1_n": n1,
            "half1_accuracy": a1,
            "half_orientation_agrees": bool((a0 < 0.5) == (a1 < 0.5)),
            "naive_orientation": naive,
            "replicated_inversion": replicated,
            "stable_orientation": stable,
            "raw_test_accuracy": test_acc,
            "heldout_test_supports_inversion": bool(test_acc < 0.5),
        })

    eligible = sorted(raw_reliability)
    raw_orientation = {w: 1 for w in eligible}
    ranking = sorted(eligible, key=lambda w: (-raw_reliability[w], w))

    def vote_accuracy(orientation):
        correct = total = 0
        for w in eligible:
            for t in test:
                if t not in labels[w]:
                    continue
                z = labels[w][t]
                if orientation[w] < 0:
                    z = 1 - z
                correct += int(z == truth[t])
                total += 1
        return correct / total

    raw_vote = vote_accuracy(raw_orientation)
    naive_vote = vote_accuracy(naive_orientation)
    stable_vote = vote_accuracy(stable_orientation)

    raw_majority, all_yt, all_yp = policy_accuracy(test, eligible, labels, truth, raw_orientation)
    stable_majority, stable_yt, _ = policy_accuracy(test, eligible, labels, truth, stable_orientation)
    if all_yt != stable_yt:
        raise AssertionError("task mismatch")

    policies = {}
    hash_reference = {}
    for k in (1, 3, 5, 7):
        selected = ranking[:k]
        acc, yt, yp = policy_accuracy(test, selected, labels, truth, raw_orientation)
        policies[str(k)] = {
            "selected_workers": selected,
            "accuracy": acc,
            "bootstrap": bootstrap_accuracy(yt, yp, seed=20266000 + k),
        }
        refs = []
        for rep in range(256):
            hash_rank = sorted(eligible, key=lambda w: stable_rank(f"bluebirds-robustness-hash|{rep}|{w}"))
            a, _, _ = policy_accuracy(test, hash_rank[:k], labels, truth, raw_orientation)
            refs.append(a)
        hash_reference[str(k)] = {
            "replicates": 256,
            "mean_accuracy": float(np.mean(refs)),
            "ci95_over_hash_rankings": [float(np.quantile(refs, 0.025)), float(np.quantile(refs, 0.975))],
            "selected_minus_hash_mean": float(acc - np.mean(refs)),
            "selected_percentile": float(np.mean(np.asarray(refs) <= acc)),
        }

    _, top5_yt, top5_yp = policy_accuracy(test, ranking[:5], labels, truth, raw_orientation)
    if top5_yt != all_yt:
        raise AssertionError("task mismatch")

    naive_flipped = [r for r in rows if r["naive_orientation"] < 0]
    stable_flipped = [r for r in rows if r["stable_orientation"] < 0]
    return {
        "schema": "dovod-q1-bluebirds-robustness-v1",
        "analysis_status": "post-primary robustness amendment after the frozen v1 external run; held-out test labels are evaluation-only",
        "dataset": {"tasks": len(tasks), "workers": len(labels), "calibration_tasks": len(calibration), "test_tasks": len(test)},
        "orientation_persistence": {
            "half_split_rule": "sha256(bluebirds-orientation-half|task) mod 2",
            "half_orientation_agreement_rate": float(np.mean([r["half_orientation_agrees"] for r in rows])),
            "naive_flipped_workers": len(naive_flipped),
            "naive_flips_supported_on_heldout_test": sum(r["heldout_test_supports_inversion"] for r in naive_flipped),
            "replicated_inversion_workers": len(stable_flipped),
            "replicated_flips_supported_on_heldout_test": sum(r["heldout_test_supports_inversion"] for r in stable_flipped),
            "raw_vote_accuracy": raw_vote,
            "naive_orientation_vote_accuracy": naive_vote,
            "stable_orientation_vote_accuracy": stable_vote,
            "raw_majority_accuracy": raw_majority,
            "stable_orientation_majority_accuracy": stable_majority,
            "rows": rows,
        },
        "source_quality_selection": {
            "ranking_rule": "descending Beta(1,1) posterior mean of raw calibration accuracy; worker-id tie-break",
            "top_k": policies,
            "label_free_hash_reference": hash_reference,
            "top5_minus_all_workers_raw_majority_paired_bootstrap": paired_bootstrap_difference(all_yt, top5_yp, all_yp, seed=20266505),
        },
        "interpretation_boundary": "The robustness analysis diagnoses whether the persistent-orientation assumption is empirically stable. It does not retroactively replace or re-label the prespecified v1 analysis. Source-quality ranking and orientation correction are reported separately.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt")
    ap.add_argument("--labels")
    ap.add_argument("--output", default=str(OUT))
    args = ap.parse_args()
    if bool(args.gt) != bool(args.labels):
        raise SystemExit("--gt and --labels must be supplied together")
    gt_text = Path(args.gt).read_text() if args.gt else fetch_text(GT_URL)
    labels_text = Path(args.labels).read_text() if args.labels else fetch_text(LABELS_URL)
    report = run(gt_text, labels_text)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    public = dict(report)
    public["orientation_persistence"] = {k: v for k, v in report["orientation_persistence"].items() if k != "rows"}
    print(json.dumps(public, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
