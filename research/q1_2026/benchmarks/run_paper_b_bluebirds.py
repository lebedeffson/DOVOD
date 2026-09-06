from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
PINNED_COMMIT = "fe5ba700f1adbb489c69af311558d64370d73d36"
BASE_URL = f"https://raw.githubusercontent.com/welinder/cubam/{PINNED_COMMIT}/demo/bluebirds"
GT_URL = f"{BASE_URL}/gt.yaml"
LABELS_URL = f"{BASE_URL}/labels.yaml"
OUT = ROOT / "results" / "paper_b_bluebirds_external.json"


def stable_rank(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def fetch_text(url: str, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "DOVOD-Q1-repro/2026-09-05"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8")


def parse_ground_truth(text: str) -> dict[int, int]:
    pairs = re.findall(r"(\d+)\s*:\s*(true|false)", text, flags=re.IGNORECASE)
    if not pairs:
        raise ValueError("no ground-truth entries parsed")
    return {int(task): int(value.lower() == "true") for task, value in pairs}


def parse_worker_labels(text: str) -> dict[int, dict[int, int]]:
    starts = list(re.finditer(r"(?m)^(\d+)\s*:\s*\{", text))
    if not starts:
        raise ValueError("no workers parsed")
    workers: dict[int, dict[int, int]] = {}
    for i, match in enumerate(starts):
        worker = int(match.group(1))
        end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        block = text[match.end():end]
        pairs = re.findall(r"(\d+)\s*:\s*(true|false)", block, flags=re.IGNORECASE)
        workers[worker] = {int(task): int(value.lower() == "true") for task, value in pairs}
    return workers


def task_split(task: int) -> str:
    return "calibration" if stable_rank(f"bluebirds-task|{task}") % 1000 < 300 else "test"


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return float("nan")
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def bootstrap_accuracy(y_true: list[int], y_pred: list[int], *, seed: int, n_boot: int = 5000) -> dict:
    truth = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    if len(truth) == 0:
        return {"mean": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, len(truth), size=len(truth))
        vals[i] = np.mean(truth[idx] == pred[idx])
    return {
        "mean": float(np.mean(truth == pred)),
        "ci95": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
        "n": int(len(truth)),
        "bootstrap_replicates": int(n_boot),
    }


def aggregate_task(task: int, workers: list[int], labels: dict[int, dict[int, int]], orientation: dict[int, int]) -> int | None:
    votes = []
    for worker in workers:
        if task not in labels[worker]:
            continue
        value = labels[worker][task]
        if orientation.get(worker, 1) < 0:
            value = 1 - value
        votes.append(value)
    if not votes:
        return None
    ones = sum(votes)
    zeros = len(votes) - ones
    if ones == zeros:
        return stable_rank(f"bluebirds-tie|{task}|{','.join(map(str, workers))}") % 2
    return int(ones > zeros)


def policy_accuracy(task_ids: list[int], selected_workers: list[int], labels: dict[int, dict[int, int]], truth: dict[int, int], orientation: dict[int, int]) -> tuple[float, list[int], list[int]]:
    yt, yp = [], []
    for task in task_ids:
        pred = aggregate_task(task, selected_workers, labels, orientation)
        if pred is None:
            continue
        yt.append(truth[task])
        yp.append(pred)
    return accuracy(yt, yp), yt, yp


def run(gt_text: str, labels_text: str) -> dict:
    truth = parse_ground_truth(gt_text)
    labels = parse_worker_labels(labels_text)
    task_ids = sorted(truth)
    calibration_tasks = [t for t in task_ids if task_split(t) == "calibration"]
    test_tasks = [t for t in task_ids if task_split(t) == "test"]
    if len(calibration_tasks) < 10 or len(test_tasks) < 20:
        raise AssertionError((len(calibration_tasks), len(test_tasks)))

    worker_rows = []
    orientation: dict[int, int] = {}
    reliability: dict[int, float] = {}
    raw_cal: dict[int, float] = {}
    raw_test: dict[int, float] = {}
    corrected_test: dict[int, float] = {}

    for worker in sorted(labels):
        cal_pairs = [(truth[t], labels[worker][t]) for t in calibration_tasks if t in labels[worker]]
        test_pairs = [(truth[t], labels[worker][t]) for t in test_tasks if t in labels[worker]]
        if not cal_pairs or not test_pairs:
            continue
        cal_correct = sum(y == z for y, z in cal_pairs)
        cal_n = len(cal_pairs)
        raw_cal_acc = cal_correct / cal_n
        orient = 1 if raw_cal_acc >= 0.5 else -1
        corrected_cal_correct = max(cal_correct, cal_n - cal_correct)
        rel = (1.0 + corrected_cal_correct) / (2.0 + cal_n)

        test_truth = [y for y, _ in test_pairs]
        test_raw_pred = [z for _, z in test_pairs]
        test_corr_pred = [z if orient > 0 else 1 - z for _, z in test_pairs]
        raw_test_acc = accuracy(test_truth, test_raw_pred)
        corr_test_acc = accuracy(test_truth, test_corr_pred)

        orientation[worker] = orient
        reliability[worker] = rel
        raw_cal[worker] = raw_cal_acc
        raw_test[worker] = raw_test_acc
        corrected_test[worker] = corr_test_acc
        worker_rows.append({
            "worker": worker,
            "calibration_n": cal_n,
            "test_n": len(test_pairs),
            "calibration_accuracy": raw_cal_acc,
            "calibrated_orientation": orient,
            "posterior_oriented_reliability": rel,
            "raw_test_accuracy": raw_test_acc,
            "orientation_corrected_test_accuracy": corr_test_acc,
        })

    eligible = sorted(reliability)
    if len(eligible) < 10:
        raise AssertionError(f"too few eligible workers: {len(eligible)}")
    ranking = sorted(eligible, key=lambda w: (-reliability[w], w))

    raw_cal_values = [raw_cal[w] for w in eligible]
    raw_test_values = [raw_test[w] for w in eligible]
    oriented_cal_values = [reliability[w] for w in eligible]
    oriented_test_values = [corrected_test[w] for w in eligible]
    pear_raw = pearsonr(raw_cal_values, raw_test_values)
    spear_raw = spearmanr(raw_cal_values, raw_test_values)
    pear_oriented = pearsonr(oriented_cal_values, oriented_test_values)
    spear_oriented = spearmanr(oriented_cal_values, oriented_test_values)

    raw_vote_correct = 0
    corrected_vote_correct = 0
    vote_total = 0
    for worker in eligible:
        for task in test_tasks:
            if task not in labels[worker]:
                continue
            y = truth[task]
            z = labels[worker][task]
            raw_vote_correct += int(z == y)
            zc = z if orientation[worker] > 0 else 1 - z
            corrected_vote_correct += int(zc == y)
            vote_total += 1

    policies = {}
    hash_baselines = {}
    for k in (1, 3, 5, 7):
        selected = ranking[: min(k, len(ranking))]
        acc, yt, yp = policy_accuracy(test_tasks, selected, labels, truth, orientation)
        boot = bootstrap_accuracy(yt, yp, seed=20260905 + k)
        policies[str(k)] = {"selected_workers": selected, "accuracy": acc, "bootstrap": boot}

        baseline_acc = []
        for replicate in range(256):
            hash_rank = sorted(eligible, key=lambda w: stable_rank(f"bluebirds-worker-baseline|{replicate}|{w}"))
            a, _, _ = policy_accuracy(test_tasks, hash_rank[: min(k, len(hash_rank))], labels, truth, orientation)
            baseline_acc.append(a)
        selected_acc = policies[str(k)]["accuracy"]
        hash_baselines[str(k)] = {
            "replicates": len(baseline_acc),
            "mean_accuracy": float(np.mean(baseline_acc)),
            "ci95_over_hash_rankings": [float(np.quantile(baseline_acc, 0.025)), float(np.quantile(baseline_acc, 0.975))],
            "selected_minus_hash_mean": float(selected_acc - np.mean(baseline_acc)),
            "selected_percentile": float(np.mean(np.asarray(baseline_acc) <= selected_acc)),
        }

    all_raw_acc, raw_yt, raw_yp = policy_accuracy(test_tasks, eligible, labels, truth, {w: 1 for w in eligible})
    all_corrected_acc, corr_yt, corr_yp = policy_accuracy(test_tasks, eligible, labels, truth, orientation)

    return {
        "schema": "dovod-q1-bluebirds-external-v1",
        "dataset": {
            "name": "Blue Birds",
            "source_repo": "welinder/cubam",
            "source_commit": PINNED_COMMIT,
            "ground_truth_blob_sha": "6791189f35bf9823b4fffaf4b656817ecb1a7048",
            "labels_blob_sha": "b4db48e832797ff151ded434d970f7bd9d3130c2",
            "tasks": len(task_ids),
            "workers_parsed": len(labels),
            "votes_parsed": int(sum(len(v) for v in labels.values())),
        },
        "split": {
            "rule": "sha256(task-id), calibration bucket <300/1000; labels never used for split",
            "calibration_tasks": len(calibration_tasks),
            "test_tasks": len(test_tasks),
            "calibration_task_ids": calibration_tasks,
            "test_task_ids": test_tasks,
        },
        "worker_validation": {
            "eligible_workers": len(eligible),
            "orientation_flipped_workers": sum(orientation[w] < 0 for w in eligible),
            "raw_calibration_vs_test_pearson": {"r": float(pear_raw.statistic), "p": float(pear_raw.pvalue)},
            "raw_calibration_vs_test_spearman": {"rho": float(spear_raw.statistic), "p": float(spear_raw.pvalue)},
            "oriented_posterior_reliability_vs_corrected_test_pearson": {"r": float(pear_oriented.statistic), "p": float(pear_oriented.pvalue)},
            "oriented_posterior_reliability_vs_corrected_test_spearman": {"rho": float(spear_oriented.statistic), "p": float(spear_oriented.pvalue)},
            "raw_vote_accuracy": raw_vote_correct / vote_total,
            "orientation_corrected_vote_accuracy": corrected_vote_correct / vote_total,
            "orientation_correction_gain": (corrected_vote_correct - raw_vote_correct) / vote_total,
            "worker_rows": worker_rows,
        },
        "source_selection": {
            "ranking_rule": "descending Beta(1,1) posterior mean of orientation-corrected calibration accuracy; worker id tie-break",
            "calibration_selected_top_k": policies,
            "label_free_hash_ranking_reference": hash_baselines,
            "all_workers_raw_majority": {"accuracy": all_raw_acc, "bootstrap": bootstrap_accuracy(raw_yt, raw_yp, seed=20262001)},
            "all_workers_orientation_corrected_majority": {"accuracy": all_corrected_acc, "bootstrap": bootstrap_accuracy(corr_yt, corr_yp, seed=20262002)},
        },
        "claim_boundary": (
            "This is an external crowdsourcing validation of persistent source quality/orientation and calibration-based source selection, "
            "not a procedural-action benchmark and not a direct evaluation of the finite-world Bellman planner. The split and ranking rules "
            "are deterministic and frozen in code; ground truth from test tasks is used only for evaluation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", help="optional local gt.yaml")
    parser.add_argument("--labels", help="optional local labels.yaml")
    parser.add_argument("--output", default=str(OUT))
    args = parser.parse_args()
    if bool(args.gt) != bool(args.labels):
        raise SystemExit("--gt and --labels must be provided together")
    if args.gt:
        gt_text = Path(args.gt).read_text(encoding="utf-8")
        labels_text = Path(args.labels).read_text(encoding="utf-8")
    else:
        gt_text = fetch_text(GT_URL)
        labels_text = fetch_text(LABELS_URL)
    report = run(gt_text, labels_text)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    public_summary = {
        "schema": report["schema"],
        "dataset": report["dataset"],
        "split": {k: v for k, v in report["split"].items() if not k.endswith("_task_ids")},
        "worker_validation": {k: v for k, v in report["worker_validation"].items() if k != "worker_rows"},
        "source_selection": report["source_selection"],
        "claim_boundary": report["claim_boundary"],
    }
    print(json.dumps(public_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
