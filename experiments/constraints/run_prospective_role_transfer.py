from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
MECCANO_DIR = ROOT / "research" / "reference_impl" / "results" / "prospective_relation_review"
IMPACT_CSV = ROOT / "research" / "reference_impl" / "results" / "impact_v11_external" / "prospective_relation_risk.csv"
OUT = ROOT / "results" / "constraints" / "paper_a_prospective_role_transfer"
OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "figures" / "constraints" / "paper_a_prospective_role_transfer.png"

ROLE_SCORE = {
    "optional_optimal": 2.0,
    "mandatory": 1.0,
    "nonoptimal_redundant": 0.0,
}
ROLE_ORDER = ["optional_optimal", "mandatory", "nonoptimal_redundant"]


def tie_aware_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    wins = sum(float(p > n) + 0.5 * float(p == n) for p in pos for n in neg)
    return float(wins / (len(pos) * len(neg)))


def threshold_average_precision(y: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    positives = int(y.sum())
    if positives == 0:
        return None
    tp = fp = 0
    prev_recall = 0.0
    ap = 0.0
    for threshold in sorted(set(score.tolist()), reverse=True):
        mask = score == threshold
        tp += int(y[mask].sum())
        fp += int(mask.sum() - y[mask].sum())
        recall = tp / positives
        precision = tp / (tp + fp)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return float(ap)


def main() -> None:
    me = pd.read_csv(MECCANO_DIR / "decision_role_refutation_rates.csv")
    me = me.rename(columns={"fit_decision_role": "role", "refutation_rate": "meccano_refutation_rate"})
    impact = pd.read_csv(IMPACT_CSV)
    impact_summary = (
        impact.groupby("role", as_index=False)["refuted"]
        .agg(relation_folds="size", heldout_refutations="sum", impact_refutation_rate="mean")
    )
    merged = pd.DataFrame({"role": ROLE_ORDER}).merge(
        me[["role", "relation_folds", "heldout_refutations", "meccano_refutation_rate"]], on="role", how="left"
    ).merge(impact_summary, on="role", how="left", suffixes=("_meccano", "_impact"))
    merged.to_csv(OUT / "role_refutation_rates_meccano_vs_impact.csv", index=False)

    fold = pd.read_csv(MECCANO_DIR / "recording_level_ranking_metrics.csv")
    positive = fold[fold["heldout_refutations"] > 0].copy()

    y_i = impact["refuted"].to_numpy(int)
    s_i = impact["role"].map(ROLE_SCORE).to_numpy(float)
    impact_auc = tie_aware_auc(y_i, s_i)
    impact_ap = threshold_average_precision(y_i, s_i)

    report = {
        "schema": "tinyapv-paper-a-prospective-role-transfer-v1",
        "meccano": {
            "relation_fold_cases": int(me["relation_folds"].sum()),
            "heldout_refutations": int(me["heldout_refutations"].sum()),
            "positive_outer_folds": int(len(positive)),
            "equal_fold_mean_auc_positive_folds": float(positive["role_score_auc"].mean()),
            "equal_fold_mean_ap_positive_folds": float(positive["role_score_average_precision"].mean()),
            "equal_fold_mean_prevalence_positive_folds": float(positive["refutation_prevalence"].mean()),
            "all_positive_folds_ap_above_prevalence": bool((positive["ap_minus_prevalence"] > 0).all()),
            "role_rates": {r.role: float(r.meccano_refutation_rate) for r in merged.itertuples(index=False)},
        },
        "impact": {
            "relation_fold_cases": int(len(impact)),
            "heldout_refutations": int(y_i.sum()),
            "participants": int(impact["held_participant"].nunique()),
            "aggregate_role_score_auc": float(impact_auc),
            "aggregate_role_score_ap": float(impact_ap),
            "aggregate_prevalence": float(y_i.mean()),
            "role_rates": {r.role: float(r.impact_refutation_rate) for r in merged.itertuples(index=False)},
        },
        "interpretation": (
            "Fit-side decision role is prospectively informative for review prioritization inside MECCANO, "
            "but the same fixed role ordering is not a transferable risk score on the frozen IMPACT prospective artifact."
        ),
        "claim_boundary": (
            "The MECCANO ranking result is exploratory same-procedure prospective evidence: only six outer recordings contain direct refutations and relation instances within a recording are dependent. "
            "The IMPACT result is a negative transfer test of the fixed MECCANO role score, not evidence that decision roles themselves are meaningless in IMPACT or that no procedure-specific score can work."
        ),
    }
    (OUT / "prospective_role_transfer_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    labels = ["optional-optimal", "mandatory", "redundant"]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    ax.bar(x - width / 2, merged["meccano_refutation_rate"].to_numpy() * 100, width, label="MECCANO")
    ax.bar(x + width / 2, merged["impact_refutation_rate"].to_numpy() * 100, width, label="IMPACT")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Held-out refutation rate (%)")
    ax.set_title("Fit-side decision role is informative but not universal")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    ax.text(
        0.99,
        0.98,
        f"MECCANO positive-fold mean AUC = {positive['role_score_auc'].mean():.3f}\n"
        f"IMPACT fixed-score AUC = {impact_auc:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(FIG, dpi=220)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
