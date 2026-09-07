from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parents[1] / "data" / "processed" / "carriers"
EVIDENCE = ROOT / "external_evidence" / "dovod_short_papers_v12_full.json"
OUT = ROOT / "results" / "paper_a_real_audit.json"


def _dataset_summary(payload: dict) -> dict:
    rows = payload["relations"]
    n = 1 + max(max(int(r["p"]), int(r["a"])) for r in rows)
    universe = n * (n - 1)
    carrier_field = "recordings" if "independent_recordings" in payload else "participants"
    independent_units = int(payload.get("independent_recordings", payload.get("independent_participants")))
    counts = {(int(r["p"]), int(r["a"])): int(r[carrier_field]) for r in rows}

    def threshold_summary(min_carriers: int) -> dict:
        selected = {k for k, c in counts.items() if c >= min_carriers}
        pairs = Counter()
        for i in range(n):
            for j in range(i + 1, n):
                ij = (i, j) in selected
                ji = (j, i) in selected
                if ij and ji:
                    pairs["both_directions_refuted"] += 1
                elif ij or ji:
                    pairs["one_direction_refuted"] += 1
                else:
                    pairs["neither_direction_refuted"] += 1
        return {
            "min_independent_carriers": min_carriers,
            "directed_relations_refuted": len(selected),
            "directed_relations_unresolved": universe - len(selected),
            "audit_queue_reduction_fraction": len(selected) / universe,
            "unordered_pair_categories": dict(pairs),
        }

    dist = Counter(counts.values())
    observed = len(counts)
    return {
        "dataset": payload["dataset"],
        "components": n,
        "directed_candidate_relations": universe,
        "independent_units": independent_units,
        "carrier_field": carrier_field,
        "observed_refuted_relations": observed,
        "unresolved_after_one_counterexample_rule": universe - observed,
        "audit_queue_reduction_fraction": observed / universe,
        "single_carrier_refutations": sum(v for c, v in dist.items() if c == 1),
        "multi_carrier_refutations": sum(v for c, v in dist.items() if c >= 2),
        "carrier_count_distribution": {str(k): int(v) for k, v in sorted(dist.items())},
        "thresholds": [threshold_summary(k) for k in (1, 2, 3)],
    }


def main() -> None:
    meccano = json.loads((DATA / "meccano_carrier_counts.json").read_text(encoding="utf-8"))
    impact = json.loads((DATA / "impact_carrier_counts.json").read_text(encoding="utf-8"))
    frozen = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    report = {
        "schema": "dovod-paper-a-real-audit-v1",
        "datasets": [_dataset_summary(meccano), _dataset_summary(impact)],
        "existing_nested_calibration": {
            "meccano_recall_before": frozen["A"]["nested_before"],
            "meccano_recall_after": frozen["A"]["nested_after"],
            "gain": frozen["A"]["nested_gain"],
            "pairs": frozen["A"]["nested_pairs"],
            "gain_ci95": frozen["A"]["nested_ci95"],
            "mean_candidate_restrictions_pruned": frozen["A"]["mean_pruned"],
        },
        "simple_threshold_warning": {
            "one_carrier_rule_recall": frozen["A"]["threshold1_recall"],
            "two_carrier_rule_recall": frozen["A"]["threshold2_recall"],
            "recall_change": frozen["A"]["threshold2_recall"] - frozen["A"]["threshold1_recall"],
            "interpretation": (
                "Requiring two independent carriers is a simple conservative heuristic, but in the frozen MECCANO LORO stress it reduces next-action recall. "
                "The carrier threshold is therefore an audit-policy choice, not a universally safer replacement for decision-level calibration."
            ),
        },
        "claim_boundary": (
            "A recorded successful counterexample falsifies only the universal empirical unary same-state prerequisite claim. "
            "Relations without a counterexample remain unresolved rather than validated as mechanically necessary. "
            "MECCANO and IMPACT component identities are not assumed to match, so no relation-level cross-dataset intersection is computed. "
            "Audit-queue reduction counts candidate relations removed from further necessity review; they do not measure human time saved."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
