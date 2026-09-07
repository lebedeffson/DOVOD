from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

TARGET_DOMAINS = ("barman", "floortile", "parking", "tpp")
ALGORITHMS = ("SAM", "OffLAM", "NOLAM", "ROSAME")
BUDGETS = (3, 10)


def _load_cases(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("schema") == "dovod-paper-a-amlgym-practical-case-v2":
            data["_path"] = str(path)
            rows.append(data)
    return rows


def _key(row: dict) -> tuple[str, str, int]:
    return row["domain"], row["algorithm"], int(row["trace_budget"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact_root")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    raw = _load_cases(Path(args.artifact_root))
    by_key = defaultdict(list)
    for row in raw:
        by_key[_key(row)].append(row)
    expected = {(d, a, b) for d in TARGET_DOMAINS for a in ALGORITHMS for b in BUDGETS}
    duplicates = {k: v for k, v in by_key.items() if len(v) > 1}
    unique = {k: v[0] for k, v in by_key.items() if len(v) == 1}
    missing = sorted(expected - set(unique))

    comparisons = []
    failures = []
    for key in sorted(expected.intersection(unique)):
        row = unique[key]
        if row.get("status", "ok") != "ok":
            failures.append({"key": list(key), "error": row.get("error"), "status": row.get("status")})
            continue
        metrics = row["metrics"]["test"]
        n = int(metrics["base"]["n"])
        if n == 0:
            status = "empty_test"
        else:
            d_risk = float(metrics["base"]["risk"]) - float(metrics["dovod"]["risk"])
            status = "improved" if d_risk > 1e-15 else "worsened" if d_risk < -1e-15 else "tied"
        comparisons.append({
            "key": list(key),
            "test_n": n,
            "status": status,
            "base": metrics["base"],
            "dovod": metrics["dovod"],
            "threshold_raw": metrics["threshold_raw"],
            "threshold_gated": metrics["threshold_gated"],
            "audit": row["audit"],
            "has_corrected_false_block_exemplar": row.get("representative_corrected_false_block") is not None,
        })

    case_studies = {}
    for domain in TARGET_DOMAINS:
        candidates = [
            r for r in raw
            if r.get("status", "ok") == "ok"
            and r["domain"] == domain
            and int(r["metrics"]["test"]["base"]["n"]) > 0
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda r: (
            -(float(r["metrics"]["test"]["base"]["risk"]) - float(r["metrics"]["test"]["dovod"]["risk"])),
            r["algorithm"],
            int(r["trace_budget"]),
        ))
        best = candidates[0]
        case_studies[domain] = {
            "selection_rule": "post-confirmatory descriptive: largest DOVOD test-risk reduction within the prespecified domain",
            "algorithm": best["algorithm"],
            "trace_budget": best["trace_budget"],
            "test": best["metrics"]["test"],
            "audit": best["audit"],
            "representative_corrected_false_block": best.get("representative_corrected_false_block"),
        }

    usable = [r for r in comparisons if r["test_n"] > 0]

    def delta(row, name):
        return float(row["base"]["risk"]) - float(row[name]["risk"])

    report = {
        "schema": "dovod-paper-a-amlgym-practical-merge-v2",
        "expected_cases": len(expected),
        "observed_unique_cases": len(unique),
        "missing_cases": [list(k) for k in missing],
        "duplicate_keys": {"|".join(map(str, k)): [x["_path"] for x in v] for k, v in duplicates.items()},
        "usable_cases": len(usable),
        "failed_cases": failures,
        "dovod_vs_upstream": {
            "wins": sum(delta(r, "dovod") > 1e-15 for r in usable),
            "ties": sum(abs(delta(r, "dovod")) <= 1e-15 for r in usable),
            "losses": sum(delta(r, "dovod") < -1e-15 for r in usable),
            "mean_risk_reduction": None if not usable else sum(delta(r, "dovod") for r in usable) / len(usable),
        },
        "raw_threshold_vs_upstream": {
            "wins": sum(delta(r, "threshold_raw") > 1e-15 for r in usable),
            "ties": sum(abs(delta(r, "threshold_raw")) <= 1e-15 for r in usable),
            "losses": sum(delta(r, "threshold_raw") < -1e-15 for r in usable),
            "mean_risk_reduction": None if not usable else sum(delta(r, "threshold_raw") for r in usable) / len(usable),
            "total_false_allows": sum(int(r["threshold_raw"]["false_allows"]) for r in usable),
        },
        "gated_threshold_vs_upstream": {
            "wins": sum(delta(r, "threshold_gated") > 1e-15 for r in usable),
            "ties": sum(abs(delta(r, "threshold_gated")) <= 1e-15 for r in usable),
            "losses": sum(delta(r, "threshold_gated") < -1e-15 for r in usable),
            "mean_risk_reduction": None if not usable else sum(delta(r, "threshold_gated") for r in usable) / len(usable),
            "total_false_allows": sum(int(r["threshold_gated"]["false_allows"]) for r in usable),
        },
        "audit_summary": {
            "mean_operator_surfaces": None if not comparisons else sum(r["audit"]["operator_decision_surfaces"] for r in comparisons) / len(comparisons),
            "mean_flagged_operators": None if not comparisons else sum(r["audit"]["operators_flagged_by_observed_applicability_counterexample"] for r in comparisons) / len(comparisons),
            "mean_nonempty_deployed_repairs": None if not comparisons else sum(r["audit"]["operators_with_deployed_contextual_repair"] for r in comparisons) / len(comparisons),
            "mean_flagged_unresolved": None if not comparisons else sum(r["audit"]["flagged_operators_left_unresolved_after_calibration_gate"] for r in comparisons) / len(comparisons),
        },
        "case_studies": case_studies,
        "rows": comparisons,
        "claim_boundary": (
            "The four domains were chosen because the frozen confirmatory aggregate had positive domain-mean DOVOD improvement. "
            "This 32-cell analysis is explanatory and post-confirmatory, not a new significance test. "
            "Case-study cells are selected descriptively after the fact and must not be interpreted as unbiased estimates of general performance. "
            "Only calibration-approved candidates with at least one selected edit count as deployed repairs; empty no-op gates never count. "
            "The 0.8 threshold comparator is a coarse non-contextual policy, not a model of human engineering practice."
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2, sort_keys=True))
    if missing or duplicates:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
