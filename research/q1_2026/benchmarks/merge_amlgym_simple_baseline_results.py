from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

DOMAINS = (
    "barman", "blocksworld", "childsnack", "depots", "elevators", "ferry",
    "floortile", "goldminer", "grippers", "matchingbw", "miconic", "nomystery",
    "npuzzle", "parking", "rovers", "satellite", "sokoban", "spanner", "tpp", "transport",
)
ALGORITHMS = ("SAM", "OffLAM", "NOLAM", "ROSAME")
BUDGETS = (3, 10)
METHODS = ("dovod", "random_single_edit_gated", "frequency_single_edit_gated")


def _key(row):
    return row["domain"], row["algorithm"], int(row["trace_budget"])


def _sign_test_two_sided(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def _summary_vs_upstream(rows, method):
    deltas = [float(r["base"]["risk"]) - float(r[method]["risk"]) for r in rows]
    return {
        "wins": sum(x > 1e-15 for x in deltas),
        "ties": sum(abs(x) <= 1e-15 for x in deltas),
        "losses": sum(x < -1e-15 for x in deltas),
        "mean_risk_reduction": None if not deltas else sum(deltas) / len(deltas),
    }


def _pairwise(rows, left, right):
    # Positive means left has lower risk than right.
    deltas = [float(r[right]["risk"]) - float(r[left]["risk"]) for r in rows]
    return {
        "left": left,
        "right": right,
        "left_wins": sum(x > 1e-15 for x in deltas),
        "ties": sum(abs(x) <= 1e-15 for x in deltas),
        "left_losses": sum(x < -1e-15 for x in deltas),
        "mean_right_minus_left_risk": None if not deltas else sum(deltas) / len(deltas),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact_root")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root = Path(args.artifact_root)
    cases = []
    for path in sorted(root.rglob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if row.get("schema") == "dovod-q1-amlgym-simple-baseline-case-v1":
            row["_path"] = str(path)
            cases.append(row)
    by_key = defaultdict(list)
    for row in cases:
        by_key[_key(row)].append(row)
    expected = {(d, a, b) for d in DOMAINS for a in ALGORITHMS for b in BUDGETS}
    duplicates = {k: v for k, v in by_key.items() if len(v) > 1}
    unique = {k: v[0] for k, v in by_key.items() if len(v) == 1}
    missing = sorted(expected - set(unique))
    unexpected = sorted(set(unique) - expected)

    failures = []
    empty = []
    usable = []
    protocol_violations = []
    for key in sorted(expected.intersection(unique)):
        row = unique[key]
        if row.get("status") != "ok":
            failures.append({"key": list(key), "status": row.get("status"), "error": row.get("error")})
            continue
        protocol = row.get("protocol") or {}
        if protocol.get("pilot_overlap_count") != 0 or protocol.get("stage") != "confirmatory":
            protocol_violations.append(list(key))
        test = row["metrics"]["test"]
        if int(test["base"]["n"]) == 0:
            empty.append(list(key))
            continue
        usable.append({"key": list(key), **{name: test[name] for name in ("base",) + METHODS}})

    domain_pairwise = {}
    for right in ("random_single_edit_gated", "frequency_single_edit_gated"):
        domain_deltas = []
        domain_rows = {}
        for domain in DOMAINS:
            rows = [r for r in usable if r["key"][0] == domain]
            if not rows:
                continue
            delta = sum(float(r[right]["risk"]) - float(r["dovod"]["risk"]) for r in rows) / len(rows)
            domain_rows[domain] = {"usable_cells": len(rows), "baseline_minus_dovod_mean_risk": delta}
            domain_deltas.append(delta)
        wins = sum(x > 1e-15 for x in domain_deltas)
        losses = sum(x < -1e-15 for x in domain_deltas)
        domain_pairwise[right] = {
            "domains": domain_rows,
            "dovod_domain_wins": wins,
            "domain_ties": sum(abs(x) <= 1e-15 for x in domain_deltas),
            "dovod_domain_losses": losses,
            "exact_two_sided_sign_test_p": _sign_test_two_sided(wins, losses),
        }

    report = {
        "schema": "dovod-q1-amlgym-simple-baseline-merge-v1",
        "expected_cases": len(expected),
        "observed_unique_cases": len(unique),
        "missing_cases": [list(k) for k in missing],
        "unexpected_cases": [list(k) for k in unexpected],
        "duplicate_keys": {"|".join(map(str, k)): [r["_path"] for r in v] for k, v in duplicates.items()},
        "failed_or_timed_out_cases": failures,
        "empty_test_cases": empty,
        "usable_cases": len(usable),
        "protocol_violations": protocol_violations,
        "vs_upstream": {method: _summary_vs_upstream(usable, method) for method in METHODS},
        "dovod_vs_random": _pairwise(usable, "dovod", "random_single_edit_gated"),
        "dovod_vs_frequency": _pairwise(usable, "dovod", "frequency_single_edit_gated"),
        "domain_pairwise": domain_pairwise,
        "rows": usable,
        "claim_boundary": (
            "Post-confirmatory comparator analysis. The frozen 160-cell AMLGym primary result is unchanged. Random and frequency heuristics are simple at-most-one-edit baselines using the same repair/calibration/test partition and the same conservative calibration gate. CI completeness is structural only and never depends on which scientific method wins."
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2, sort_keys=True))
    if missing or unexpected or duplicates or protocol_violations:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
