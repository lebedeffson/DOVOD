from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path


def _sign_test(values, atol=1e-15):
    wins = sum(v < -atol for v in values)
    losses = sum(v > atol for v in values)
    ties = len(values) - wins - losses
    n = wins + losses
    if n == 0:
        p = 1.0
    else:
        k = min(wins, losses)
        tail = sum(comb(n, i) for i in range(k + 1)) / (2**n)
        p = min(1.0, 2.0 * tail)
    return {"wins": wins, "ties": ties, "losses": losses, "two_sided_p": p}


def _mean(xs):
    return None if not xs else sum(xs) / len(xs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("--output", required=True)
    ap.add_argument("--expected", type=int, default=160)
    args = ap.parse_args()

    files = sorted(Path(args.input_dir).rglob("*.json"))
    rows = []
    seen = set()
    for path in files:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if row.get("schema") != "dovod-q1-amlgym-v5-baseline-case-v1":
            continue
        key = (row.get("domain"), row.get("algorithm"), int(row.get("trace_budget", -1)))
        if key in seen:
            raise RuntimeError(f"duplicate V5 baseline cell: {key}")
        seen.add(key)
        rows.append(row)

    usable = [
        r for r in rows
        if r.get("status") == "ok" and int(r.get("test", {}).get("base", {}).get("n", 0)) > 0
    ]
    failures = [
        {
            "domain": r.get("domain"),
            "algorithm": r.get("algorithm"),
            "trace_budget": r.get("trace_budget"),
            "status": r.get("status"),
            "error": r.get("error"),
        }
        for r in rows if r.get("status") != "ok"
    ]

    comparisons = {
        "dovod_minus_base": [],
        "dovod_minus_frequency": [],
        "dovod_minus_random_mean": [],
        "frequency_minus_base": [],
        "random_mean_minus_base": [],
    }
    by_domain = defaultdict(lambda: defaultdict(list))
    for r in usable:
        t = r["test"]
        base = float(t["base"]["risk"])
        dovod = float(t["dovod"]["risk"])
        freq = float(t["frequency_one_edit"]["risk"])
        random_mean = float(t["random_one_edit_mean_risk"])
        vals = {
            "dovod_minus_base": dovod - base,
            "dovod_minus_frequency": dovod - freq,
            "dovod_minus_random_mean": dovod - random_mean,
            "frequency_minus_base": freq - base,
            "random_mean_minus_base": random_mean - base,
        }
        for name, value in vals.items():
            comparisons[name].append(value)
            by_domain[r["domain"]][name].append(value)

    domain_means = {
        domain: {name: _mean(values) for name, values in sorted(metrics.items())}
        for domain, metrics in sorted(by_domain.items())
    }
    domain_tests = {}
    for name in comparisons:
        vals = [m[name] for m in domain_means.values() if m.get(name) is not None]
        domain_tests[name] = _sign_test(vals)

    report = {
        "schema": "dovod-q1-amlgym-v5-baselines-merged-v1",
        "expected_case_count": args.expected,
        "accounted_case_count": len(rows),
        "complete_accounting": len(rows) == args.expected,
        "usable_test_cells": len(usable),
        "failed_or_missing_test_cells": len(rows) - len(usable),
        "failures": failures,
        "cell_mean_differences": {name: _mean(values) for name, values in comparisons.items()},
        "cell_sign_tests": {name: _sign_test(values) for name, values in comparisons.items()},
        "domain_mean_differences": domain_means,
        "domain_sign_tests": domain_tests,
        "random_replicates_per_cell": sorted({int(r.get("random_replicates", 0)) for r in usable}),
        "claim_boundary": (
            "Post-freeze V5 comparator analysis using the frozen confirmatory semantic-state protocol. Negative differences mean the first named method has lower test risk. "
            "The frequency comparator is restricted to one action-local one-feature edit; random comparator risk is averaged over prespecified deterministic replicates, never selected by test outcome. "
            "This study does not alter the frozen V4 confirmatory result and CI success is based on accounting/execution, not favorable scientific direction."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["complete_accounting"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
