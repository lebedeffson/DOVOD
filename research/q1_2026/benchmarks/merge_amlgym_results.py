from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_a.statistics import exact_sign_test


def _case_key(case):
    return (case.get("domain"), case.get("algorithm"), int(case.get("trace_budget", -1)))


def _comparison_summary(cases, baseline_name: str, *, all_ok: bool) -> dict:
    cell_improvements: list[float] = []
    by_domain: dict[str, list[float]] = {}
    paired_better = paired_equal = paired_worse = 0

    for case in cases:
        if case.get("status") != "ok":
            continue
        test = case.get("decision", {}).get("test", {})
        baseline = test.get(baseline_name, {}).get("risk")
        dovod = test.get("dovod", {}).get("risk")
        if baseline is None or dovod is None:
            continue
        improvement = float(baseline) - float(dovod)
        cell_improvements.append(improvement)
        by_domain.setdefault(case["domain"], []).append(improvement)
        if improvement > 1e-12:
            paired_better += 1
        elif improvement < -1e-12:
            paired_worse += 1
        else:
            paired_equal += 1

    domain_means = {d: sum(v) / len(v) for d, v in by_domain.items()}
    return {
        "baseline": baseline_name,
        "evaluated_cell_count": len(cell_improvements),
        "mean_test_risk_improvement_over_cells": (
            sum(cell_improvements) / len(cell_improvements) if cell_improvements else None
        ),
        "cell_wins": paired_better,
        "cell_ties": paired_equal,
        "cell_losses": paired_worse,
        "domain_mean_test_risk_improvement": domain_means,
        "domain_sign_test": (
            exact_sign_test(list(domain_means.values()))
            if all_ok and len(domain_means) == 20
            else None
        ),
    }


def summarize(cases, *, all_ok: bool):
    ok = [c for c in cases if c.get("status") == "ok"]
    return {
        "ok_count": len(ok),
        "failed_count": len(cases) - len(ok),
        "dovod_vs_base": _comparison_summary(ok, "base", all_ok=all_ok),
        "dovod_vs_global_override": _comparison_summary(
            ok, "global_override", all_ok=all_ok
        ),
        "statistical_note": (
            "The exact sign tests treat domains, not learner/budget cells, as the "
            "independent units and are emitted only when all 160 frozen cells succeeded. "
            "Cell win/tie/loss counts are descriptive only."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument(
        "--output", default=str(ROOT / "results" / "paper_a_amlgym_matrix.json")
    )
    args = parser.parse_args()

    contract = json.loads((ROOT / "configs" / "amlgym_q1_contract.json").read_text())
    expected = {
        (domain, algorithm, int(budget))
        for domain in contract["domains"]
        for algorithm in contract["learner_families"]
        for budget in contract["trace_budgets"]
    }

    cases = []
    for path in sorted(Path(args.input_dir).rglob("*.json")):
        try:
            obj = json.loads(path.read_text())
        except Exception as exc:
            cases.append(
                {"status": "failed", "path": str(path), "error": f"invalid JSON: {exc}"}
            )
            continue
        if "domain" in obj and "algorithm" in obj and "trace_budget" in obj:
            cases.append(obj)

    seen = {_case_key(c) for c in cases if c.get("domain") is not None}
    missing = sorted(expected - seen)
    unexpected = sorted(seen - expected)
    complete = not missing and not unexpected and len(cases) == len(expected)
    all_ok = complete and all(c.get("status") == "ok" for c in cases)
    report = {
        "schema": "dovod-q1-amlgym-merged-v4",
        "expected_case_count": len(expected),
        "observed_case_count": len(cases),
        "complete": complete,
        "all_ok": all_ok,
        "missing": [list(x) for x in missing],
        "unexpected": [list(x) for x in unexpected],
        "summary": summarize(cases, all_ok=all_ok),
        "cases": cases,
        "claim_boundary": (
            "Primary external benchmark artifact. Failed cells remain in the denominator "
            "and are never silently dropped. DOVOD must beat not only the upstream model "
            "but also the fitted non-contextual global-override comparator before a broad "
            "contextual-repair benefit claim is made."
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "complete",
                    "all_ok",
                    "expected_case_count",
                    "observed_case_count",
                    "summary",
                )
            },
            indent=2,
        )
    )
    raise SystemExit(0 if complete else 3)


if __name__ == "__main__":
    main()
