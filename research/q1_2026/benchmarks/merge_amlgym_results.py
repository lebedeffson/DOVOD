from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _iter_case_files(paths: list[str]):
    seen = set()
    for raw in paths:
        root = Path(raw)
        candidates = [root] if root.is_file() else root.rglob("*.json")
        for path in candidates:
            if path in seen:
                continue
            seen.add(path)
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if data.get("schema") == "dovod-q1-amlgym-case-v2":
                yield data


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+")
    p.add_argument("--contract", default=str(ROOT / "configs" / "amlgym_q1_contract.json"))
    p.add_argument("--output", default=str(ROOT / "results" / "paper_a_amlgym_matrix.json"))
    args = p.parse_args()

    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    cases = list(_iter_case_files(args.inputs))
    cases.sort(key=lambda c: (c.get("domain", ""), c.get("algorithm", ""), c.get("trace_budget", -1)))

    expected = {
        (d, a, int(n))
        for d in contract["domains"]
        for a in contract["learner_families"]
        for n in contract["trace_budgets"]
    }
    observed = {(c.get("domain"), c.get("algorithm"), int(c.get("trace_budget", -1))) for c in cases}
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    ok = [c for c in cases if c.get("status") == "ok"]
    with_decision = [c for c in ok if c.get("decision") is not None]

    deltas = []
    per_domain = defaultdict(list)
    for c in with_decision:
        test = c["decision"]["test"]
        if test["base"]["n"]:
            delta = float(test["base"]["risk"]) - float(test["dovod"]["risk"])
            deltas.append(delta)
            per_domain[c["domain"]].append(delta)

    domain_deltas = {d: sum(v) / len(v) for d, v in sorted(per_domain.items()) if v}
    report = {
        "schema": "dovod-q1-amlgym-matrix-v2",
        "contract": contract,
        "cases": cases,
        "expected_case_count": len(expected),
        "observed_case_count": len(cases),
        "ok_count": len(ok),
        "decision_bridge_count": len(with_decision),
        "missing_cells": [list(x) for x in missing],
        "unexpected_cells": [list(x) for x in unexpected],
        "complete_matrix": not missing and not unexpected,
        "mean_test_risk_reduction_over_cases": sum(deltas) / len(deltas) if deltas else None,
        "domain_mean_test_risk_reduction": domain_deltas,
        "domains_with_decision_evidence": len(domain_deltas),
        "improved_test_cases": sum(x > 0 for x in deltas),
        "worsened_test_cases": sum(x < 0 for x in deltas),
        "tied_test_cases": sum(x == 0 for x in deltas),
        "claim_boundary": (
            "Broad comparisons use domains as the intended statistical unit. Individual state/action pairs "
            "are not treated as independent replicates. Missing, failed and unsupported cells remain visible."
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
