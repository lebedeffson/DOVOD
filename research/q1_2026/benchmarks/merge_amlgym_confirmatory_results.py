from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "amlgym_q1_contract.json"


def case_key(case):
    return (str(case["domain"]), str(case["algorithm"]), int(case["trace_budget"]))


def expected_keys(contract):
    return {
        (domain, algorithm, int(budget))
        for domain in contract["domains"]
        for algorithm in contract["learner_families"]
        for budget in contract["trace_budgets"]
    }


def load_cases(root: Path):
    cases = []
    for path in sorted(root.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("schema") == "dovod-q1-amlgym-confirmatory-case-v1":
            data["_artifact_path"] = str(path)
            cases.append(data)
        elif data.get("schema") == "dovod-q1-amlgym-confirmatory-shard-v1":
            for case in data.get("cases", []):
                if case.get("schema") == "dovod-q1-amlgym-confirmatory-case-v1":
                    copied = dict(case)
                    copied["_artifact_path"] = str(path)
                    cases.append(copied)
    return cases


def exact_domain_sign_test(domain_deltas):
    nonzero = [delta for delta in domain_deltas.values() if abs(delta) > 1e-15]
    wins = sum(delta > 0 for delta in nonzero)
    losses = sum(delta < 0 for delta in nonzero)
    if not nonzero:
        return {"wins": 0, "losses": 0, "ties": len(domain_deltas), "two_sided_p": None}
    p = float(binomtest(wins, n=len(nonzero), p=0.5, alternative="two-sided").pvalue)
    return {
        "wins": wins,
        "losses": losses,
        "ties": len(domain_deltas) - len(nonzero),
        "two_sided_p": p,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root")
    parser.add_argument("--contract", default=str(CONTRACT))
    parser.add_argument("--output", default=str(ROOT / "results" / "paper_a_amlgym_confirmatory_matrix.json"))
    args = parser.parse_args()

    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    if contract.get("schema") != "dovod-q1-amlgym-contract-v4":
        raise SystemExit(f"unexpected contract schema: {contract.get('schema')}")

    raw_cases = load_cases(Path(args.artifact_root))
    by_key = defaultdict(list)
    for case in raw_cases:
        by_key[case_key(case)].append(case)

    duplicates = {key: rows for key, rows in by_key.items() if len(rows) > 1}
    unique = {key: rows[0] for key, rows in by_key.items() if len(rows) == 1}
    expected = expected_keys(contract)
    observed = set(unique)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)

    protocol_violations = []
    scientific_rows = []
    failure_rows = []
    for key in sorted(expected.intersection(observed)):
        case = unique[key]
        if case.get("status") != "ok":
            failure_rows.append({
                "key": list(key),
                "status": case.get("status"),
                "failure_stage": case.get("failure_stage"),
                "error": case.get("error"),
            })
            continue
        protocol = case.get("protocol") or {}
        if protocol.get("stage") != "confirmatory":
            protocol_violations.append({"key": list(key), "reason": "wrong_stage"})
        if protocol.get("pilot_overlap_count") != 0:
            protocol_violations.append({key": list(key), "reason": "pilot_overlap"})
        if protocol.get("split_unit") != "semantic_state_fingerprint":
            protocol_violations.append({key": list(key), "reason": "wrong_split_unit"})

        test = (case.get("decision") or {}).get("test") or {}
        base = test.get("base") or {}
        dovod = test.get("dovod") or {}
        global_baseline = test.get("global_override_gated") or {}
        if not base.get("n"):
            scientific_rows.append({key": list(key), "test_n": int(base.get("n") or 0), "risk_reduction": None, "global_to_dovod_risk_reduction": None, "class_balanced_risk_reduction": None})
            continue
        base_risk = float(base["risk"])
        dovod_risk = float(dovod["risk"])
        base_bal = base.get("class_balanced_risk")
        dovod_bal = dovod.get("class_balanced_risk")
        scientific_rows.append({
            "key": list(key),
            "test_n": int(base["n"]),
            "base_risk": base_risk,
            "dovod_risk": dovod_risk,
            "risk_reduction": base_risk - dovod_risk,
            "global_baseline_risk": None if not global_baseline else float(global_baseline["risk"]),
            "global_to_dovod_risk_reduction": None if not global_baseline else float(global_baseline["risk"]) - dovod_risk,
            "base_false_allows": int(base["false_allows"]),
            "dovod_false_allows": int(dovod["false_allows"]),
            "base_false_blocks": int(base["false_blocks"]),
            "dovod_false_blocks": int(dovod["false_blocks"]),
            "class_balanced_risk_reduction": (
                None if base_bal is None or dovod_bal is None else float(base_bal) - float(dovod_bal)
            ),
        })

    usable = [row for row in scientific_rows if row.get("risk_reduction") is not None]
    domain_values = defaultdict(list)
    global_domain_values = defaultdict(list)
    for row in usable:
        domain_values[row["key"][0]].append(float(row["risk_reduction"]))
        if row.get("global_to_dovod_risk_reduction") is not None:
            global_domain_values[row["key"][0]].append(float(row["global_to_dovod_risk_reduction"]))
    domain_deltas = {domain: sum(values) / len(values) for domain, values in sorted(domain_values.items())}
    global_domain_deltas = {domain: sum(values) / len(values) for domain, values in sorted(global_domain_values.items())}

    balanced_rows = [row for row in usable if row.get("class_balanced_risk_reduction") is not None]
    mean_risk_reduction = None if not usable else sum(row["risk_reduction"] for row in usable) / len(usable)
    mean_balanced_reduction = None if not balanced_rows else sum(row["class_balanced_risk_reduction"] for row in balanced_rows) / len(balanced_rows)

    complete_execution = (
        not missing
        and not unexpected
        and not duplicates
        and len(unique) == int(contract["confirmatory_integrity"]["expected_case_count"])
    )
    protocol_clean = not protocol_violations

    report = {
        "schema": "dovod-q1-amlgym-confirmatory-matrix-v1",
        "contract": contract,
        "observed_case_count": len(unique),
        "raw_case_records": len(raw_cases),
        "expected_case_count": len(expected),
        "complete_execution": complete_execution,
        "protocol_clean": protocol_clean,
        "missing_cells": [list(x) for x in missing],
        "unexpected_cells": [list(x) for x in unexpected],
        "duplicate_cells": {"|".join(map(str, key)): [row.get("_artifact_path") for row in rows] for key, rows in duplicates.items()},
        "failed_or_timeout_count": len(failure_rows),
        "failures": failure_rows,
        "protocol_violations": protocol_violations,
        "scientific_summary": {
            "usable_test_cells": len(usable),
            "improved_cells": sum(row["risk_reduction"] > 0 for row in usable),
            "tied_cells": sum(abs(row["risk_reduction"]) <= 1e-15 for row in usable),
            "worsened_cells": sum(row["risk_reduction"] < 0 for row in usable),
            "mean_cell_risk_reduction": mean_risk_reduction,
            "mean_cell_class_balanced_risk_reduction": mean_balanced_reduction,
            "domain_mean_risk_reduction": domain_deltas,
            "domain_sign_test": exact_domain_sign_test(domain_deltas),
            "domain_mean_global_to_dovod_risk_reduction": global_domain_deltas,
            "global_to_dovod_domain_sign_test": exact_domain_sign_test(global_domain_deltas),
            "rows": scientific_rows,
        },
        "claim_boundary": (
            "CI success depends only on execution completeness and protocol integrity, never on favorable scientific performance. "
            "All improved, tied, worsened, failed, and timeout cells remain in the artifact. Broad statistical interpretation uses domain-level summaries."
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "scientific_summary"}, indent=2, sort_keys=True))
    print(json.dumps(report["scientific_summary"], indent=2, sort_keys=True))

    if not complete_execution:
        raise SystemExit(3)
    if not protocol_clean:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
