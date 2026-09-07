from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.run_paper_a_amlgym_practical_case import run as run_v1


def normalize_audit_semantics(report: dict) -> dict:
    """Separate calibration-gate no-ops from actual non-empty repairs.

    The v1 explanatory runner inherited the deployment gate convention that an
    empty candidate reproducing the upstream decision is marked as deployed for
    bookkeeping. That is correct for prediction, but it must not be counted as
    a deployed *repair* in an audit-cost table.
    """
    if report.get("status") != "ok":
        return report
    details = report.get("operator_details") or {}
    for row in details.values():
        gate_deployed = bool(row.get("deployed"))
        nonempty_deployed = gate_deployed and int(row.get("candidate_edit_count", 0)) > 0
        row["gate_deployed_including_empty_noop"] = gate_deployed
        row["nonempty_contextual_repair_deployed"] = nonempty_deployed
        row["deployed"] = nonempty_deployed
    audit = report.setdefault("audit", {})
    audit["operators_with_deployed_contextual_repair"] = sum(
        bool(row.get("nonempty_contextual_repair_deployed")) for row in details.values()
    )
    audit["flagged_operators_left_unresolved_after_calibration_gate"] = sum(
        bool(row.get("flagged_by_counterexample"))
        and not bool(row.get("nonempty_contextual_repair_deployed"))
        for row in details.values()
    )
    audit["gate_bookkeeping_noops"] = sum(
        bool(row.get("gate_deployed_including_empty_noop"))
        and not bool(row.get("nonempty_contextual_repair_deployed"))
        for row in details.values()
    )
    audit["boundary"] = (
        "Flagged/deployed/unresolved counts are operator-level applicability audit units. "
        "A deployed contextual repair means a calibration-approved candidate with at least one selected edit; "
        "empty no-op gates are reported separately and never counted as repairs. "
        "The lifted precondition count is schema context only."
    )
    report["schema"] = "dovod-paper-a-amlgym-practical-case-v2"
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--algorithm", required=True)
    ap.add_argument("--trace-budget", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--max-problems", type=int, default=2)
    ap.add_argument("--max-states", type=int, default=12)
    ap.add_argument("--pilot-states-per-problem", type=int, default=12)
    ap.add_argument("--max-actions-per-operator", type=int, default=4)
    ap.add_argument("--min-repair-samples", type=int, default=4)
    ap.add_argument("--max-features", type=int, default=8)
    ap.add_argument("--context-width", type=int, default=1)
    ap.add_argument("--edit-penalty", type=float, default=0.25)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--threshold-min-samples", type=int, default=4)
    args = ap.parse_args()
    try:
        report = normalize_audit_semantics(run_v1(args))
    except Exception as exc:
        report = {
            "schema": "dovod-paper-a-amlgym-practical-case-v2",
            "status": "failed",
            "domain": args.domain,
            "algorithm": args.algorithm,
            "trace_budget": args.trace_budget,
            "seed": args.seed,
            "error": f"{type(exc).__name__}: {exc}",
            "claim_boundary": "Upstream/tool failure retained as an outcome; no scientific metric is imputed.",
        }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
