from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "external_evidence" / "dovod_short_papers_v12_full.json"
OUT = ROOT / "results" / "paper_a_review_queue_sensitivity.json"


def hours(items: int, minutes_per_item: float) -> float:
    return float(items) * float(minutes_per_item) / 60.0


def main() -> None:
    frozen = json.loads(EVIDENCE.read_text(encoding="utf-8"))["A"]
    total = int(frozen["candidate_relations"])
    retained = {
        "MECCANO": total - int(frozen["meccano_refuted"]),
        "IMPACT": total - int(frozen["impact_refuted"]),
    }
    rows = []
    for minutes in (1.0, 3.0, 5.0, 10.0):
        for dataset, kept in retained.items():
            before = hours(total, minutes)
            after = hours(kept, minutes)
            rows.append(
                {
                    "dataset": dataset,
                    "minutes_per_reviewed_hypothesis": minutes,
                    "hypotheses_before": total,
                    "hypotheses_after_audit": kept,
                    "review_hours_before": before,
                    "review_hours_after": after,
                    "review_hours_avoided": before - after,
                    "fraction_of_queue_removed": 1.0 - kept / total,
                }
            )
    report = {
        "schema": "dovod-paper-a-review-queue-sensitivity-v1",
        "source_counts": {
            "candidate_relations": total,
            "MECCANO_retained": retained["MECCANO"],
            "IMPACT_retained": retained["IMPACT"],
        },
        "rows": rows,
        "claim_boundary": (
            "Parametric translation of already-frozen audit counts into hypothetical review time. "
            "The 1/3/5/10 minute rates are sensitivity assumptions, not measured human-study timings; "
            "therefore these values must be reported as scenario estimates rather than empirical labor savings."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
