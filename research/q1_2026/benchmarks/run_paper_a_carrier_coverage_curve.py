from __future__ import annotations

"""Carrier-coverage analysis for the real counterexample audit.

The existing audit records, for each refuted directed relation, how many independent
recordings (MECCANO) or participants (IMPACT) contain at least one successful
counterexample.  This script asks a practical sample-efficiency question without
introducing a new threshold or reusing a test set for tuning:

    If only s independent carriers had been available, how many of the refutations
    observed in the full audit would we expect to have discovered?

For a relation carried by k of m independent units, a uniformly chosen subset of s
units misses the relation with probability C(m-k, s)/C(m, s).  The expected number
of discovered refutations is therefore the sum of the complementary probabilities
across relations.  We also report an adversarial deletion guarantee: after removing
m-s arbitrary carriers, a relation is guaranteed to remain observed iff k > m-s.

The expectation is conditional on the frozen empirical carrier counts.  It is a
subsampling analysis of the observed audit, not a population confidence interval and
not evidence that unresolved relations are mechanically necessary.
"""

import csv
import json
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parents[1] / "data" / "processed" / "carriers"
OUT_JSON = ROOT / "results" / "paper_a_carrier_coverage_curve.json"
OUT_CSV = ROOT / "results" / "paper_a_carrier_coverage_curve.csv"


def discovery_probability(total_carriers: int, relation_carriers: int, sampled_carriers: int) -> float:
    m = int(total_carriers)
    k = int(relation_carriers)
    s = int(sampled_carriers)
    if m <= 0:
        raise ValueError("total_carriers must be positive")
    if not 1 <= k <= m:
        raise ValueError("relation_carriers must lie in [1, total_carriers]")
    if not 0 <= s <= m:
        raise ValueError("sampled_carriers must lie in [0, total_carriers]")
    if s == 0:
        return 0.0
    if s > m - k:
        return 1.0
    return float(1.0 - comb(m - k, s) / comb(m, s))


def summarize_dataset(payload: dict) -> dict:
    rows = payload["relations"]
    carrier_field = "recordings" if "independent_recordings" in payload else "participants"
    m = int(payload.get("independent_recordings", payload.get("independent_participants")))
    counts = [int(row[carrier_field]) for row in rows]
    if not counts:
        raise ValueError("no observed refuted relations")

    n_components = 1 + max(max(int(row["p"]), int(row["a"])) for row in rows)
    universe = n_components * (n_components - 1)
    observed = len(counts)

    curve = []
    for s in range(0, m + 1):
        expected = sum(discovery_probability(m, k, s) for k in counts)
        removed = m - s
        guaranteed = sum(k > removed for k in counts)
        curve.append(
            {
                "sampled_carriers": s,
                "expected_refutations_discovered": float(expected),
                "expected_fraction_of_full_observed_refutations": float(expected / observed),
                "expected_audit_queue_reduction_fraction_of_universe": float(expected / universe),
                "guaranteed_refutations_after_adversarial_deletion": int(guaranteed),
                "guaranteed_fraction_of_full_observed_refutations": float(guaranteed / observed),
            }
        )

    milestones = {}
    for target in (0.50, 0.75, 0.90, 0.95):
        hit = next(row for row in curve if row["expected_fraction_of_full_observed_refutations"] >= target)
        milestones[f"expected_{int(target * 100)}pct_full_refutations"] = int(hit["sampled_carriers"])

    return {
        "dataset": payload["dataset"],
        "carrier_unit": carrier_field,
        "independent_carriers": m,
        "directed_candidate_relations": universe,
        "full_observed_refutations": observed,
        "full_audit_queue_reduction_fraction": float(observed / universe),
        "milestones": milestones,
        "curve": curve,
    }


def main() -> None:
    payloads = [
        json.loads((DATA / "meccano_carrier_counts.json").read_text(encoding="utf-8")),
        json.loads((DATA / "impact_carrier_counts.json").read_text(encoding="utf-8")),
    ]
    datasets = [summarize_dataset(payload) for payload in payloads]
    report = {
        "schema": "dovod-paper-a-carrier-coverage-v1",
        "datasets": datasets,
        "analysis_contract": (
            "Expected discovery curves use exact hypergeometric subsampling conditional on the frozen empirical carrier counts. "
            "The adversarial curve is a deterministic worst-case carrier-deletion guarantee. No carrier-count threshold is tuned from these results."
        ),
        "claim_boundary": (
            "The curves quantify how the already observed real-data counterexample evidence accumulates across independent recordings or participants. "
            "They are not population confidence intervals, do not establish causal or mechanical necessity, and cannot recover refutations absent from the full observed data."
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "sampled_carriers",
                "expected_refutations_discovered",
                "expected_fraction_of_full_observed_refutations",
                "expected_audit_queue_reduction_fraction_of_universe",
                "guaranteed_refutations_after_adversarial_deletion",
                "guaranteed_fraction_of_full_observed_refutations",
            ],
        )
        writer.writeheader()
        for ds in datasets:
            for row in ds["curve"]:
                writer.writerow({"dataset": ds["dataset"], **row})

    compact = {
        ds["dataset"]: {
            "independent_carriers": ds["independent_carriers"],
            "full_observed_refutations": ds["full_observed_refutations"],
            "milestones": ds["milestones"],
            "half_carrier_point": ds["curve"][ds["independent_carriers"] // 2],
        }
        for ds in datasets
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
