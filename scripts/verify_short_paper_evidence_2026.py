from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def close(actual: float, expected: float, tol: float = 5e-6) -> None:
    assert math.isclose(float(actual), float(expected), abs_tol=tol, rel_tol=0.0), (actual, expected)


def main() -> None:
    data = json.loads((ROOT / "results/paper_claims_2026/frozen_evidence.json").read_text(encoding="utf-8"))
    A = data["A"]
    B = data["B"]

    assert data["manuscript_groups"] == 19
    assert data["secondary_groups"] == ["B12"]

    assert (A["candidate_relations"], A["meccano_refuted"], A["impact_refuted"]) == (272, 201, 259)
    close(A["nested_after"] - A["nested_before"], A["nested_gain"])
    assert A["nested_pairs"] == 110
    close(A["mean_pruned"], 3.8545454545)
    assert (A["meccano_robust_one_carrier"], A["impact_robust_one_carrier"]) == (162, 184)
    assert (A["rework_only_meccano"], A["rework_only_impact"]) == (16, 11)
    close(A["threshold1_recall"], 0.9013079612)
    close(A["threshold2_recall"], 0.7883626178)
    assert (A["rarefaction_meccano_95"], A["rarefaction_impact_participant_95"]) == (9, 11)
    assert A["zero_fail_n_for_below_005"] == 59

    assert (B["episodes"], B["mixed_episodes"]) == (777, 187)
    close((B["myopic_cost"] - B["bellman_cost"]) / B["myopic_cost"], B["relative_improvement"])
    close(B["source_change_rate"], 0.3315508021)
    close(B["regret_true_cost5_assumed1"], 0.247639)
    close(B["regret_true_cost10_assumed1"], 0.442896)
    close(B["prevalidation_runtime"] + B["prevalidation_one_time_cost"] / B["episodes"], B["prevalidation_amortized_total"])
    close((B["prevalidation_baseline"] - B["prevalidation_amortized_total"]) / B["prevalidation_baseline"], B["prevalidation_amortized_reduction"])
    assert B["k10_state_min"] == 157464
    assert B["k10_state_max"] == 531414
    assert B["k10_state_max"] <= B["k10_g8_upper_bound"] == 531441

    print("DOVOD final v12 evidence: PASS (19 manuscript groups + B12 preserved secondary evidence)")


if __name__ == "__main__":
    main()
