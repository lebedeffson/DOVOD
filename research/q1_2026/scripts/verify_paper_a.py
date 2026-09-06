from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "results" / "paper_a_contextual_benchmark.json"
STRESS = ROOT / "results" / "paper_a_stress.json"


def main() -> None:
    b = json.loads(BENCH.read_text(encoding="utf-8"))
    s = json.loads(STRESS.read_text(encoding="utf-8"))

    assert b["schema"] == "dovod-paper-a-contextual-repair-v4"
    assert b["problem"]["frozen_vocabulary_size"] == 128
    assert b["fit"]["train_fit_exact"] is True
    assert b["fit"]["exact_planted_edit_recovery"] is True
    assert b["fit"]["selected_edit_count"] == 3

    assert b["complete_support"]["contextual_risk"] == 0.0
    assert b["complete_support"]["base_risk"] >= 0.10
    assert b["complete_support"]["global_deletion_risk"] >= 0.10

    holdout = b["independent_holdout"]
    assert holdout["contextual"]["errors"] == 0
    assert holdout["contextual"]["upper_risk"] < 0.002
    assert holdout["contextual_false_allow"]["errors"] == 0
    assert holdout["contextual_false_block"]["errors"] == 0
    assert holdout["base_vs_contextual"]["risk_difference"] < -0.10
    assert holdout["global_vs_contextual"]["risk_difference"] < -0.10

    cert = b["finite_class_training_certificate"]
    assert cert["hypothesis_count_size_at_most_selected"] == 349633
    assert cert["zero_training_error_uniform_risk_upper"] < 0.025
    assert b["counterexample_discovery"]["minimum_positive_observations"] == 59

    assert s["schema"] == "dovod-paper-a-stress-v2"
    lookup = {(row["n"], row["label_noise"]): row for row in s["summary"]}
    assert lookup[(256, 0.0)]["exact_recovery_rate"] == 1.0
    assert lookup[(256, 0.0)]["contextual_risk_mean"] == 0.0
    assert lookup[(640, 0.0)]["exact_recovery_rate"] == 1.0
    assert lookup[(640, 0.0)]["contextual_risk_mean"] == 0.0
    assert lookup[(640, 0.05)]["contextual_risk_mean"] < 0.02
    assert lookup[(640, 0.05)]["contextual_risk_max"] < 0.05
    assert lookup[(640, 0.05)]["contextual_risk_mean"] < lookup[(640, 0.05)]["global_risk_mean"]

    summary = {
        "selected_edits": b["fit"]["selected_edit_count"],
        "holdout_errors": holdout["contextual"]["errors"],
        "holdout_upper95": holdout["contextual"]["upper_risk"],
        "finite_class_upper95": cert["zero_training_error_uniform_risk_upper"],
        "stress_n640_noise5_mean": lookup[(640, 0.05)]["contextual_risk_mean"],
        "stress_n640_noise5_max": lookup[(640, 0.05)]["contextual_risk_max"],
    }
    print("Paper A verification: PASS")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
