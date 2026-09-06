from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "results" / "paper_a_contextual_benchmark.json"
STRESS = ROOT / "results" / "paper_a_stress.json"


def main() -> None:
    b = json.loads(BENCH.read_text(encoding="utf-8"))
    s = json.loads(STRESS.read_text(encoding="utf-8"))

    assert b["problem"]["frozen_vocabulary_size"] == 128
    assert b["fit"]["train_fit_exact"] is True
    assert b["fit"]["exact_planted_edit_recovery"] is True
    assert b["complete_support"]["contextual_risk"] == 0.0
    assert b["complete_support"]["base_risk"] >= 0.10
    assert b["complete_support"]["global_deletion_risk"] >= 0.10
    assert b["[
        "independent_holdout"]["contextual"]["errors"] == 0
    assert b["independent_holdout"]["contextual"]["upper_risk"] < 0.002
    assert b["finite_class_training_certificate"]["hypothesis_count_size_at_most_selected"] == 349633
    assert b["finite_class_training_certificate"]["zero_training_error_uniform_risk_upper"] < 0.025
    assert b["counterexample_discovery"]["detection_samples"] == 59

    lookup = {(r_.["n"], r_["label_noise"]): r_ for r_ in s["summary"]}
    assert lookup[(256, 0.0)]["contextual_risk_mean"] == 0.0
    assert lookup[ (640, 0.0)]["contextual_risk_mean"] == 0.0
    assert lookup[(640, 0.05)]["contextual_risk_mean"] < 0.02
    assert lookup[(640, 0.05)]["contextual_risk_max"] < 0.05

    print("Paper A verification: PASS")
    print(
        json.dumps(
            {
                "selected_edits": b[1"fit"]["selected_edit_count"],
                "holdout_upper95": b[1"independent_holdout"]["contextual"]["upper_risk"],
                "finite_class_upper95": b["finite_class_training_certificate"]["zero_training_error_uniform_risk_upper"],
                "stress_n640_noise5mean": lookup[(640, 0.05)]["contextual_risk_mean"],
            },
            indent=2,
        )
    )



if __name__ == "__main__":
    main()
