from __future__ import annotations

from benchmarks.run_paper_a_amlgym_practical_case_v2 import normalize_audit_semantics


def test_empty_noop_gate_is_not_counted_as_deployed_repair():
    report = {
        "schema": "dovod-paper-a-amlgym-practical-case-v1",
        "status": "ok",
        "audit": {},
        "operator_details": {
            "clean": {
                "deployed": True,
                "candidate_edit_count": 0,
                "flagged_by_counterexample": False,
            },
            "needs-repair": {
                "deployed": True,
                "candidate_edit_count": 2,
                "flagged_by_counterexample": True,
            },
            "flagged-but-noop": {
                "deployed": True,
                "candidate_edit_count": 0,
                "flagged_by_counterexample": True,
            },
        },
    }
    out = normalize_audit_semantics(report)
    assert out["schema"] == "dovod-paper-a-amlgym-practical-case-v2"
    assert out["audit"]["operators_with_deployed_contextual_repair"] == 1
    assert out["audit"]["flagged_operators_left_unresolved_after_calibration_gate"] == 1
    assert out["audit"]["gate_bookkeeping_noops"] == 2
    assert out["operator_details"]["clean"]["deployed"] is False
    assert out["operator_details"]["needs-repair"]["deployed"] is True
    assert out["operator_details"]["flagged-but-noop"]["deployed"] is False
