from __future__ import annotations

from benchmarks.run_paper_a_amlgym_practical_case import (
    _fit_threshold_policy,
    _gate_threshold_policies,
    _threshold_metrics,
)
from paper_a.amlgym_bridge import DecisionObservation


def obs(operator: str, base: int, truth: int, i: int) -> DecisionObservation:
    return DecisionObservation((f'(p {i})',), f'({operator} x)', base, truth)


def test_coarse_threshold_can_create_false_allows_but_gate_can_reject():
    rows = []
    for i in range(8):
        rows.append(('repair', obs('op', 0, 1, i)))
    for i in range(8, 10):
        rows.append(('repair', obs('op', 0, 0, i)))
    rows += [('calibration', obs('op', 0, 0, 10)), ('calibration', obs('op', 0, 0, 11))]
    rows += [('test', obs('op', 0, 0, 12)), ('test', obs('op', 0, 1, 13))]

    policies = _fit_threshold_policy(rows, threshold=0.8, min_samples=4)
    assert policies['op']['override_blocks'] is True
    raw = _threshold_metrics(rows, policies)
    assert raw['test']['false_allows'] == 1

    gates = _gate_threshold_policies(rows, policies)
    assert gates['op'].deployed is False
    gated = _threshold_metrics(rows, policies, gates)
    assert gated['test']['false_allows'] == 0
