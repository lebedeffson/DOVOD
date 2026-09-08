from __future__ import annotations

from benchmarks.run_paper_b_adaptive_exact_triage import root_margin


def test_root_margin_is_best_vs_second_best_gap():
    values = {
        ("DECIDE", 0): 0.4,
        ("DECIDE", 1): 0.2,
        ("QUERY", 0): 0.25,
    }
    assert abs(root_margin(values) - 0.05) < 1e-12


def test_root_margin_allows_ties():
    values = {
        ("DECIDE", 0): 0.2,
        ("DECIDE", 1): 0.2,
        ("QUERY", 0): 0.3,
    }
    assert root_margin(values) == 0.0
