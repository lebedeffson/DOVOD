from __future__ import annotations

from benchmarks.run_paper_b_likelihood_misspecification import transform_reliability


def test_alpha_one_preserves_reliability():
    for r in (0.5, 0.6, 0.75, 0.9, 1.0):
        assert abs(transform_reliability(r, 1.0) - r) < 1e-12


def test_underconfidence_moves_toward_chance():
    assert transform_reliability(0.9, 0.5) == 0.7
    assert transform_reliability(0.7, 0.5) == 0.6


def test_overconfidence_is_clipped_to_valid_range():
    assert transform_reliability(0.95, 1.25) == 1.0
    assert 0.5 <= transform_reliability(0.6, 1.25) <= 1.0
