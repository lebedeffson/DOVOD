from __future__ import annotations

from benchmarks.run_paper_a_carrier_coverage_curve import discovery_probability


def test_discovery_probability_boundaries():
    assert discovery_probability(5, 1, 0) == 0.0
    assert discovery_probability(5, 1, 5) == 1.0
    assert discovery_probability(5, 5, 1) == 1.0


def test_discovery_probability_single_carrier_is_sample_fraction():
    # One supporting carrier among m is discovered exactly when that carrier is sampled.
    assert abs(discovery_probability(10, 1, 3) - 0.3) < 1e-12


def test_discovery_probability_known_hypergeometric_case():
    # m=5, k=2, s=2: miss probability = C(3,2)/C(5,2) = 3/10.
    assert abs(discovery_probability(5, 2, 2) - 0.7) < 1e-12


def test_more_sampled_carriers_never_reduce_discovery_probability():
    vals = [discovery_probability(11, 3, s) for s in range(12)]
    assert all(a <= b + 1e-15 for a, b in zip(vals, vals[1:]))
