import pytest

from procedural_ai.evidence_math import (
    expected_discovery_fraction,
    minimum_zero_failure_observations,
    probability_relation_missed,
    zero_failure_upper_bound,
)


def test_finite_population_miss_probability():
    assert probability_relation_missed(carriers=1, total_units=11, sampled_units=9) == pytest.approx(2 / 11)
    assert probability_relation_missed(carriers=3, total_units=11, sampled_units=9) == 0.0


def test_expected_recovery_is_monotone():
    counts = [1, 1, 2, 5, 11]
    values = [expected_discovery_fraction(counts, total_units=11, sampled_units=k) for k in range(1, 12)]
    assert values == sorted(values)
    assert values[-1] == pytest.approx(1.0)


def test_zero_failure_numbers():
    assert zero_failure_upper_bound(9) == pytest.approx(0.2831288356)
    assert zero_failure_upper_bound(11) == pytest.approx(0.2384041904)
    assert minimum_zero_failure_observations(0.05) == 59
