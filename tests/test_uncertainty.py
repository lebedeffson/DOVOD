import numpy as np
import pytest

from procedural_ai.uncertainty import bayes_update, bernoulli_variance, normalize


def test_normalize_returns_probability_vector():
    p = normalize([2.0, 1.0, 1.0])
    assert np.allclose(p, [0.5, 0.25, 0.25])
    assert np.isclose(float(p.sum()), 1.0)


def test_bayes_update_is_normalized():
    posterior = bayes_update([0.5, 0.5], [0.9, 0.1])
    assert np.allclose(posterior, [0.9, 0.1])
    assert np.isclose(float(posterior.sum()), 1.0)


def test_zero_mass_is_rejected():
    with pytest.raises(ValueError):
        normalize([0.0, 0.0])


def test_bernoulli_variance_peaks_at_half():
    assert bernoulli_variance(0.5) == pytest.approx(0.25)
    assert bernoulli_variance(0.0) == pytest.approx(0.0)
    assert bernoulli_variance(1.0) == pytest.approx(0.0)
