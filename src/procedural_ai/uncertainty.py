from __future__ import annotations

from collections.abc import Sequence
import numpy as np


def normalize(probabilities: Sequence[float]) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    total = float(p.sum())
    if total <= 0:
        raise ValueError("probabilities must have positive mass")
    return p / total


def bayes_update(prior: Sequence[float], likelihood: Sequence[float]) -> np.ndarray:
    """Generic discrete Bayesian update used by noisy-source experiments."""
    posterior = normalize(prior) * np.asarray(likelihood, dtype=float)
    return normalize(posterior)


def bernoulli_variance(p: float) -> float:
    p = float(p)
    return p * (1.0 - p)
