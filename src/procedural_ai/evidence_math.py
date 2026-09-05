from __future__ import annotations

from collections.abc import Iterable
from math import ceil, comb, log


def probability_relation_missed(*, carriers: int, total_units: int, sampled_units: int) -> float:
    """Exact finite-population probability that a sample misses all carriers."""
    carriers = int(carriers); total_units = int(total_units); sampled_units = int(sampled_units)
    if total_units < 1:
        raise ValueError("total_units must be positive")
    if not 0 <= carriers <= total_units:
        raise ValueError("carriers must be in [0,total_units]")
    if not 0 <= sampled_units <= total_units:
        raise ValueError("sampled_units must be in [0,total_units]")
    if sampled_units > total_units - carriers:
        return 0.0
    return comb(total_units - carriers, sampled_units) / comb(total_units, sampled_units)


def expected_discovery_fraction(carrier_counts: Iterable[int], *, total_units: int, sampled_units: int) -> float:
    """Expected fraction of the already-observed refutation set recovered by rarefaction."""
    counts = [int(c) for c in carrier_counts]
    if not counts:
        return 0.0
    if any(c < 1 or c > total_units for c in counts):
        raise ValueError("carrier counts outside finite population")
    return sum(
        1.0 - probability_relation_missed(carriers=c, total_units=total_units, sampled_units=sampled_units)
        for c in counts
    ) / len(counts)


def minimum_units_for_expected_recovery(carrier_counts: Iterable[int], *, total_units: int, target: float = 0.95) -> int:
    counts = tuple(int(c) for c in carrier_counts)
    if not counts:
        return 0
    if not 0 < target <= 1:
        raise ValueError("target must be in (0,1]")
    for k in range(1, total_units + 1):
        if expected_discovery_fraction(counts, total_units=total_units, sampled_units=k) >= target:
            return k
    return total_units


def zero_failure_upper_bound(observations: int, *, confidence: float = 0.95) -> float:
    """Exact one-sided Clopper-Pearson upper bound after zero Bernoulli failures."""
    n = int(observations)
    if n < 1 or not 0 < confidence < 1:
        raise ValueError
    return 1.0 - (1.0 - confidence) ** (1.0 / n)


def minimum_zero_failure_observations(max_upper_bound: float, *, confidence: float = 0.95) -> int:
    if not 0 < max_upper_bound < 1 or not 0 < confidence < 1:
        raise ValueError
    return int(ceil(log(1.0 - confidence) / log(1.0 - max_upper_bound)))
