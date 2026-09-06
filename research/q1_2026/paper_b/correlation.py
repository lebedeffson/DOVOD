from __future__ import annotations


def repeated_correctness_covariance(reliabilities: tuple[float, ...], weights: tuple[float, ...]) -> float:
    if len(reliabilities) != len(weights) or not reliabilities:
        raise ValueError("length mismatch")
    z = sum(weights)
    if z <= 0:
        raise ValueError("non-positive total weight")
    w = tuple(x/z for x in weights)
    mean = sum(a*b for a,b in zip(w,reliabilities))
    second = sum(a*(b**2) for a,b in zip(w,reliabilities))
    return second - mean**2


def repeated_correctness_correlation(reliabilities: tuple[float, ...], weights: tuple[float, ...]) -> float:
    z = sum(weights)
    w = tuple(x/z for x in weights)
    mean = sum(a*b for a,b in zip(w,reliabilities))
    denom = mean*(1.0-mean)
    if denom <= 0:
        raise ValueError("degenerate marginal")
    return repeated_correctness_covariance(reliabilities, weights)/denom


def all_correct_probability(reliabilities: tuple[float, ...], weights: tuple[float, ...], k: int) -> float:
    if k < 0:
        raise ValueError("k must be non-negative")
    z = sum(weights)
    w = tuple(x/z for x in weights)
    return sum(a*(r**k) for a,r in zip(w,reliabilities))


def mean_plugin_all_correct_probability(reliabilities: tuple[float, ...], weights: tuple[float, ...], k: int) -> float:
    z = sum(weights)
    w = tuple(x/z for x in weights)
    mean = sum(a*b for a,b in zip(w,reliabilities))
    return mean**k
