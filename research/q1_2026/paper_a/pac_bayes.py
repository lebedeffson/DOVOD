from __future__ import annotations

import math


def binary_kl(q: float, p: float) -> float:
    if not (0.0 <= q <= 1.0 and 0.0 <= p <= 1.0):
        raise ValueError("probabilities must be in [0,1]")
    out = 0.0
    if q > 0:
        if p <= 0:
            return math.inf
        out += q * math.log(q / p)
    if q < 1:
        if p >= 1:
            return math.inf
        out += (1 - q) * math.log((1 - q) / (1 - p))
    return out


def sparse_description_kl(vocabulary_size: int, selected: int, rho: float) -> float:
    M = int(vocabulary_size)
    k = int(selected)
    if not (0 <= k <= M):
        raise ValueError("selected must lie in [0,M]")
    if not (0.0 < rho < 1.0):
        raise ValueError("rho must lie in (0,1)")
    return k * math.log(1.0 / rho) + (M - k) * math.log(1.0 / (1.0 - rho))


def pac_bayes_kl_rhs(
    n: int, delta: float, *, vocabulary_size: int, selected: int, rho: float
) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    if not (0 < delta < 1):
        raise ValueError("delta must lie in (0,1)")
    complexity = sparse_description_kl(vocabulary_size, selected, rho)
    return (complexity + math.log(2.0 * math.sqrt(n) / delta)) / n


def invert_binary_kl_upper(qhat: float, rhs: float, *, tol: float = 1e-12) -> float:
    if not 0 <= qhat <= 1:
        raise ValueError("qhat must be in [0,1]")
    if rhs < 0:
        raise ValueError("rhs must be non-negative")
    if qhat >= 1.0:
        return 1.0
    lo, hi = qhat, 1.0 - 1e-15
    if binary_kl(qhat, hi) <= rhs:
        return 1.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if binary_kl(qhat, mid) <= rhs:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol:
            break
    return min(1.0, lo)


def sparse_repair_risk_upper(
    empirical_risk: float,
    n: int,
    delta: float,
    *,
    vocabulary_size: int,
    selected: int,
    rho: float,
) -> float:
    rhs = pac_bayes_kl_rhs(
        n, delta, vocabulary_size=vocabulary_size, selected=selected, rho=rho
    )
    return invert_binary_kl_upper(empirical_risk, rhs)
