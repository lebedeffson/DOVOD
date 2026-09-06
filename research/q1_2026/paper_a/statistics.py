from __future__ import annotations

from dataclasses import dataclass
from math import comb, log, sqrt
from typing import Sequence

from scipy.stats import beta, binomtest


@dataclass(frozen=True)
class ErrorCertificate:
    n: int
    errors: int
    empirical_risk: float
    upper_risk: float
    alpha: float


@dataclass(frozen=True)
class PairedComparison:
    n: int
    baseline_only_errors: int
    repaired_only_errors: int
    both_wrong: int
    both_correct: int
    risk_baseline: float
    risk_repaired: float
    risk_difference: float
    mcnemar_exact_pvalue: float


def clopper_pearson_upper(errors: int, n: int, *, alpha: float = 0.05) -> float:
    """One-sided exact Clopper-Pearson upper confidence limit for Bernoulli risk."""
    errors = int(errors)
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= errors <= n:
        raise ValueError("errors must lie in [0,n]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    if errors == n:
        return 1.0
    return float(beta.ppf(1.0 - alpha, errors + 1, n - errors))


def certify_binary_errors(
    labels: Sequence[int], predictions: Sequence[int], *, alpha: float = 0.05
) -> ErrorCertificate:
    labels = tuple(map(int, labels))
    predictions = tuple(map(int, predictions))
    if len(labels) != len(predictions):
        raise ValueError("length mismatch")
    if not labels:
        raise ValueError("empty sample")
    if any(y not in (0, 1) for y in labels + predictions):
        raise ValueError("labels and predictions must be binary")
    errors = sum(y != p for y, p in zip(labels, predictions))
    return ErrorCertificate(
        n=len(labels),
        errors=errors,
        empirical_risk=errors / len(labels),
        upper_risk=clopper_pearson_upper(errors, len(labels), alpha=alpha),
        alpha=float(alpha),
    )


def paired_error_comparison(
    labels: Sequence[int], baseline: Sequence[int], repaired: Sequence[int]
) -> PairedComparison:
    labels = tuple(map(int, labels))
    baseline = tuple(map(int, baseline))
    repaired = tuple(map(int, repaired))
    if not (len(labels) == len(baseline) == len(repaired)):
        raise ValueError("length mismatch")
    if not labels:
        raise ValueError("empty sample")
    if any(v not in (0, 1) for seq in (labels, baseline, repaired) for v in seq):
        raise ValueError("all values must be binary")

    b_only = r_only = both_wrong = both_correct = 0
    for y, b, r in zip(labels, baseline, repaired):
        be = b != y
        re = r != y
        if be and not re:
            b_only += 1
        elif re and not be:
            r_only += 1
        elif be and re:
            both_wrong += 1
        else:
            both_correct += 1

    discordant = b_only + r_only
    pvalue = 1.0 if discordant == 0 else float(
        binomtest(min(b_only, r_only), discordant, p=0.5, alternative="two-sided").pvalue
    )
    n = len(labels)
    rb = (b_only + both_wrong) / n
    rr = (r_only + both_wrong) / n
    return PairedComparison(
        n=n,
        baseline_only_errors=b_only,
        repaired_only_errors=r_only,
        both_wrong=both_wrong,
        both_correct=both_correct,
        risk_baseline=rb,
        risk_repaired=rr,
        risk_difference=rr - rb,
        mcnemar_exact_pvalue=pvalue,
    )


def exact_sign_test(improvements: Sequence[float]) -> dict[str, float | int]:
    """Two-sided exact sign test over independent units, dropping exact ties."""
    vals = tuple(float(x) for x in improvements)
    wins = sum(x > 0 for x in vals)
    losses = sum(x < 0 for x in vals)
    ties = len(vals) - wins - losses
    n = wins + losses
    pvalue = 1.0 if n == 0 else float(
        binomtest(min(wins, losses), n, p=0.5, alternative="two-sided").pvalue
    )
    return {"wins": wins, "losses": losses, "ties": ties, "n_nonties": n, "pvalue": pvalue}


def finite_mask_count(vocabulary_size: int, max_edits: int) -> int:
    """Number of edit masks of size at most ``max_edits`` in a fixed vocabulary."""
    m = int(vocabulary_size)
    k = int(max_edits)
    if m < 0:
        raise ValueError("vocabulary_size must be non-negative")
    if k < 0:
        raise ValueError("max_edits must be non-negative")
    k = min(k, m)
    return sum(comb(m, j) for j in range(k + 1))


def zero_error_uniform_upper(
    n: int, vocabulary_size: int, max_edits: int, *, delta: float = 0.05
) -> float:
    """Uniform zero-training-error risk bound for a fixed finite edit class.

    If a learner may select *any* mask with at most ``max_edits`` edits from a
    vocabulary fixed before labels are inspected, then with probability at least
    ``1-delta`` every selected mask that makes zero errors on ``n`` IID examples has
    true risk at most this value.  It is the exact union-bound inversion

        |H| (1-epsilon)^n <= delta.

    Unlike a post-selection Clopper-Pearson interval, this bound explicitly pays for
    adaptive hypothesis selection on the same sample.
    """
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0,1)")
    h = finite_mask_count(vocabulary_size, max_edits)
    return float(1.0 - (delta / h) ** (1.0 / n))


def finite_class_hoeffding_upper(
    empirical_risk: float, n: int, hypothesis_count: int, *, delta: float = 0.05
) -> float:
    """Uniform Hoeffding upper bound over a fixed finite hypothesis class."""
    r = float(empirical_risk)
    n = int(n)
    h = int(hypothesis_count)
    if not 0.0 <= r <= 1.0:
        raise ValueError("empirical_risk must lie in [0,1]")
    if n <= 0 or h <= 0:
        raise ValueError("n and hypothesis_count must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0,1)")
    radius = sqrt(log(h / delta) / (2.0 * n))
    return float(min(1.0, r + radius))


def counterexample_detection_probability(counterexample_mass: float, n: int) -> float:
    """Probability of seeing at least one falsifying positive counterexample."""
    q = float(counterexample_mass)
    n = int(n)
    if not 0.0 <= q <= 1.0:
        raise ValueError("counterexample_mass must lie in [0,1]")
    if n < 0:
        raise ValueError("n must be non-negative")
    return float(1.0 - (1.0 - q) ** n)


def counterexample_samples_for_detection(
    counterexample_mass_lower: float, *, delta: float = 0.05
) -> int:
    """Minimum IID positive observations for >=1-delta discovery probability.

    Requires a declared lower bound ``q`` on the probability mass of successful
    states that falsify the candidate prerequisite.  With no positive lower bound on
    q there is no finite uniform sample guarantee.
    """
    q = float(counterexample_mass_lower)
    if not 0.0 < q <= 1.0:
        raise ValueError("counterexample_mass_lower must lie in (0,1]")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0,1)")
    if q == 1.0:
        return 1
    # Need (1-q)^n <= delta.  Ceiling implemented without floating-point edge loss.
    n = max(1, int(log(delta) / log(1.0 - q)))
    while (1.0 - q) ** n > delta:
        n += 1
    while n > 1 and (1.0 - q) ** (n - 1) <= delta:
        n -= 1
    return n
