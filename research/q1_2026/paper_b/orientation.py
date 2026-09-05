from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import log2


@dataclass(frozen=True)
class CalibrationResult:
    reliability: float
    calibration_observation: int
    posterior_honest_orientation: float
    raw_direct_correctness: float
    optimal_oriented_accuracy: float
    bayes_error: float


def _check_r(r: float) -> float:
    r = float(r)
    if not 0.5 < r < 1.0:
        raise ValueError("r must lie in (0.5,1)")
    return r


def sequence_likelihood(sequence: tuple[int, ...], truth: int, r: float) -> float:
    r = _check_r(r)
    truth = int(truth)
    if truth not in (0, 1):
        raise ValueError("truth must be binary")
    k = len(sequence)
    m = sum(int(o == truth) for o in sequence)
    return 0.5 * (r**m) * ((1-r)**(k-m)) + 0.5 * ((1-r)**m) * (r**(k-m))


def direct_mutual_information(k: int, r: float) -> float:
    if k < 0:
        raise ValueError("k must be non-negative")
    mi = 0.0
    for seq in product((0, 1), repeat=k):
        py = []
        for y in (0, 1):
            py.append(0.5 * sequence_likelihood(seq, y, r))
        po = sum(py)
        for joint in py:
            if joint > 0 and po > 0:
                mi += joint * log2(joint / (0.5 * po))
    return mi


def posterior_honest_orientation(r: float, calibration_observation: int) -> float:
    r = _check_r(r)
    c = int(calibration_observation)
    if c not in (0, 1):
        raise ValueError("calibration observation must be binary")
    return r if c == 1 else 1.0 - r


def raw_direct_correctness_after_calibration(r: float, calibration_observation: int) -> float:
    r = _check_r(r)
    pi = posterior_honest_orientation(r, calibration_observation)
    return pi*r + (1-pi)*(1-r)


def oriented_accuracy_after_calibration(r: float) -> float:
    r = _check_r(r)
    return r*r + (1-r)*(1-r)


def bayes_error_after_calibration_and_direct(r: float) -> float:
    r = _check_r(r)
    return 2.0*r*(1.0-r)


def calibration_risk_gain(r: float) -> float:
    r = _check_r(r)
    return 2.0*(r-0.5)**2


def two_query_policy_is_better(r: float, calibration_cost: float, direct_cost: float) -> bool:
    if calibration_cost < 0 or direct_cost < 0:
        raise ValueError("costs must be non-negative")
    return calibration_cost + direct_cost < calibration_risk_gain(r)


def result_for_calibration_outcome(r: float, c: int) -> CalibrationResult:
    raw = raw_direct_correctness_after_calibration(r, c)
    acc = max(raw, 1.0-raw)
    return CalibrationResult(
        reliability=r,
        calibration_observation=c,
        posterior_honest_orientation=posterior_honest_orientation(r, c),
        raw_direct_correctness=raw,
        optimal_oriented_accuracy=acc,
        bayes_error=1.0-acc,
    )
