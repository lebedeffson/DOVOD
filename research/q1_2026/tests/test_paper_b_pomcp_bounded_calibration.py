from __future__ import annotations

from benchmarks.run_paper_b_pomcp_bounded_calibration import (
    EXPLORATION_CANDIDATES,
    N_CONFIRMATION,
    N_DEVELOPMENT,
    POMCP_SEEDS,
    SIMULATIONS,
)


def test_bounded_protocol_is_nonempty_and_fixed():
    assert EXPLORATION_CANDIDATES == (1.25, 0.50, 0.20, 0.10, 0.05)
    assert N_DEVELOPMENT == 12
    assert N_CONFIRMATION == 12
    assert POMCP_SEEDS == (0, 1, 2)
    assert SIMULATIONS == 15000


def test_candidate_constants_are_positive_and_unique():
    assert all(c > 0 for c in EXPLORATION_CANDIDATES)
    assert len(set(EXPLORATION_CANDIDATES)) == len(EXPLORATION_CANDIDATES)
