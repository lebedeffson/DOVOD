from __future__ import annotations

from benchmarks.run_paper_b_structural_core_screen import structural_core_indices
from paper_b.static_world import Query


def test_structural_core_drops_only_calibration_queries():
    queries = (
        Query("cp", "calibrate_physical", 0, 0.1),
        Query("cs", "calibrate_semantic", 0, 0.1),
        Query("s0", "state", 0, 0.1),
        Query("m0", "model_feature", 0, 0.1),
        Query("s1", "state", 1, 0.1),
        Query("m1", "model_feature", 1, 0.1),
    )
    assert structural_core_indices(queries) == (2, 3, 4, 5)
