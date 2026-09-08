from __future__ import annotations

from benchmarks.run_paper_b_stratified_screen_confirmation import (
    source_family,
    stratified_indices,
)
from paper_b.static_world import Query


def test_source_family_partition():
    assert source_family("state") == "physical"
    assert source_family("calibrate_physical") == "physical"
    assert source_family("model_feature") == "semantic"
    assert source_family("calibrate_semantic") == "semantic"


def test_stratified_screen_keeps_three_from_each_family():
    queries = (
        Query("p0", "state", 0, 0.1),
        Query("p1", "state", 1, 0.1),
        Query("p2", "calibrate_physical", 0, 0.1),
        Query("p3", "state", 2, 0.1),
        Query("s0", "model_feature", 0, 0.1),
        Query("s1", "calibrate_semantic", 0, 0.1),
        Query("s2", "model_feature", 1, 0.1),
        Query("s3", "model_feature", 2, 0.1),
    )
    order = tuple(range(len(queries)))
    selected = stratified_indices(order, queries)
    assert selected == (0, 1, 2, 4, 5, 6)
