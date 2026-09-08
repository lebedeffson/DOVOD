from __future__ import annotations

from benchmarks.run_paper_b_screen_transfer import map_action, query_family


def test_query_family_partition():
    assert query_family("state") == "physical"
    assert query_family("calibrate_physical") == "physical"
    assert query_family("model_feature") == "semantic"
    assert query_family("calibrate_semantic") == "semantic"


def test_map_action_from_reduced_to_full_query_index():
    selected = (2, 5, 8)
    assert map_action(("QUERY", 1), selected) == ("QUERY", 5)
    assert map_action(("DECIDE", 1), selected) == ("DECIDE", 1)
