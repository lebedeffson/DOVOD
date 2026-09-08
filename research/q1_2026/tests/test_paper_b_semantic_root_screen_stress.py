from __future__ import annotations

from benchmarks.run_paper_b_semantic_root_screen_stress import map_action, source_family


def test_source_family_partition():
    assert source_family("state") == "physical"
    assert source_family("calibrate_physical") == "physical"
    assert source_family("model_feature") == "semantic"
    assert source_family("calibrate_semantic") == "semantic"


def test_map_action_from_screen_to_full_vocabulary():
    selected = (0, 2, 4, 7, 8, 9)
    assert map_action(("QUERY", 3), selected) == ("QUERY", 7)
    assert map_action(("DECIDE", 0), selected) == ("DECIDE", 0)
