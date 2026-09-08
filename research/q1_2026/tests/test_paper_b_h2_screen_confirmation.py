from __future__ import annotations

from benchmarks.run_paper_b_h2_screen_confirmation import source_family


def test_source_family_partition():
    assert source_family("state") == "physical"
    assert source_family("calibrate_physical") == "physical"
    assert source_family("model_feature") == "semantic"
    assert source_family("calibrate_semantic") == "semantic"
