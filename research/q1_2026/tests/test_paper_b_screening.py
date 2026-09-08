from __future__ import annotations

from paper_b.screening import (
    decision_bearing_query_indices,
    map_reduced_action,
    solve_typed_core,
)
from paper_b.static_world import Query, World


def test_decision_bearing_screen_drops_calibration_queries():
    queries = (
        Query("cp", "calibrate_physical", 0, 0.01),
        Query("cs", "calibrate_semantic", 0, 0.01),
        Query("s0", "state", 0, 0.01),
        Query("m0", "model_feature", 0, 0.01),
    )
    assert decision_bearing_query_indices(queries) == (2, 3)


def test_reduced_query_action_maps_to_original_vocabulary():
    assert map_reduced_action(("QUERY", 1), (2, 5, 7)) == ("QUERY", 5)
    assert map_reduced_action(("DECIDE", 0), (2, 5, 7)) == ("DECIDE", 0)


def test_solve_typed_core_returns_original_query_index():
    models = ((0,),)
    worlds = (
        World((0,), 0, 1.0, 1.0),
        World((1,), 0, 1.0, 1.0),
    )
    belief = (0.5, 0.5)
    queries = (
        Query("calibrate", "calibrate_physical", 0, 0.4),
        Query("inspect", "state", 0, 0.1),
    )
    result = solve_typed_core(belief, worlds, models, queries, horizon=1)
    assert result.selected_query_indices == (1,)
    assert result.action == ("QUERY", 1)
    assert abs(result.value - 0.1) < 1e-12
