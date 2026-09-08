from __future__ import annotations

from benchmarks.run_paper_b_screened_exact import one_step_query_order
from paper_b.static_world import Query, World


def test_one_step_screen_ranks_informative_query_first():
    models = ((0,),)
    worlds = (
        World((0, 0), 0, 1.0, 1.0),
        World((1, 0), 0, 1.0, 1.0),
    )
    belief = (0.5, 0.5)
    queries = (
        Query("informative", "state", 0, 0.1),
        Query("uninformative", "state", 1, 0.1),
    )

    order, values, seconds = one_step_query_order(belief, worlds, models, queries)
    assert order[0] == 0
    assert values[0] < values[1]
    assert seconds >= 0.0
