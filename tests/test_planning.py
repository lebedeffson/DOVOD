from procedural_ai.planning import SourceAwareResolutionPlanner, MyopicSourceSelector


def test_exact_planner_resolves_small_problem():
    graphs = [
        {0: [], 1: [], 2: [0]},
        {0: [], 1: [], 2: [1]},
    ]
    q = [0.5, 0.5, 0.0]
    planner = SourceAwareResolutionPlanner(graphs, 2, [0,1], physical_cost=1.0, semantic_cost=1.0)
    decision = planner.solve(q)
    assert decision.kind in {"PHYSICAL_QUERY", "SEMANTIC_REVIEW"}
    assert decision.expected_remaining_cost > 0


def test_myopic_is_deterministic():
    graphs = [
        {0: [], 1: [], 2: [0]},
        {0: [], 1: [], 2: [1]},
    ]
    q = [0.5, 0.5, 0.0]
    planner = MyopicSourceSelector(graphs, 2, [0,1], physical_cost=1.0, semantic_cost=1.0)
    a = planner.solve_myopic(q)
    b = planner.solve_myopic(q)
    assert a == b
