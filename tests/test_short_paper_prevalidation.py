from procedural_ai.prevalidation import (
    expected_runtime_cost_after_prevalidation,
    partition_graphs_by_actions,
    rank_prevalidation_subsets,
)


def graph_family():
    return [
        {0: [], 1: [], 2: [0], 3: [0]},
        {0: [], 1: [], 2: [1], 3: [0]},
        {0: [], 1: [], 2: [0], 3: [1]},
        {0: [], 1: [], 2: [1], 3: [1]},
    ]


def test_partition_reveals_requested_semantics():
    groups = partition_graphs_by_actions(graph_family(), [2])
    assert len(groups) == 2
    assert sorted(len(g) for g in groups) == [2, 2]


def test_prevalidation_can_reduce_downstream_cost():
    graphs = graph_family()
    q = [0.5, 0.5, 0.0, 0.0]
    base = expected_runtime_cost_after_prevalidation(
        q=q, graphs=graphs, action=2, queryable_components=[0, 1], reviewed_actions=[], semantic_cost=2.0
    )
    reviewed = expected_runtime_cost_after_prevalidation(
        q=q, graphs=graphs, action=2, queryable_components=[0, 1], reviewed_actions=[2], semantic_cost=2.0
    )
    assert reviewed.expected_runtime_cost <= base.expected_runtime_cost


def test_amortized_total_includes_one_time_review_cost():
    result = expected_runtime_cost_after_prevalidation(
        q=[0.5, 0.5, 0.0, 0.0],
        graphs=graph_family(),
        action=2,
        queryable_components=[0, 1],
        reviewed_actions=[2, 3],
        semantic_cost=2.0,
        prevalidation_review_cost=2.0,
        amortization_horizon=100,
    )
    assert result.prevalidation_cost == 4.0
    assert result.amortized_total_cost == result.expected_runtime_cost + 0.04


def test_ranked_subsets_are_sorted_by_declared_objective():
    rows = rank_prevalidation_subsets(
        q=[0.5, 0.5, 0.0, 0.0],
        graphs=graph_family(),
        action=2,
        queryable_components=[0, 1],
        candidate_actions=[2, 3],
        max_reviews=1,
        semantic_cost=2.0,
    )
    costs = [r.expected_runtime_cost for r in rows]
    assert costs == sorted(costs)
