from procedural_ai.procedure import action_set, infer_unary_prerequisites
from procedural_ai.evidence import prune_with_counterexample


def test_action_set_respects_prerequisites():
    graph = {0: [], 1: [0], 2: [0, 1]}
    assert action_set([1, 0, 0], graph) == [1]
    assert action_set([1, 1, 0], graph) == [2]


def test_counterexample_prunes_only_violated_candidate():
    graph = {0: [], 1: [0], 2: [0, 1]}
    rows = [("0", [1, 0, 0]), ("1", [1, 0, 1])]
    pruned = prune_with_counterexample(graph, rows)
    assert pruned[2] == [0]
    assert pruned[1] == [0]


def test_inference_is_one_sided_and_unary():
    records = [("train", "r1", [("0", [1,0,0]), ("1", [1,1,0]), ("2", [1,1,1])])]
    graph = infer_unary_prerequisites(records, n_components=3)
    assert 0 in graph[1]
    assert 0 in graph[2] and 1 in graph[2]
