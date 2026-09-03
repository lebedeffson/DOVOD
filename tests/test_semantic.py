from procedural_ai.semantic import SemanticVersionSpace


def test_semantic_version_space_marks_ambiguity():
    graphs = [
        {0: [], 1: [], 2: [0]},
        {0: [], 1: [], 2: [1]},
    ]
    space = SemanticVersionSpace(graphs)
    d = space.evaluate([1,0,0])
    assert 2 in d.ambiguous_actions
