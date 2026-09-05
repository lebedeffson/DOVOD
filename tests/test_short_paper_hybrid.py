from procedural_ai.hybrid import HybridSourceSelector, exact_state_upper_bound


def graphs():
    return [
        {0: [], 1: [], 2: [0]},
        {0: [], 1: [], 2: [1]},
    ]


def test_state_envelope():
    assert exact_state_upper_bound(10, 8) == 531441
    assert exact_state_upper_bound(0, 2) == 3


def test_exact_to_myopic_switch():
    q = [0.5, 0.5, 0.0]
    exact = HybridSourceSelector(graphs(), 2, [0, 1], max_exact_states=100)
    assert exact.solve(q).policy == "exact"

    myopic = HybridSourceSelector(graphs(), 2, [0, 1], max_exact_states=1)
    result = myopic.solve(q)
    assert result.policy == "myopic"
    assert result.state_upper_bound > 1
