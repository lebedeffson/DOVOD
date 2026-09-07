from paper_b.robust_cost import minimax_regret_action


def test_minimax_regret_can_reject_nominal_query_when_cost_is_uncertain():
    q = ("QUERY", 0)
    d0 = ("DECIDE", 0)
    d1 = ("DECIDE", 1)
    result = minimax_regret_action({
        "cheap": {q: 0.20, d0: 0.50, d1: 0.70},
        "expensive": {q: 0.80, d0: 0.40, d1: 0.65},
    })
    assert result.action == d0
    assert abs(result.worst_case_regret - 0.30) < 1e-12


def test_minimax_regret_keeps_query_when_it_is_uniformly_good():
    q = ("QUERY", 0)
    d0 = ("DECIDE", 0)
    result = minimax_regret_action({
        "a": {q: 0.20, d0: 0.50},
        "b": {q: 0.25, d0: 0.45},
    })
    assert result.action == q
    assert result.worst_case_regret == 0.0
