from procedural_ai.certificates import issue_certificate, verify_certificate


def test_certificate_roundtrip():
    graph = {0: [], 1: [0], 2: [0, 1]}
    state = [1, 1, 0]

    certificate = issue_certificate(state, graph, action=2)

    assert certificate.authorized is True
    assert certificate.prerequisites == (0, 1)
    assert verify_certificate(state, graph, certificate) is True


def test_certificate_is_bound_to_state_and_graph():
    graph = {0: [], 1: [0], 2: [0, 1]}
    certificate = issue_certificate([1, 1, 0], graph, action=2)

    assert verify_certificate([1, 0, 0], graph, certificate) is False
    assert verify_certificate([1, 1, 0], {0: [], 1: [0], 2: [1]}, certificate) is False
