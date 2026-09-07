from paper_a.amlgym_bridge import DecisionObservation
from paper_a.simple_baselines import (
    fit_frequency_one_edit,
    predict_one_edit,
    random_one_edit,
)


def _obs(state, base, truth):
    return DecisionObservation(
        state_literals=tuple(state),
        action_label="(move a)",
        base_allow=base,
        truth_allow=truth,
    )


def test_frequency_one_edit_finds_simple_exception():
    rows = (
        _obs(("(clear a)",), 0, 1),
        _obs(("(clear a)",), 0, 1),
        _obs((), 0, 0),
        _obs((), 0, 0),
        _obs(("(clear a)",), 1, 1),
        _obs((), 1, 1),
    )
    model = fit_frequency_one_edit("move", rows, max_features=4, edit_penalty=0.1)
    preds = [predict_one_edit(model, r) for r in rows]
    assert model.mode == "exception_if_present"
    assert preds == [1, 1, 0, 0, 1, 1]


def test_random_one_edit_is_deterministic_and_label_independent():
    rows_a = (
        _obs(("(clear a)",), 0, 1),
        _obs(("(holding a)",), 1, 0),
        _obs(("(clear a)", "(holding a)"), 0, 1),
        _obs((), 1, 0),
    )
    rows_b = tuple(
        DecisionObservation(r.state_literals, r.action_label, r.base_allow, 1-r.truth_allow)
        for r in rows_a
    )
    a = random_one_edit("move", rows_a, seed_key="cell|0", max_features=4)
    b = random_one_edit("move", rows_a, seed_key="cell|0", max_features=4)
    c = random_one_edit("move", rows_b, seed_key="cell|0", max_features=4)
    assert a == b
    assert (a.mode, a.feature) == (c.mode, c.feature)
