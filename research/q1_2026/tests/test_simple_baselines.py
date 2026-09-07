from paper_a.amlgym_bridge import DecisionObservation
from paper_a.simple_baselines import (
    fit_counterexample_frequency_single_edit,
    fit_random_single_edit,
    predict_single_edit,
)


def obs(literals, base, truth):
    return DecisionObservation(tuple(literals), "(act x)", base, truth)


def test_frequency_single_edit_finds_local_exception():
    rows = (
        obs(("(ctx x)",), 0, 1),
        obs(("(ctx x)",), 0, 1),
        obs(("(other x)",), 0, 0),
        obs(("(other x)",), 1, 1),
    )
    model = fit_counterexample_frequency_single_edit("act", rows, max_features=2)
    assert model.edit is not None
    preds = [predict_single_edit(model, row) for row in rows]
    assert preds == [1, 1, 0, 1]
    assert model.repair_corrected_errors == 2
    assert model.repair_introduced_errors == 0


def test_frequency_single_edit_abstains_without_positive_net_gain():
    rows = (
        obs(("(ctx x)",), 0, 0),
        obs(("(ctx x)",), 1, 1),
        obs(("(other x)",), 0, 0),
        obs(("(other x)",), 1, 1),
    )
    model = fit_counterexample_frequency_single_edit("act", rows, max_features=2)
    assert model.edit is None
    assert [predict_single_edit(model, row) for row in rows] == [0, 1, 0, 1]


def test_random_single_edit_is_seed_deterministic_and_label_free_selection():
    rows_a = (
        obs(("(ctx x)",), 0, 1),
        obs(("(other x)",), 0, 0),
        obs(("(ctx x)",), 1, 1),
        obs(("(other x)",), 1, 0),
    )
    rows_b = tuple(DecisionObservation(r.state_literals, r.action_label, r.base_allow, 1-r.truth_allow) for r in rows_a)
    m1 = fit_random_single_edit("act", rows_a, seed=17, max_features=2)
    m2 = fit_random_single_edit("act", rows_a, seed=17, max_features=2)
    m3 = fit_random_single_edit("act", rows_b, seed=17, max_features=2)
    assert m1.edit == m2.edit == m3.edit
