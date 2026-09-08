import pytest

from benchmarks import run_paper_b_assembly101_acquisition as m


def row(video, start, action, verb, noun):
    return {
        "video": video,
        "start_frame": start,
        "end_frame": start + 1,
        "action_id": action,
        "verb_id": verb,
        "noun_id": noun,
        "action_cls": f"a{action}",
        "verb_cls": f"v{verb}",
        "noun_cls": f"n{noun}",
    }


def small_train():
    train = []
    for i in range(3):
        train += [row(f"t{i}a", 0, 0, 0, 0), row(f"t{i}a", 10, 1, 1, 10)]
        train += [row(f"t{i}b", 0, 0, 0, 0), row(f"t{i}b", 10, 2, 2, 20)]
        train += [row(f"t{i}c", 0, 0, 0, 0), row(f"t{i}c", 10, 3, 3, 30)]
    return train


def test_adapter_builds_train_only_prior_and_evaluates_test():
    train = small_train()
    test = [row("x", 0, 0, 0, 0), row("x", 10, 2, 2, 20)]
    report = m.evaluate(train, test, min_train_count=5)
    assert report["coverage"]["evaluated_transitions"] == 1
    assert report["coverage"]["train_sequences"] == 9
    assert report["comparison"]["exact_mean_realized_cost"] is not None
    assert report["metadata_isolation"]["metadata_source_for_policy"] == "official_train_only"
    assert report["metadata_isolation"]["evaluated_transitions_using_test_only_action_metadata"] == 0


def test_test_only_action_metadata_never_enters_policy_properties():
    train = small_train()
    test = [
        row("x", 0, 0, 0, 0),
        row("x", 10, 99, 999, 999),
    ]
    props = m._properties(train)
    meta = m._validate_test_metadata(props, test)
    assert 99 not in props
    assert meta["unknown_test_action_id_values"] == [99]

    report = m.evaluate(train, test, min_train_count=5)
    assert report["coverage"]["evaluated_transitions"] == 0
    assert report["coverage"]["skip_counts"]["true_next_unseen_after_current_in_train"] == 1
    assert report["metadata_isolation"]["unknown_test_action_ids"] == 1


def test_shared_action_metadata_mismatch_is_rejected_not_imported():
    train = small_train()
    bad_test = [row("x", 0, 0, 0, 0), row("x", 10, 2, 777, 20)]
    with pytest.raises(ValueError, match="train/test verb/noun metadata mismatch"):
        m.evaluate(train, bad_test, min_train_count=5)


def test_aggregation_preserves_transition_frequency():
    train = small_train()
    test = []
    for i in range(4):
        test += [row(f"x{i}", 0, 0, 0, 0), row(f"x{i}", 10, 2, 2, 20)]
    report = m.evaluate(train, test, min_train_count=5, bootstrap_draws=50)
    assert report["coverage"]["raw_test_transitions"] == 4
    assert report["coverage"]["evaluated_transitions"] == 4
    assert len(report["rows"]) == 1
    assert report["rows"][0]["test_frequency"] == 4
    assert report["comparison"]["paired_exact_minus_myopic_bootstrap_ci95"] is not None


def test_policy_can_stop_when_information_is_too_expensive():
    props = {
        1: {"verb": 1, "noun": 10, "description": "a1"},
        2: {"verb": 2, "noun": 20, "description": "a2"},
    }
    prior = {1: 0.6, 2: 0.4}
    value, action = m.solve_policy(prior, props, {"verb": 1.0, "noun": 1.0}, horizon=2)
    assert action[0] == "DECIDE"
    assert abs(value - 0.4) < 1e-12
