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


def test_adapter_builds_train_only_prior_and_evaluates_test():
    train = []
    # Current action 0 has three plausible next actions, each with distinct
    # verb/noun attributes. Repeat transitions so support threshold is met.
    for i in range(3):
        train += [row(f"t{i}a", 0, 0, 0, 0), row(f"t{i}a", 10, 1, 1, 10)]
        train += [row(f"t{i}b", 0, 0, 0, 0), row(f"t{i}b", 10, 2, 2, 20)]
        train += [row(f"t{i}c", 0, 0, 0, 0), row(f"t{i}c", 10, 3, 3, 30)]
    test = [row("x", 0, 0, 0, 0), row("x", 10, 2, 2, 20)]
    report = m.evaluate(train, test, min_train_count=5)
    assert report["coverage"]["evaluated_transitions"] == 1
    assert report["coverage"]["train_sequences"] == 9
    assert report["comparison"]["exact_mean_realized_cost"] is not None


def test_policy_can_stop_when_information_is_too_expensive():
    props = {
        1: {"verb": 1, "noun": 10, "description": "a1"},
        2: {"verb": 2, "noun": 20, "description": "a2"},
    }
    prior = {1: 0.6, 2: 0.4}
    value, action = m.solve_policy(prior, props, {"verb": 1.0, "noun": 1.0}, horizon=2)
    assert action[0] == "DECIDE"
    assert abs(value - 0.4) < 1e-12
