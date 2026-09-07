from benchmarks.run_paper_a_assembly101_ordering_audit import Event, evaluate_relations, learn_relations, toy_id_from_name


def ev(verb, a, b, label="correct", remark=""):
    return Event(0, 1, verb, a, b, label, remark)


def test_toy_id_parser_matches_official_filename_shape():
    name = "nusar-2021_action_both_9013-c09c_9013_user_id_2021-02-24_113951.csv"
    assert toy_id_from_name(name) == "c09c"


def test_high_frequency_relation_can_be_refuted_by_heldout_correct_event_and_detect_order_mistake():
    predecessor = ev("attach", "wheel", "chassis")
    target = ev("attach", "body", "chassis")
    train = {f"s{i}": (predecessor, target) for i in range(10)}
    relations, _ = learn_relations(train, min_target_support=5, predecessor_frequency=0.9)
    key = (predecessor.token, target.token)
    assert key in relations
    test = {
        "correct-refutation": (target,),
        "order-mistake": (ev("attach", "body", "chassis", "mistake", "wrong order"),),
    }
    metrics = evaluate_relations(test, relations)
    assert metrics["correct_eligible"] == 1
    assert metrics["correct_violations"] == 1
    assert metrics["order_mistake_eligible"] == 1
    assert metrics["order_mistake_violations"] == 1
    assert metrics["relations_refuted_by_heldout_correct_behavior"] == 1


def test_correction_events_may_restore_history_for_evaluation():
    predecessor = ev("attach", "wheel", "chassis")
    target = ev("attach", "body", "chassis")
    relations = {(predecessor.token, target.token): {"train_frequency": 1.0}}
    correction = ev("attach", "wheel", "chassis", "correction")
    metrics = evaluate_relations({"s": (correction, target)}, relations)
    assert metrics["correct_eligible"] == 1
    assert metrics["correct_violations"] == 0
