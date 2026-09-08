from benchmarks import run_paper_b_assembly101_cluster_bootstrap as c


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


def training_rows():
    rows = []
    for i in range(3):
        rows += [row(f"a{i}", 0, 0, 0, 0), row(f"a{i}", 1, 1, 1, 10)]
        rows += [row(f"b{i}", 0, 0, 0, 0), row(f"b{i}", 1, 2, 2, 20)]
        rows += [row(f"c{i}", 0, 0, 0, 0), row(f"c{i}", 1, 3, 3, 30)]
    return rows


def test_video_cluster_bootstrap_uses_videos_not_transitions():
    test = []
    for i in range(4):
        test += [row(f"x{i}", 0, 0, 0, 0), row(f"x{i}", 1, 2, 2, 20)]
    report = c.analyze(training_rows(), test, min_train_count=5, draws=100, seed=123)
    assert report["coverage"]["test_videos"] == 4
    assert report["coverage"]["evaluated_test_videos"] == 4
    assert report["coverage"]["evaluated_transitions"] == 4
    assert report["cluster_analysis"]["resampling_unit"] == "official_test_video"
    assert report["cluster_analysis"]["bootstrap"]["draws"] == 100
    assert report["policy_isolation"]["test_only_metadata_used_for_policy"] is False


def test_cluster_analysis_excludes_test_only_action_metadata():
    test = [row("unknown", 0, 0, 0, 0), row("unknown", 1, 99, 999, 999)]
    report = c.analyze(training_rows(), test, min_train_count=5, draws=20, seed=1)
    assert report["coverage"]["evaluated_transitions"] == 0
    assert report["policy_isolation"]["unknown_test_action_id_values"] == [99]
    assert report["cluster_analysis"]["bootstrap"]["transition_weighted_ci95"] is None
