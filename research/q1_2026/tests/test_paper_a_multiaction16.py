from benchmarks import run_paper_a_multiaction16 as m


def test_two_action_specs_overlap_but_are_distinct():
    specs = m.action_specs()
    assert set(specs) == {"action_A", "action_B"}
    assert set(specs["action_A"]["base"]) & set(specs["action_B"]["base"]) == {2, 3}
    assert set(specs["action_A"]["context_indices"]) & set(specs["action_B"]["context_indices"]) == {10, 11}
    assert all(len(x["planted"]) == 4 for x in specs.values())


def test_planted_edits_belong_to_frozen_vocabularies():
    for spec in m.action_specs().values():
        vocab = m._weighted_vocab(
            m._context_support(spec["context_indices"]),
            spec["base"],
            spec["context_indices"],
        )
        assert set(spec["planted"]).issubset(set(vocab))
