from __future__ import annotations

from benchmarks.run_paper_b_pomcp_budget_curve import BUDGETS, POMCP_SEEDS, summarize


def test_budgets_and_seeds_are_prespecified():
    assert tuple(sorted(BUDGETS)) == BUDGETS
    assert len(set(BUDGETS)) == len(BUDGETS)
    assert BUDGETS[0] > 0
    assert len(POMCP_SEEDS) >= 3
    assert len(set(POMCP_SEEDS)) == len(POMCP_SEEDS)


def test_summary_uses_all_cases_seeds_and_budgets():
    cases = []
    for case_id in range(3):
        rows = []
        for budget in BUDGETS:
            for pomcp_seed in POMCP_SEEDS:
                rows.append(
                    {
                        "simulations": budget,
                        "pomcp_seed": pomcp_seed,
                        "action_is_exact_optimal": (case_id + pomcp_seed + budget) % 2 == 0,
                        "action_family_matches_exact": (pomcp_seed % 2 == 0),
                        "true_root_action_regret": float(case_id + pomcp_seed) / budget,
                        "absolute_pomcp_value_error": float(case_id + 1) / budget,
                        "seconds": float(budget) / 1000.0,
                    }
                )
        cases.append({"rows": rows})

    summary_rows = summarize(cases)
    assert [r["simulations"] for r in summary_rows] == list(BUDGETS)
    assert all(r["cases"] == 3 for r in summary_rows)
    assert all(r["runs"] == 3 * len(POMCP_SEEDS) for r in summary_rows)
    assert all(0.0 <= r["exact_optimal_action_rate"] <= 1.0 for r in summary_rows)
    assert all(0.0 <= r["query_vs_decide_family_accuracy"] <= 1.0 for r in summary_rows)
    assert all(r["max_true_root_action_regret"] >= r["median_true_root_action_regret"] for r in summary_rows)
