from __future__ import annotations

from benchmarks.run_paper_b_pomcp_budget_curve import BUDGETS, summarize


def test_budgets_are_strictly_increasing_and_prespecified():
    assert tuple(sorted(BUDGETS)) == BUDGETS
    assert len(set(BUDGETS)) == len(BUDGETS)
    assert BUDGETS[0] > 0


def test_summary_uses_all_cases_and_budgets():
    cases = [
        {
            "rows": [
                {
                    "simulations": budget,
                    "action_is_exact_optimal": (case_id + budget) % 2 == 0,
                    "absolute_value_error": float(case_id + 1) / budget,
                    "seconds": float(budget) / 1000.0,
                }
                for budget in BUDGETS
            ]
        }
        for case_id in range(3)
    ]
    rows = summarize(cases)
    assert [r["simulations"] for r in rows] == list(BUDGETS)
    assert all(r["cases"] == 3 for r in rows)
    assert all(0.0 <= r["exact_optimal_action_rate"] <= 1.0 for r in rows)
    assert all(r["max_absolute_value_error"] >= r["median_absolute_value_error"] for r in rows)
