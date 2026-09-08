from __future__ import annotations

"""Compute-allocation curve for exact non-myopic planning.

A deployment system does not necessarily need horizon-three exact planning on every
local decision.  This benchmark evaluates a deterministic triage rule on the first
48 balanced hidden-world problems, without filtering on the horizon-three outcome.

For every case we first solve the horizon-one problem and compute the root margin
between its best and second-best actions.  If that margin is at most a fixed
threshold, the triage policy escalates to the exact horizon-three count-DP;
otherwise it keeps the horizon-one policy.  Thresholds are fixed in advance and all
are reported.  The horizon-three result is used only for evaluation, never to decide
whether a case escalates.

This is an engineering compute-allocation heuristic.  A large one-step margin is
not a theorem that deeper information synergies are absent, so any missed lookahead
value is retained as regret rather than hidden.
"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_paper_b_scalability_robustness import (
    MODELS,
    cost_sensitive_queries,
    make_balanced_worlds,
    uniform_belief,
)
from paper_b.count_dp import EvidenceCountDP

OUT = ROOT / "results" / "paper_b_adaptive_exact_triage.json"
N_CASES = 48
H1 = 1
H3 = 3
MARGIN_THRESHOLDS = (0.0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05)


def root_margin(action_values: dict[tuple[str, int], float]) -> float:
    values = sorted(float(v) for v in action_values.values())
    if len(values) < 2:
        raise ValueError("at least two root actions are required")
    return float(values[1] - values[0])


def make_case(case_id: int) -> dict:
    seed = 20263000 + int(case_id)
    worlds = make_balanced_worlds(seed)
    belief = uniform_belief(worlds)
    queries = cost_sensitive_queries(1.0, 1.0)

    h1_solver = EvidenceCountDP(belief, worlds, MODELS, queries, horizon=H1)
    h1 = h1_solver.solve()
    h1_values = h1_solver.root_action_values()
    margin = root_margin(h1_values)

    h3_solver = EvidenceCountDP(belief, worlds, MODELS, queries, horizon=H3)
    h3 = h3_solver.solve()

    regret = float(h1.value - h3.value)
    if regret < -1e-10:
        raise AssertionError(f"horizon-one value beat horizon-three value: {regret}")

    return {
        "case": int(case_id),
        "source_seed": int(seed),
        "worlds": len(worlds),
        "queries": len(queries),
        "h1_value": float(h1.value),
        "h3_value": float(h3.value),
        "h1_root_action": list(h1.action),
        "h3_root_action": list(h3.action),
        "h1_root_margin": float(margin),
        "h1_vs_h3_root_action_match": bool(h1.action == h3.action),
        "h1_vs_h3_action_family_match": bool(h1.action[0] == h3.action[0]),
        "lookahead_value_gain": max(0.0, regret),
        "h1_states": int(h1.states),
        "h3_states": int(h3.states),
        "h1_seconds": float(h1.seconds),
        "h3_seconds": float(h3.seconds),
    }


def summarize(cases: list[dict]) -> dict:
    gains = [float(c["lookahead_value_gain"]) for c in cases]
    positive = [g > 1e-12 for g in gains]
    base = {
        "cases": len(cases),
        "cases_with_positive_nonmyopic_value": int(sum(positive)),
        "fraction_with_positive_nonmyopic_value": float(sum(positive) / len(cases)),
        "mean_h1_minus_h3_value": float(statistics.fmean(gains)),
        "max_h1_minus_h3_value": float(max(gains)),
        "h1_root_exact_action_rate": float(
            sum(c["h1_vs_h3_root_action_match"] for c in cases) / len(cases)
        ),
        "h1_query_vs_decide_family_accuracy": float(
            sum(c["h1_vs_h3_action_family_match"] for c in cases) / len(cases)
        ),
    }

    rows = []
    total_positive_gain = sum(gains)
    for threshold in MARGIN_THRESHOLDS:
        escalated = [float(c["h1_root_margin"]) <= threshold + 1e-15 for c in cases]
        residual_regrets = [
            0.0 if esc else float(c["lookahead_value_gain"])
            for c, esc in zip(cases, escalated)
        ]
        missed_positive = [
            (not esc) and float(c["lookahead_value_gain"]) > 1e-12
            for c, esc in zip(cases, escalated)
        ]
        captured_gain = sum(g - r for g, r in zip(gains, residual_regrets))
        rows.append(
            {
                "margin_threshold": float(threshold),
                "exact_h3_escalation_rate": float(sum(escalated) / len(cases)),
                "exact_h3_cases": int(sum(escalated)),
                "non_escalated_cases": int(len(cases) - sum(escalated)),
                "missed_positive_lookahead_cases": int(sum(missed_positive)),
                "mean_residual_regret_vs_always_h3": float(statistics.fmean(residual_regrets)),
                "max_residual_regret_vs_always_h3": float(max(residual_regrets)),
                "fraction_of_total_positive_lookahead_value_captured": (
                    float(captured_gain / total_positive_gain) if total_positive_gain > 0 else 1.0
                ),
            }
        )

    # Reference endpoints make the compute/regret curve explicit.
    rows.append(
        {
            "margin_threshold": "always_exact",
            "exact_h3_escalation_rate": 1.0,
            "exact_h3_cases": len(cases),
            "non_escalated_cases": 0,
            "missed_positive_lookahead_cases": 0,
            "mean_residual_regret_vs_always_h3": 0.0,
            "max_residual_regret_vs_always_h3": 0.0,
            "fraction_of_total_positive_lookahead_value_captured": 1.0,
        }
    )
    return {"unconditional_h1_vs_h3": base, "triage_curve": rows}


def main() -> None:
    cases = [make_case(i) for i in range(N_CASES)]
    report = {
        "schema": "dovod-paper-b-adaptive-exact-triage-v1",
        "case_selection": (
            "first 48 deterministic balanced-prior cases with seeds 20263000..20263047; no filtering on h1/h3 actions or gains"
        ),
        "margin_thresholds": list(MARGIN_THRESHOLDS),
        "cases": cases,
        "summary": summarize(cases),
        "analysis_contract": (
            "Escalation uses only the horizon-one root action margin. Horizon-three values are hidden from the triage rule and used only for evaluation. "
            "All fixed thresholds are reported; no threshold is selected or tuned from the results."
        ),
        "claim_boundary": (
            "This controlled curve measures a compute-allocation heuristic for static balanced worlds. "
            "It does not prove that a one-step margin is a sufficient ambiguity statistic, and exact-h3 case counts are a proxy for solver allocation rather than measured end-to-end service cost."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "claim_boundary": report["claim_boundary"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
