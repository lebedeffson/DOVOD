from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from random import Random
from statistics import median
from time import perf_counter

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_b.count_dp import (
    EvidenceCountDP,
    evidence_count_state_count,
    ordered_history_state_count,
)
from paper_b.exact_dp import solve_exact_belief_dp
from paper_b.history_dp import OrderedHistoryDP
from paper_b.orientation import calibration_risk_gain, direct_mutual_information
from paper_b.static_world import Query, cartesian_worlds

OUT = ROOT / "results" / "recovered_core_benchmark.json"


def make_case(seed: int):
    models = ((0,), (0, 1), (0, 2), (0, 1, 3))
    worlds = cartesian_worlds(
        state_bits=4,
        models=models,
        physical_reliabilities=(0.58, 0.93),
        semantic_reliabilities=(0.72, 0.97),
    )
    rng = Random(seed)
    p_one = 0.76 + 0.015 * seed
    b = []
    for w in worlds:
        ps = 1.0
        for x in w.state:
            ps *= p_one if x else (1 - p_one)
        b.append(ps * (1.0 + 0.08 * rng.random() + 0.08 * int(w.model == 3)))
    z = sum(b)
    b = tuple(x / z for x in b)
    queries = tuple(
        [Query(f"state-{j}", "state", j, 0.010 + 0.002 * j) for j in range(4)]
        + [
            Query(f"model-{j}", "model_feature", j, 0.012 + 0.002 * j)
            for j in range(1, 4)
        ]
        + [
            Query("cal-physical", "calibrate_physical", 0, 0.003),
            Query("cal-semantic", "calibrate_semantic", 0, 0.0035),
        ]
    )
    return worlds, b, models, queries


def main() -> None:
    rows = []
    for seed in range(3):
        worlds, b, models, queries = make_case(seed)
        belief_times = []
        belief_runs = []
        for _ in range(3):
            t0 = perf_counter()
            br, bc = solve_exact_belief_dp(
                b, worlds, models, queries, horizon=3, canonical_ndigits=12
            )
            belief_times.append(perf_counter() - t0)
            cache_states = bc.cache_info().currsize
            belief_runs.append((br, cache_states))
            bc.cache_clear()
        belief = belief_runs[0][0]
        if any(abs(run[0][0] - belief[0]) > 1e-14 or run[0][1] != belief[1] for run in belief_runs):
            raise AssertionError("numerical belief DP timing repetitions disagree")
        belief_s = median(belief_times)
        belief_cache_states = belief_runs[0][1]

        history = OrderedHistoryDP(b, worlds, models, queries, horizon=3).solve()

        count_times = []
        count_runs = []
        for _ in range(3):
            solver = EvidenceCountDP(b, worlds, models, queries, horizon=3)
            t0 = perf_counter()
            cr = solver.solve()
            count_times.append(perf_counter() - t0)
            count_runs.append(cr)
        count = count_runs[0]
        if any(abs(run.value - count.value) > 1e-14 or run.action != count.action or run.states != count.states for run in count_runs):
            raise AssertionError("count DP timing repetitions disagree")
        count_s = median(count_times)
        rows.append(
            {
                "seed": seed,
                "worlds": len(worlds),
                "queries": len(queries),
                "horizon": 3,
                "numerical_belief_dp_value": belief[0],
                "numerical_belief_dp_action": list(belief[1]),
                "numerical_belief_dp_seconds": belief_s,
                "numerical_belief_dp_seconds_repetitions": belief_times,
                "numerical_belief_dp_cache_states": belief_cache_states,
                "history_dp_value": history.value,
                "history_dp_action": list(history.action),
                "history_dp_seconds": history.seconds,
                "history_dp_states": history.states,
                "count_dp_value": count.value,
                "count_dp_action": list(count.action),
                "count_dp_seconds": count_s,
                "count_dp_seconds_repetitions": count_times,
                "count_dp_states": count.states,
                "belief_value_abs_error": abs(belief[0] - count.value),
                "history_value_abs_error": abs(history.value - count.value),
                "belief_action_match": belief[1] == count.action,
                "history_action_match": history.action == count.action,
                "count_vs_belief_speedup": belief_s / count_s,
                "history_to_count_state_ratio": history.states / count.states,
                "theoretical_history_states": ordered_history_state_count(len(queries), 3),
                "theoretical_count_states": evidence_count_state_count(len(queries), 3),
                "history_state_formula_match": history.states == ordered_history_state_count(len(queries), 3),
                "count_state_formula_match": count.states == evidence_count_state_count(len(queries), 3),
            }
        )

    deeper = []
    worlds, b, models, queries = make_case(0)
    for h in (4, 5, 6):
        count = EvidenceCountDP(b, worlds, models, queries, horizon=h).solve()
        deeper.append(
            {
                "horizon": h,
                "value": count.value,
                "action": list(count.action),
                "seconds": count.seconds,
                "states": count.states,
                "query_evaluations": count.query_evaluations,
                "posterior_evaluations": count.posterior_evaluations,
                "theoretical_states": evidence_count_state_count(len(queries), h),
                "state_formula_match": count.states == evidence_count_state_count(len(queries), h),
            }
        )

    report = {
        "schema": "dovod-q1-recovered-core-v2",
        "timing_protocol": {
            "h3_repetitions_per_solver_case": 3,
            "h3_summary_statistic": "median wall-clock seconds per solver/case",
            "deeper_count_dp_repetitions": 1,
            "timing_claim": "engineering evidence only; exact value/action/state-count identities are primary",
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "summary": {
            "all_h3_count_matches_numerical_belief_value": all(
                r["belief_value_abs_error"] < 1e-8 for r in rows
            ),
            "all_h3_count_matches_history_value": all(
                r["history_value_abs_error"] < 1e-10 for r in rows
            ),
            "all_h3_actions_match_numerical_belief": all(
                r["belief_action_match"] for r in rows
            ),
            "all_h3_actions_match_history": all(r["history_action_match"] for r in rows),
            "all_h3_history_state_formula_matches": all(r["history_state_formula_match"] for r in rows),
            "all_h3_count_state_formula_matches": all(r["count_state_formula_match"] for r in rows),
            "mean_h3_count_vs_belief_speedup": sum(
                r["count_vs_belief_speedup"] for r in rows
            )
            / len(rows),
            "min_h3_count_vs_belief_speedup": min(
                r["count_vs_belief_speedup"] for r in rows
            ),
            "max_h3_count_vs_belief_speedup": max(
                r["count_vs_belief_speedup"] for r in rows
            ),
            "mean_h3_history_to_count_state_ratio": sum(
                r["history_to_count_state_ratio"] for r in rows
            )
            / len(rows),
            "orientation_direct_mi_k6_r09": direct_mutual_information(6, 0.9),
            "orientation_calibration_gain_r09": calibration_risk_gain(0.9),
        },
        "h3_rows": rows,
        "deeper_count_dp": deeper,
        "claim_boundary": (
            "Reconstructed finite static-world synthetic benchmark. OrderedHistoryDP is the no-rounding/no-history-merge oracle. "
            "The posterior-vector solver uses 12-digit cache canonicalization and is reported as a numerical Bellman implementation. "
            "Timing is runtime-specific and is not external Q1 evidence."
        ),
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
