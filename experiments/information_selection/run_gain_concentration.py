from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "research" / "reference_impl"
if str(REFERENCE) not in sys.path:
    sys.path.insert(0, str(REFERENCE))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from q1_uncertainty_source_aware import build_episodes
from source_aware_resolution_planner import SourceAwareResolutionPlanner, EPS
from run_bellman_vs_myopic import myopic_expected_cost

OUT = ROOT / "results" / "information_selection" / "gain_concentration" / "gain_concentration.json"
SEMANTIC_COSTS = (1.0, 2.0, 5.0, 10.0)


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"q00": None, "q25": None, "q50": None, "q75": None, "q90": None, "q95": None, "q100": None}
    arr = np.asarray(values, dtype=float)
    return {
        "q00": float(np.quantile(arr, 0.00)),
        "q25": float(np.quantile(arr, 0.25)),
        "q50": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "q90": float(np.quantile(arr, 0.90)),
        "q95": float(np.quantile(arr, 0.95)),
        "q100": float(np.quantile(arr, 1.00)),
    }


def _top_share(gains: list[float], fraction: float) -> float | None:
    positive = sorted((max(0.0, x) for x in gains), reverse=True)
    total = sum(positive)
    if total <= 1e-15:
        return None
    k = max(1, int(math.ceil(fraction * len(positive))))
    return float(sum(positive[:k]) / total)


def main() -> None:
    episodes, graphs = build_episodes()
    rows = []
    for episode_id, ep in enumerate(episodes):
        if ep["initial_semantic_variance"] <= EPS:
            continue
        for semantic_cost in SEMANTIC_COSTS:
            planner = SourceAwareResolutionPlanner(
                graphs,
                ep["action"],
                ep["masked_list"],
                physical_cost=1.0,
                semantic_cost=semantic_cost,
            )
            optimal = planner.solve(ep["q"], mode="optimal")
            myopic_cost, myopic_decision = myopic_expected_cost(planner, ep["q"])
            myopic_kind = "RESOLVED" if myopic_decision is None else myopic_decision[0]
            gain = float(myopic_cost - optimal.expected_remaining_cost)
            rows.append({
                "episode": episode_id,
                "recording": ep["recording"],
                "frame": str(ep["frame"]),
                "action": int(ep["action"]),
                "mask_k": int(ep["mask_k"]),
                "semantic_cost": float(semantic_cost),
                "initial_semantic_variance": float(ep["initial_semantic_variance"]),
                "bellman_expected_cost": float(optimal.expected_remaining_cost),
                "myopic_expected_cost": float(myopic_cost),
                "myopic_minus_bellman": gain,
                "bellman_first_intervention": optimal.kind,
                "myopic_first_intervention": myopic_kind,
                "different_first_intervention": bool(optimal.kind != myopic_kind),
            })

    summaries = []
    for semantic_cost in SEMANTIC_COSTS:
        subset = [r for r in rows if r["semantic_cost"] == semantic_cost]
        gains = [r["myopic_minus_bellman"] for r in subset]
        disagreement = [r for r in subset if r["different_first_intervention"]]
        agreement = [r for r in subset if not r["different_first_intervention"]]
        positive = [g for g in gains if g > 1e-12]
        summaries.append({
            "semantic_cost": semantic_cost,
            "episodes": len(subset),
            "recordings": len({r["recording"] for r in subset}),
            "mean_gain": None if not gains else float(np.mean(gains)),
            "gain_quantiles": _quantiles(gains),
            "positive_gain_fraction": 0.0 if not subset else len(positive) / len(subset),
            "exact_tie_fraction": 0.0 if not subset else sum(abs(g) <= 1e-12 for g in gains) / len(subset),
            "first_intervention_disagreement_fraction": 0.0 if not subset else len(disagreement) / len(subset),
            "mean_gain_when_first_intervention_differs": None if not disagreement else float(np.mean([r["myopic_minus_bellman"] for r in disagreement])),
            "mean_gain_when_first_intervention_agrees": None if not agreement else float(np.mean([r["myopic_minus_bellman"] for r in agreement])),
            "top_10pct_share_of_positive_gain": _top_share(gains, 0.10),
            "top_20pct_share_of_positive_gain": _top_share(gains, 0.20),
        })

    mask_strata = []
    for (semantic_cost, mask_k), subset in sorted(
        defaultdict(list, {
            key: [r for r in rows if (r["semantic_cost"], r["mask_k"]) == key]
            for key in {(r["semantic_cost"], r["mask_k"]) for r in rows}
        }).items()
    ):
        gains = [r["myopic_minus_bellman"] for r in subset]
        mask_strata.append({
            "semantic_cost": semantic_cost,
            "mask_k": mask_k,
            "episodes": len(subset),
            "mean_gain": float(np.mean(gains)),
            "positive_gain_fraction": sum(g > 1e-12 for g in gains) / len(gains),
            "first_intervention_disagreement_fraction": sum(r["different_first_intervention"] for r in subset) / len(subset),
        })

    report = {
        "schema": "dovod-paper-b-meccano-gain-concentration-v1",
        "source_experiment": "experiments/information_selection/run_bellman_vs_myopic.py",
        "semantic_costs": list(SEMANTIC_COSTS),
        "mixed_uncertainty_episodes": len({r["episode"] for r in rows}),
        "summaries": summaries,
        "mask_k_strata": mask_strata,
        "rows": rows,
        "claim_boundary": (
            "Post-reviewer explanatory analysis over the existing frozen MECCANO perfect-reveal acquisition model. "
            "It characterizes where Bellman-vs-myopic expected-cost gains concentrate; it does not create new sensor-cost evidence or measured industrial latency. "
            "All semantic-cost settings are reported, including settings where gains are negligible or zero."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
