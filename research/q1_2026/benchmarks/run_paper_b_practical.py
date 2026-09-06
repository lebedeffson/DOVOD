from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_b.procedural_problem import SourceMode, build_prerequisite_acquisition_problem

OUT = ROOT / "results" / "paper_b_practical_source_selection.json"
EVIDENCE = ROOT / "external_evidence" / "dovod_short_papers_v12_full.json"


def run_scenario(name: str, problem, horizon: int, false_allow: float = 1.0, false_block: float = 1.0):
    result = problem.solve_exact(horizon=horizon, false_allow=false_allow, false_block=false_block)
    values = problem.root_action_values(horizon=horizon, false_allow=false_allow, false_block=false_block)
    action_name = problem.queries[result.action[1]].name if result.action[0] == "QUERY" else f"DECIDE:{result.action[1]}"
    return {"scenario": name, "horizon": horizon, "worlds": len(problem.worlds), "queries": [q.name for q in problem.queries], "value": result.value, "action": list(result.action), "action_name": action_name, "root_action_values": {f"{a[0]}:{a[1]}": v for a, v in values.items()}, "states_visited": result.states}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.append(run_scenario("physical_only", build_prerequisite_acquisition_problem((0.0, 0.5), action_index=0, semantic_prerequisites=((1,),), physical_modes=(SourceMode(1.0),), semantic_modes=(SourceMode(1.0),), physical_query_cost=0.05), 1))
    rows.append(run_scenario("semantic_only", build_prerequisite_acquisition_problem((0.0, 0.0), action_index=0, semantic_prerequisites=((), (1,)), physical_modes=(SourceMode(1.0),), semantic_modes=(SourceMode(1.0),), semantic_query_cost=0.05), 1))
    rows.append(run_scenario("query_too_expensive_stop", build_prerequisite_acquisition_problem((0.0, 0.5), action_index=0, semantic_prerequisites=((1,),), physical_modes=(SourceMode(1.0),), semantic_modes=(SourceMode(1.0),), physical_query_cost=0.6), 1))
    independent = build_prerequisite_acquisition_problem((0.0, 0.5, 0.5), action_index=0, semantic_prerequisites=((1, 2),), physical_modes=(SourceMode(1.0),), semantic_modes=(SourceMode(1.0),), physical_query_cost=0.10)
    correlated = build_prerequisite_acquisition_problem((0.0, 0.5, 0.5), action_index=0, semantic_prerequisites=((1, 2),), physical_modes=(SourceMode(1.0),), semantic_modes=(SourceMode(1.0),), physical_query_cost=0.10, physical_state_hypotheses=(((0, 0, 0), 0.5), ((0, 1, 1), 0.5)))
    rows.append(run_scenario("same_marginals_independent", independent, 1))
    rows.append(run_scenario("same_marginals_correlated", correlated, 1))
    rows.append(run_scenario("persistent_orientation_needs_calibration", build_prerequisite_acquisition_problem((0.0, 0.0), action_index=0, semantic_prerequisites=((), (1,)), physical_modes=(SourceMode(1.0),), semantic_modes=(SourceMode(0.9, 1), SourceMode(0.9, -1)), semantic_query_cost=0.01, semantic_calibration_cost=0.01), 2))
    frozen = json.loads(EVIDENCE.read_text(encoding="utf-8"))["B"]
    report = {"schema": "dovod-paper-b-practical-source-selection-v1", "practical_adapter_scenarios": rows, "frozen_real_procedural_evidence": frozen, "checks": {"physical_only_selects_physical": rows[0]["action_name"].startswith("physical:"), "semantic_only_selects_semantic": rows[1]["action_name"].startswith("semantic:"), "expensive_query_stops": rows[2]["action_name"].startswith("DECIDE:"), "independent_same_marginals_stops": rows[3]["action_name"].startswith("DECIDE:"), "correlated_same_marginals_queries": rows[4]["action_name"].startswith("physical:"), "orientation_case_acquires_information": not rows[5]["action_name"].startswith("DECIDE:"), "real_bellman_beats_myopic": frozen["bellman_cost"] < frozen["myopic_cost"]}, "claim_boundary": "The adapter scenarios are algorithmic regression cases, not a substitute for real data. Persistent reliability/orientation parameters remain uncalibrated on real sources."}
    assert all(report["checks"].values()), report["checks"]
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
