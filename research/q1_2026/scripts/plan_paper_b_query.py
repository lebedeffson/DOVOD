from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_b.procedural_problem import SourceMode, build_prerequisite_acquisition_problem


def _modes(rows):
    return tuple(
        SourceMode(
            reliability=float(row["reliability"]),
            orientation=int(row.get("orientation", 1)),
            weight=float(row.get("weight", 1.0)),
        )
        for row in rows
    )


def build_from_json(data: dict):
    hypotheses = data.get("physical_state_hypotheses")
    if hypotheses is not None:
        hypotheses = tuple((tuple(row["state"]), float(row["weight"])) for row in hypotheses)
    return build_prerequisite_acquisition_problem(
        data["completion_probabilities"],
        action_index=int(data["action_index"]),
        semantic_prerequisites=data["semantic_prerequisites"],
        semantic_weights=data.get("semantic_weights"),
        physical_modes=_modes(data.get("physical_modes", [{"reliability": 0.9}])),
        semantic_modes=_modes(data.get("semantic_modes", [{"reliability": 0.9}])),
        physical_query_cost=float(data.get("physical_query_cost", 0.05)),
        semantic_query_cost=float(data.get("semantic_query_cost", 0.10)),
        physical_calibration_cost=data.get("physical_calibration_cost"),
        semantic_calibration_cost=data.get("semantic_calibration_cost"),
        physical_state_hypotheses=hypotheses,
        max_worlds=int(data.get("max_worlds", 65_536)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan the next DOVOD physical/semantic information acquisition action."
    )
    parser.add_argument("input", help="JSON problem file")
    parser.add_argument("--output", help="optional JSON output file")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    problem = build_from_json(data)
    horizon = int(data.get("horizon", 2))
    false_allow = float(data.get("false_allow_cost", 2.0))
    false_block = float(data.get("false_block_cost", 1.0))
    result = problem.solve_exact(
        horizon=horizon, false_allow=false_allow, false_block=false_block
    )
    root = problem.root_action_values(
        horizon=horizon, false_allow=false_allow, false_block=false_block
    )

    if result.action[0] == "QUERY":
        selected = {
            "kind": "QUERY",
            "query_index": result.action[1],
            "query_name": problem.queries[result.action[1]].name,
        }
    else:
        selected = {"kind": "DECIDE", "decision": int(result.action[1])}

    payload = {
        "schema": "dovod-paper-b-query-plan-v1",
        "selected": selected,
        "expected_total_loss": float(result.value),
        "states_visited": int(result.states),
        "world_count": len(problem.worlds),
        "queries": [q.name for q in problem.queries],
        "root_action_values": {
            f"{kind}:{idx}": float(value) for (kind, idx), value in root.items()
        },
        "claim_boundary": (
            "This command plans information acquisition under the supplied probabilistic "
            "model. It does not infer source reliability, procedure truth, or physical safety "
            "unless those inputs have been externally calibrated."
        ),
    }
    text = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
