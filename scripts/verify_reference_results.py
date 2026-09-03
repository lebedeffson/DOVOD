from __future__ import annotations

import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
m = json.loads((root / "results" / "reference_metrics.json").read_text())

assert m["constraints"]["candidate_relations"] == 272
assert m["constraints"]["meccano_refuted_relations"] == 201
assert m["constraints"]["impact_refuted_relations"] == 259
assert m["constraints"]["nested_calibrated_recall"] > m["constraints"]["nested_fit9_recall"]
assert m["information_selection"]["semantic_cost_1_bellman"] < m["information_selection"]["semantic_cost_1_myopic"]
assert abs(m["information_selection"]["noisy_source_primary_scenario"]["source_change_rate"] - 0.3315) < 0.005

print("Reference snapshot: PASS")
print(json.dumps(m, indent=2))
