import json
from pathlib import Path


def test_headline_reference_metrics_are_present():
    p = Path(__file__).resolve().parents[1] / "results" / "reference_metrics.json"
    m = json.loads(p.read_text())
    assert m["constraints"]["candidate_relations"] == 272
    assert m["constraints"]["meccano_refuted_relations"] == 201
    assert m["constraints"]["impact_refuted_relations"] == 259
    assert m["constraints"]["nested_calibrated_recall"] > m["constraints"]["nested_fit9_recall"]
    assert m["information_selection"]["semantic_cost_1_bellman"] < m["information_selection"]["semantic_cost_1_myopic"]
