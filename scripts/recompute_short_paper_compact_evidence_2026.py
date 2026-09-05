from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(ROOT / "src"))

from procedural_ai.evidence_math import (
    expected_discovery_fraction,
    minimum_units_for_expected_recovery,
    minimum_zero_failure_observations,
    zero_failure_upper_bound,
)
from procedural_ai.hybrid import exact_state_upper_bound

BASE = ROOT / "data" / "processed" / "carriers"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    provenance = json.loads((BASE / "PROVENANCE_SHORT_PAPERS_2026.json").read_text(encoding="utf-8"))
    for item in provenance["files"]:
        path = BASE / item["path"]
        assert path.is_file(), path
        assert digest(path) == item["sha256"], item["path"]

    me = json.loads((BASE / "meccano_carrier_counts.json").read_text(encoding="utf-8"))
    im = json.loads((BASE / "impact_carrier_counts.json").read_text(encoding="utf-8"))

    me_counts = [int(r["recordings"]) for r in me["relations"]]
    im_participant_counts = [int(r["participants"]) for r in im["relations"]]

    assert len(me_counts) == 201
    assert len(im_participant_counts) == 259
    assert sum(c > 1 for c in me_counts) == 162
    assert sum(c > 1 for c in im_participant_counts) == 184

    assert minimum_units_for_expected_recovery(me_counts, total_units=11, target=0.95) == 9
    assert minimum_units_for_expected_recovery(im_participant_counts, total_units=13, target=0.95) == 11
    assert expected_discovery_fraction(me_counts, total_units=11, sampled_units=11) == 1.0

    assert abs(zero_failure_upper_bound(9) - 0.2831288356) < 1e-9
    assert abs(zero_failure_upper_bound(11) - 0.2384041904) < 1e-9
    assert minimum_zero_failure_observations(0.05) == 59

    assert exact_state_upper_bound(10, 8) == 531441

    print("Compact short-paper recomputation: PASS")
    print("MECCANO: 201 relations, 162 survive one-carrier loss, 9/11 for 95% expected recovery")
    print("IMPACT: 259 relations, 184 survive one-carrier loss, 11/13 participants for 95% expected recovery")
    print("Zero-failure bounds: n=9 -> 28.31%, n=11 -> 23.84%, n=59 -> below 5%")
    print("Bellman state envelope: N_upper(10,8)=531441")


if __name__ == "__main__":
    main()
