from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_a.protocol import (
    confirmatory_states_excluding_pilot,
    pilot_index_ranked_states,
    state_split,
)


def main() -> None:
    from amlgym.benchmarks import get_problems_path, get_test_states

    contract = json.loads((ROOT / "configs" / "amlgym_q1_contract.json").read_text())
    sampling = contract["predictive_sampling"]
    report = {"schema": "dovod-q1-amlgym-confirmatory-protocol-preflight-v1", "domains": []}

    for domain in contract["domains"]:
        all_states = get_test_states(domain, kind="predictive_power")
        problem_paths = get_problems_path(domain, kind="predictive_power")[: sampling["problems_per_domain"]]
        pilot_by_problem = {}
        all_pilot_fingerprints = set()
        for problem_path in problem_paths:
            problem_name = Path(problem_path).name
            pilot = pilot_index_ranked_states(
                all_states[problem_name],
                domain=domain,
                problem_name=problem_name,
                limit=sampling["pilot_states_per_problem_reserved"],
            )
            pilot_by_problem[problem_name] = pilot
            all_pilot_fingerprints.update(x.fingerprint for x in pilot)

        problems = []
        all_confirmatory = set()
        for problem_path in problem_paths:
            problem_name = Path(problem_path).name
            states = all_states[problem_name]
            confirmatory = confirmatory_states_excluding_pilot(
                states,
                domain=domain,
                problem_name=problem_name,
                states_per_problem=sampling["confirmatory_states_per_problem"],
                pilot_states_per_problem=sampling["pilot_states_per_problem_reserved"],
                excluded_fingerprints=all_pilot_fingerprints,
            )
            confirmatory_fingerprints = {x.fingerprint for x in confirmatory}
            overlap = confirmatory_fingerprints.intersection(all_pilot_fingerprints)
            if overlap:
                raise AssertionError(f"{domain}/{problem_name}: pilot overlap={len(overlap)}")
            cross_confirmatory_duplicates = confirmatory_fingerprints.intersection(all_confirmatory)
            if cross_confirmatory_duplicates:
                raise AssertionError(
                    f"{domain}/{problem_name}: duplicate confirmatory states across problems={len(cross_confirmatory_duplicates)}"
                )
            all_confirmatory.update(confirmatory_fingerprints)
            split_counts = {"repair": 0, "calibration": 0, "test": 0}
            for item in confirmatory:
                split_counts[state_split(
                    domain=domain,
                    problem_name=problem_name,
                    fingerprint=item.fingerprint,
                )] += 1
            problems.append({
                "problem": problem_name,
                "available_state_rows": len(states),
                "historical_pilot_rows_replayed": len(pilot_by_problem[problem_name]),
                "confirmatory_unique_states": len(confirmatory),
                "split_counts": split_counts,
            })
        report["domains"].append({
            "domain": domain,
            "historical_pilot_unique_fingerprints": len(all_pilot_fingerprints),
            "confirmatory_unique_fingerprints": len(all_confirmatory),
            "overlap": 0,
            "problems": problems,
        })

    report["all_domains_clean"] = all(row["overlap"] == 0 for row in report["domains"])
    out = ROOT / "artifacts" / "confirmatory-protocol-preflight.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "schema": report["schema"],
        "domain_count": len(report["domains"]),
        "all_domains_clean": report["all_domains_clean"],
        "minimum_confirmatory_states_per_domain": min(
            row["confirmatory_unique_fingerprints"] for row in report["domains"]
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
