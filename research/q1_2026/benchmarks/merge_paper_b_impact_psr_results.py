from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED = (3, 5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('artifact_root')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    found = {}
    for path in Path(args.artifact_root).rglob('*.json'):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        if data.get('schema') != 'dovod-paper-b-impact-psr-acquisition-v1':
            continue
        count = int((data.get('coverage') or {}).get('min_train_transition_support', -1))
        if count in EXPECTED:
            if count in found:
                raise SystemExit(f'duplicate min_train_count={count}')
            found[count] = {'path': str(path), 'report': data}
    missing = [x for x in EXPECTED if x not in found]
    report = {
        'schema': 'dovod-paper-b-impact-psr-sensitivity-v1',
        'primary_min_train_count': 5,
        'sensitivity_min_train_count': 3,
        'complete': not missing,
        'missing': missing,
        'results': {str(k): v['report'] for k, v in sorted(found.items())},
        'comparison': {
            str(k): {
                'coverage_fraction': v['report']['coverage']['coverage_fraction'],
                'evaluated_transitions': v['report']['coverage']['evaluated_transitions'],
                'exact_mean_realized_cost': v['report']['comparison']['exact_mean_realized_cost'],
                'myopic_mean_realized_cost': v['report']['comparison']['myopic_mean_realized_cost'],
                'relative_reduction': v['report']['comparison']['relative_reduction'],
                'ci95_exact_minus_myopic': v['report']['comparison']['paired_exact_minus_myopic_bootstrap_ci95'],
                'root_action_disagreement_fraction': v['report']['comparison']['root_action_disagreement_fraction'],
                'exact_better_tied_worse': [
                    v['report']['comparison']['transitions_exact_better'],
                    v['report']['comparison']['transitions_tied'],
                    v['report']['comparison']['transitions_exact_worse'],
                ],
                'top_20pct_share_of_positive_gain': v['report']['comparison']['top_20pct_share_of_positive_gain'],
                'cost_switch_fraction': v['report']['cost_sensitivity']['switch_fraction'],
            }
            for k, v in sorted(found.items())
        },
        'claim_boundary': (
            'The min-train-count=5 analysis is the prespecified primary IMPACT PSR extension. '
            'The min-train-count=3 run is a coverage sensitivity analysis declared before viewing the official test outcomes. '
            'Both are retained regardless of sign or significance; the lower threshold is not used to replace an unfavorable primary result.'
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'results'}, indent=2, sort_keys=True))
    if missing:
        raise SystemExit(3)


if __name__ == '__main__':
    main()
