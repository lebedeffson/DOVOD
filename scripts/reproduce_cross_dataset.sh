#!/usr/bin/env bash
set -euo pipefail

# Cross-dataset analyses require derived MECCANO/IMPACT evidence tables produced
# from datasets obtained under their original terms. They are intentionally not
# bundled as raw third-party data.

required=(
  research/reference_impl/results/hierarchical_counterexample_carriers/meccano_absent_relation_carriers.csv
  research/reference_impl/results/hierarchical_counterexample_carriers/impact_absent_relation_carriers.csv
  research/reference_impl/results/prospective_relation_review/decision_role_refutation_rates.csv
  research/reference_impl/results/prospective_relation_review/recording_level_ranking_metrics.csv
  research/reference_impl/results/impact_v11_external/prospective_relation_risk.csv
)

for f in "${required[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing derived cross-dataset evidence: $f" >&2
    echo "See data/README.md and docs/reproducibility.md." >&2
    exit 2
  fi
done

python experiments/constraints/run_carrier_robustness.py
python experiments/constraints/run_prospective_role_transfer.py
