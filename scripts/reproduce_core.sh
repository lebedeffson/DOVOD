#!/usr/bin/env bash
set -euo pipefail

# Core MECCANO + synthetic mechanism rerun. Requires the MECCANO PSR archive,
# but does not require redistribution of IMPACT or other third-party data.
python experiments/constraints/run_nested_calibration.py

python experiments/information_selection/run_bellman_vs_myopic.py
python experiments/information_selection/run_noisy_sources.py
python experiments/information_selection/run_cost_sensitivity.py
python experiments/information_selection/run_exact_scaling.py
python experiments/information_selection/run_reliability_sweep.py

python scripts/verify_reference_results.py
python scripts/check_public_repo.py
