#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -m compileall -q src scripts tests
python -m pytest -q \
  tests/test_short_paper_evidence_math.py \
  tests/test_short_paper_hybrid.py \
  tests/test_short_paper_prevalidation.py
python scripts/recompute_short_paper_compact_evidence_2026.py
python scripts/verify_short_paper_evidence_2026.py
python scripts/verify_reference_results.py

echo 'DOVOD 2026 short-paper release gate: PASS'
echo 'Level-3 raw-data reruns remain separate and require the original third-party datasets.'
