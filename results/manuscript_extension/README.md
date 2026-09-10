# Manuscript extension artifacts — 2026-09-10

This directory contains the browsable summary outputs for the controlled WCIS and SUMMA manuscript-hardening experiments. Full scripts, reports, stress tables and per-repetition SUMMA runs are preserved in `artifacts/manuscript_extension_20260910.tar.gz`.

Archive SHA-256: `5d11b69451411a35b230f7bf3e8aa99ed324476db7071ea2ecf938d0082fe028`.

The archive intentionally excludes the exploratory POMCP-budget experiment because that result was judged unstable and is not used in the manuscripts.

Files in this directory:

- `wcis_correlated_source_summary.csv` — exact full-history vs misspecified count-policy stress under source-error correlation.
- `wcis_state_drift_summary.csv` — exact dynamic-state vs static count-policy stress under hidden-state drift.
- `summa_identifiability_summary.csv` — repair recovery, uniqueness and full-state error over sample size and label noise.
- `summa_identifiability_intervals.csv` — Wilson and bootstrap intervals for the identifiability experiment.
- `summa_false_allow_tradeoff_summary.csv` — false-allow / false-block trade-off under different false-allow penalties.

These are controlled experiments and must not be interpreted as measured live sensor, expert, mechanical-safety or SOP-validation results.
