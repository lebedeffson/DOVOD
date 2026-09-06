# DOVOD Q1 research core — two-paper working release

This directory is the reproducible research branch built on top of the frozen DOVOD 2026 short-paper package. It contains **two separate journal-scale research lines**, their executable mathematical core, practical benchmark adapters, frozen empirical provenance, tests, and manuscript drafts.

It is intentionally not the frozen public `main` package.

## Paper A — identifiable decision repair

**Working title:** *Identifiability and Certified Contextual Repair of Learned Action Models for Decision Support*

Implemented here:

- passive-positive prerequisite non-identifiability witness and finite version spaces;
- exact minimum decision-equivalent prerequisite selection as minimum hitting set;
- exact MILP and exhaustive-oracle regression;
- mandatory / optional-optimal / redundant prerequisite classification;
- bidirectional contextual repair: exceptions recover false blocks and guards suppress false allows;
- exact soft MILP for finite vocabularies when external labels cannot be interpolated;
- action-local, object-renaming-invariant contextual features for a learner-agnostic AMLGym applicability layer;
- label-independent vocabulary freezing;
- PAC-Bayes-kl sparse-repair risk certificate under an IID sampling statement and sample-independent prior/vocabulary;
- controlled benchmark with exact recovery of the planted contextual model;
- a frozen AMLGym 1.0.11 / 20-domain / SAM-OffLAM-NOLAM-ROSAME contract and executable matrix runner.

## Paper B — identifiability-aware evidence acquisition

**Working title:** *Identifiability-Aware Bayesian Evidence Acquisition with Persistent Source Reliability and Orientation*

Implemented here:

- all-finite-sequence source-orientation non-identifiability theorem and exact mutual-information check;
- closed-form calibration theorem: one known-truth calibration plus one direct query has Bayes error `2r(1-r)`;
- persistent latent-reliability covariance/correlation identities;
- a single static hidden-world model containing task state, semantic model, source reliability, and source orientation;
- posterior-vector Bellman solver with explicit numerical cache canonicalization;
- independent ordered-history Bellman oracle with no belief rounding and no history merging;
- exact evidence-count sufficient-statistic DP;
- exact count-state combinatorics: for `Q` binary queries and horizon `h`, at most `C(h+2Q, h)` evidence-count states;
- POMCP-style independent approximate tree-search baseline validated against exact optimal-action sets;
- a pinned external Blue Birds crowdsourcing calibration/source-selection protocol with gold labels and individual worker votes.

## Current reproducible evidence

Run:

```bash
python -m pytest -q
python benchmarks/run_paper_a.py
python benchmarks/run_paper_b.py
python benchmarks/run_core.py
python scripts/verify_release.py
```

The current package has 40 unit/regression tests. The benchmark JSON files under `results/` are the canonical local evidence; runtime seconds are machine-dependent and should never be treated as theorem-level quantities.

The current recovered-core run confirms:

- 3/3 H3 exact count-DP matches against the no-merge history oracle in both value and first action;
- H3 median-of-three count-vs-posterior-vector runtime speedup about 8.22x in this runtime (7.79x–8.46x);
- H3 state compression 6175 ordered histories -> 1330 evidence-count states;
- H4/H5/H6 count states exactly 7315 / 33649 / 134596, matching the closed-form combinatorial count;
- the count solver now uses per-instance Bellman memoization without retaining a redundant full posterior vector at every count state, eliminating the prior sequential-run memory pathology;
- orientation model `MI(Y; O_1:6)=0` at `r=0.9` and calibration pre-cost risk gain `0.32`;
- Paper A controlled contextual benchmark: 3 planted edits recovered exactly, zero error on 4096 independent certification draws, PAC-Bayes-kl upper risk about 0.00538;
- Paper B POMCP validation: all five 20k-simulation runs choose an exact-optimal root action on the current oriented test problem; max absolute root-value error below 0.018.

Historical 18–22x timing numbers from the lost runtime are **not current evidence** and are not used by this release.

## Real-data provenance already retained

`external_evidence/dovod_short_papers_v12.json` is a byte-for-value snapshot of the frozen public DOVOD v12 evidence at commit `aed82d667a4cf058fc32fb2a7fa131bb4b7a3cbb`. It carries the already established MECCANO/IMPACT and 777-episode source-aware short-paper evidence, but it is not presented as a fresh raw-data rerun.

## Blue Birds external source-validation gate

`configs/bluebirds_external_contract.json` pins the public Welinder *Blue Birds* crowd-label dataset to commit `fe5ba700...` and to exact ground-truth/label blob SHAs. The protocol freezes a hash-based task split before test outcomes are used, estimates each worker's persistent orientation/reliability from calibration tasks only, and evaluates held-out source-quality association plus calibration-selected top-k source sets against label-free hash-ranked references.

The current local container is offline, so the external files are fetched only in the networked CI job. Until `results/paper_b_bluebirds_external.json` is produced and reviewed, the protocol is an implemented external gate rather than a numerical result. It validates real persistent source behavior, not the procedural-action domain and not the finite-world Bellman planner itself.

## AMLGym external gate

`configs/amlgym_q1_contract.json` freezes AMLGym `1.0.11`, 20 IPC-learning domains, four learner families, two trace budgets, hash-based repair/calibration/test roles, and the DOVOD post-hoc applicability-repair hyperparameters. The runner records failures and unsupported cells rather than silently dropping them.

Local network access is not available in the current execution container, so the third-party package cannot be installed here. The GitHub workflow runs that external matrix in a networked CI environment. Until the complete matrix is produced and reviewed, **no broad AML improvement claim is made**.

## Scientific boundary

This repository is a strong reproducible **Q1-scale core**, not a declaration that either manuscript has already passed a Q1 journal release gate. The remaining empirical gates are explicitly tracked in `RECOVERY_STATUS.md` and the manuscript drafts. No mechanical-safety claim is inferred from observational procedure data, and no source prevalence/reliability claim is inferred from the synthetic hidden-world models.
