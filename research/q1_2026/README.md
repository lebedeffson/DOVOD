# DOVOD Q1 research core — two-paper working release

This directory is the reproducible research branch built on top of the frozen DOVOD 2026 short-paper package. It contains two separate journal-scale research lines, executable mathematical cores, practical benchmark adapters, tests, result JSON files, and manuscript drafts.

## Paper A — identifiable decision repair

Working title: **Identifiability and Certified Contextual Repair of Learned Action Models for Decision Support**.

Implemented: positive-trace prerequisite non-identifiability; finite version spaces; exact minimum decision-equivalent hitting set; exhaustive/MILP agreement; bidirectional contextual exceptions and guards; weighted soft repair; PAC-Bayes-kl sparse-mask certificate; controlled planted-edit benchmark; learner-agnostic AMLGym applicability bridge.

## Paper B — identifiability-aware evidence acquisition

Working title: **Identifiability-Aware Bayesian Evidence Acquisition with Persistent Source Reliability and Orientation**.

Implemented: all-finite direct source-orientation non-identifiability; closed-form calibration theorem; persistent latent-reliability dependence; unified static hidden-world model; numerical posterior-vector DP; no-merge ordered-history oracle; exact evidence-count DP; closed-form count-state cardinality; independent POMCP-style baseline.

## Current reproducible evidence

```bash
python -m pytest -q
python benchmarks/run_paper_a.py
python benchmarks/run_paper_b.py
python benchmarks/run_core.py
```

Current local release: **38 tests pass**. Three H3 cases match the no-merge history oracle in value and first action. H3 evidence-count states are 1330 versus 6175 ordered histories. Exact count-state totals are H4=7315, H5=33649, H6=134596. For `r=0.9`, six direct observations give MI=0 under the orientation theorem assumptions and the pre-cost calibration risk gain is 0.32. Current H3 wall-clock speedup over the numerical belief-vector implementation is about 8x in this runtime; timing is environment-specific and is secondary to exact value/action/state-count identities.

Historical 18–22x timing numbers from the lost runtime are not current evidence and must not be used.

## Scientific boundary

This is a strong reproducible Q1-scale core, not a declaration that either manuscript has passed a Q1 journal release gate. Paper A still needs the full external AMLGym/IPC matrix and authoritative real positive labels. Paper B still needs external real source behavior/cost validation and an independent external POMDP implementation cross-check. The frozen `main` evidence is not modified.
