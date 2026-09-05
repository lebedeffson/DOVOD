# DOVOD Q1 research core

This directory is the reproducible research expansion built on top of the frozen DOVOD 2026 short-paper evidence. The active objective is to turn the Tver DOVOD thesis on confusing habitual action order with mandatory prerequisites into a journal-scale study of decision-layer identifiability and contextual repair.

## Paper A

Working title: **From Sequence Regularity to Action Authorization: Decision-Equivalent and Certified Repair of Learned Preconditions**.

Implemented:

- positive-only prerequisite non-identifiability;
- version-space analysis;
- exact decision-equivalent hitting-set optimization;
- contextual exception/guard repair;
- exact and soft MILP;
- statistical certification;
- AMLGym 1.0.11 frozen protocol.

The external confirmatory matrix is:

`20 domains x 4 learners x 2 trace budgets = 160 cells`.

The runner keeps failures and timeouts visible, prevents state/action leakage by splitting predictive states before action expansion, and reports applicability risk, false allows and false blocks.

## Paper B

Working title: **Source-Typed Information Acquisition for Procedural Action Decisions under Persistent Source Uncertainty**.

Implemented:

- source orientation identifiability theorem;
- calibration value theorem;
- persistent reliability model;
- exact evidence-count DP;
- ordered-history oracle;
- POMCP baseline;
- practical physical/semantic information-selection adapter;
- joint posterior over correlated physical states;
- JSON decision interface.

## Validation status

The local package has 71 tests. Paper B benchmark scripts create their result directory automatically for clean CI execution. The remaining major external Paper A gate is the full AMLGym matrix.

## Claim boundary

The package does not claim that demonstrations alone prove mechanical necessity, that repaired applicability implies effect correctness, or that synthetic source uncertainty equals real sensor calibration. Real external measurements remain separate evidence layers.
