# Q1 2026 recovery and validation status

## Branch integrity

- Active research branch: `q1/full-rebuild-20260905`.
- Frozen `main` baseline: `aed82d667a4cf058fc32fb2a7fa131bb4b7a3cbb`.
- `main` is not modified by this work.

## Core release

The final branch assembles the complete two-paper release:

- 36-test suite (35 prior clean-core tests plus the confirmatory-merge regression); clean CI is the acceptance check;
- Paper A controlled benchmark;
- Paper A stress suite;
- Paper A verifier/certificates;
- Paper B exact-orientation regression and POMCP baseline;
- Paper B practical source-selection cases;
- recovered exact count-DP core.

## Paper A

The synthetic mechanism is fully reproducible and certified. The external confirmatory protocol is AMLGym v4: 20 domains, four learner families, two trace budgets, 160 prespecified cells. The label-free preflight passes and proves zero semantic-state overlap with the historically inspected pilot selections before confirmatory test labels are used.

The original sequential ROSAME/n=10 shard exceeded hosted-runner wall time. It was recovered by executing the exact same frozen per-domain cases independently in parallel; no data split, model, gate, hyperparameter, or test-label rule changed. The original frozen per-case limit of 900 seconds is retained. The canonical aggregate contains 160/160 cells and is protocol clean: 5 failures/timeouts, 80 empty-test cells, and 75 usable cells. Among usable cells, 7 improve, 68 tie, and 0 worsen. The domain-level sign summary is 4 wins / 6 ties / 0 losses with exact two-sided p=0.125.

Known reproducible upstream failures are retained rather than repaired post hoc: OffLAM/childsnack fails with `KeyError: kitchen` at both budgets; NOLAM/childsnack fails in upstream PDDL parsing on `(xist ?param_1)` at both budgets. ROSAME/n=10 on sokoban exceeded the prespecified 900-second limit and is retained as a timeout.

The external result therefore supports conservative selective deployment versus the upstream applicability decision; it does not establish broad superiority. Against the calibration-gated global-override baseline, the domain summary is 1 win / 8 ties / 1 loss (p=1.0), so no broad advantage over that baseline is claimed either.

## Paper B

Exact count-DP, exact reference solvers, POMCP-style baseline, practical adapter, and frozen procedural evidence are restored and validated. External Blue Birds source calibration supports held-out reliability-based source selection (0.90 top-5 vs 0.75 raw majority; +0.15 paired bootstrap difference, 95% interval [0.0625, 0.2375]). Naive orientation correction is a retained negative result (0.70) and is not promoted into an empirical orientation claim.

## Manuscripts

Two separate drafts are maintained:

- `papers/PAPER_A_DRAFT.md` — authorization repair, certification, AMLGym confirmatory evidence;
- `papers/PAPER_B_DRAFT.md` — source-typed acquisition, persistent source uncertainty, exact count-DP, procedural and Blue Birds evidence.

The joint system story is only architectural: learned candidate restrictions -> Paper A repair/certification -> residual uncertainty -> Paper B evidence acquisition -> act/review.
