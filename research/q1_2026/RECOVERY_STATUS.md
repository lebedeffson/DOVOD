# Q1 2026 recovery and validation status

## Branch integrity

- Active research branch: `q1/full-rebuild-20260905`.
- Frozen `main` baseline: `aed82d667a4cf058fc32fb2a7fa131bb4b7a3cbb`.
- `main` is not modified by this work.

## Core release

The final branch assembles the complete two-paper release:

- 38-test suite; clean CI on Python 3.10/3.11/3.12 is the acceptance check;
- Paper A controlled benchmark;
- Paper A stress suite;
- Paper A verifier/certificates;
- Paper B exact-orientation regression and POMCP baseline;
- Paper B practical source-selection cases;
- recovered exact count-DP core.

## Paper A

The synthetic mechanism is fully reproducible and certified. The external confirmatory protocol is AMLGym v4: 20 domains, four learner families, two trace budgets, 160 prespecified cells. The label-free preflight passes and proves zero semantic-state overlap with the historically inspected pilot selections before confirmatory test labels are used.

The canonical primary aggregate contains 160/160 cells and is protocol clean: 5 failures/timeouts, 80 empty-test cells, and 75 usable cells. Among usable cells, 7 improve, 68 tie, and 0 worsen. The domain-level sign summary is 4 wins / 6 ties / 0 losses with exact two-sided p=0.125. The original frozen per-case limit is 900 seconds.

Known reproducible upstream failures are retained rather than repaired post hoc: OffLAM/childsnack fails with `KeyError: kitchen` at both budgets; NOLAM/childsnack fails in upstream PDDL parsing on `(xist ?param_1)` at both budgets. In the primary frozen run, ROSAME/n=10 on sokoban exceeded the prespecified 900-second limit and is retained as a timeout.

A post-freeze seeded reproducibility replay was run after the primary result was frozen. It pins `PYTHONHASHSEED=0`, Python/NumPy seed `20260906`, and the same PyTorch seed for ROSAME without changing domains, budgets, semantic split, repair vocabulary, deployment gate, metrics, or the 900-second scientific limit. It produced 159 per-case artifacts. The hosted GitHub runner executing `sokoban/ROSAME/10` received a shutdown signal after about 751 seconds, before the 900-second scientific timeout, so this replay records that cell as `infrastructure_missing`, not as a scientific timeout. With that explicit non-scientific failure retained, the seeded diagnostic has 75 usable cells: 6 improve, 69 tie, and 0 worsen; domain-level 3 wins / 7 ties / 0 losses, p=0.25.

A repeated seeded `barman/ROSAME/10` case reproduces every scientific field exactly. The only differences between the two JSON records are `learn_seconds` and `wall_seconds`, confirming deterministic scientific output under the amendment while correctly leaving runtime measurements environment-dependent.

Infrastructure accounting is narrowly allowlisted to the actually observed `sokoban|ROSAME|10` hosted-runner artifact loss. The merger refuses to synthesize any other absent confirmatory cell, so an unrelated future execution bug cannot silently pass as infrastructure.

The external result therefore supports conservative selective deployment versus the upstream applicability decision; it does not establish broad superiority. Against the calibration-gated global-override baseline, no broad advantage is claimed.

## Paper B

Exact count-DP, exact reference solvers, POMCP-style baseline, practical adapter, and frozen procedural evidence are restored and validated. External Blue Birds source calibration supports held-out reliability-based source selection (0.90 top-5 vs 0.75 raw majority; +0.15 paired bootstrap difference, 95% interval [0.0625, 0.2375]). Naive orientation correction is a retained negative result (0.70) and is not promoted into an empirical orientation claim.

The final clean CI timing reports a mean count-vs-posterior-vector speedup of about 6.33x. This is runtime-specific engineering evidence, not an asymptotic theorem.

## Manuscripts

Two separate drafts are maintained:

- `papers/PAPER_A_DRAFT.md` — authorization repair, certification, AMLGym confirmatory evidence;
- `papers/PAPER_B_DRAFT.md` — source-typed acquisition, persistent source uncertainty, exact count-DP, procedural and Blue Birds evidence.

The joint system story is only architectural: learned candidate restrictions -> Paper A repair/certification -> residual uncertainty -> Paper B evidence acquisition -> act/review.
