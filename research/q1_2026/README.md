# DOVOD Q1 research core

This directory is the reproducible research expansion built on top of the frozen DOVOD 2026 short-paper evidence. The current active objective is **Paper A**, which grows the Tver thesis on confusing habitual action order with mandatory prerequisites into a journal-scale study of decision-layer identifiability and contextual repair.

Paper B is preserved as a separate research line and is not used as evidence for Paper A.

## Paper A

Working title: **From Sequence Regularity to Action Authorization: Decision-Equivalent and Certified Repair of Learned Preconditions**.

Implemented and locally verified:

- positive-only prerequisite non-identifiability witness;
- finite version-space utilities;
- exact minimum empirical decision-equivalent prerequisite selection as hitting set;
- brute-force/MILP agreement and optimum-family classification;
- contextual exception and guard edits;
- exact and weighted-soft repair MILPs;
- learner-agnostic AMLGym applicability bridge;
- independent one-sided exact Clopper-Pearson holdout certification;
- exact paired McNemar/binomial baseline comparison;
- secondary PAC-Bayes-kl sparse-mask certificate for a predetermined synthetic vocabulary;
- complete 256-state planted-rule benchmark;
- sample-size and 5% label-corruption stress benchmark;
- frozen MECCANO/IMPACT evidence snapshot from the two-page DOVOD line;
- 2021-2026 action-model-learning / planning-domain-repair literature audit;
- real-problem audit from contemporary procedural-assembly datasets and mistake-analysis work.

The frozen external confirmatory experiment is AMLGym 1.0.11:

`20 domains x 4 learners x 2 trace budgets = 160 cells`.

The runner keeps failures/timeouts, splits predictive states at state level by SHA-256, and reports test decision risk, false allows and false blocks before/after the DOVOD repair layer.

## Paper B

Working title: **Source-Typed Information Acquisition for Procedural Action Decisions under Persistent Source Uncertainty**.

Implemented and locally verified:

- persistent physical/semantic source reliability and honest/inverted orientation modes;
- direct-query non-identifiability and calibration closed forms;
- exact evidence-count DP with an ordered-history oracle and posterior-vector cross-check;
- POMCP-style approximate baseline;
- action-local `ProceduralAcquisitionProblem` adapter from completion probabilities or an arbitrary joint physical-state posterior plus alternative prerequisite sets;
- nonuniform semantic priors, source calibration queries and query-cost-sensitive stop/decide behavior;
- explicit guard against accidental exponential world expansion;
- executable JSON decision interface `scripts/plan_paper_b_query.py`;
- same-marginals/different-correlation witness showing why a joint state posterior can change the optimal information action;
- frozen real-procedure evidence snapshot from 777 MECCANO episodes (187 mixed physical/semantic uncertainty);
- 2025-2026 active-feature-acquisition / Value-of-Information / proactive-assistance related-work audit.

The real-procedure v12 evidence and the Q1 persistent-source theory are intentionally reported as separate evidence layers. The former uses controlled perfect physical reveals; the latter must not be described as empirically calibrated source reliability until real sensor/expert measurements exist.

Reproduce the Paper B algorithmic package:

```bash
python benchmarks/run_paper_b.py
python benchmarks/run_paper_b_practical.py
```

## Reproduce the local Paper A package

```bash
python -m pip install -r requirements.txt
make paper-a-release
```

This runs the complete local test suite, regenerates the controlled benchmark and stress benchmark, and validates the stored invariants.

For AMLGym in an environment with external package access:

```bash
python -m pip install -r requirements-amlgym.txt
make amlgym-matrix
```

## Current local evidence

- 71 tests pass.
- The controlled benchmark recovers all three planted edits from 640 training decisions.
- Complete-support risk: contextual repair `0.0`; best global prerequisite subset `0.125`.
- Untouched 4096-decision holdout: 0 errors; one-sided exact 95% risk upper limit `0.0007311126`.
- Under 5% label corruption, mean complete-support risk in the five-seed stress falls to `0.0015625` at `n=640`.
- Frozen real procedural evidence: 201/272 candidate restrictions refuted in MECCANO TRAIN and 259/272 in IMPACT Reassembly-A; this is falsification evidence, not proof that the remaining restrictions are mechanically necessary.

## Claim boundary

The current Paper A package supports formal claims about observational identifiability, finite decision equivalence, exact optimization within a frozen edit vocabulary, contextual permissiveness containment, and finite-sample applicability error under the declared holdout protocol.

It does **not** claim that demonstrations alone identify mechanical necessity, that repaired actions are physically safe, that action effects are repaired, or that long-horizon plans are valid. Broad cross-domain improvement over modern action-model learners remains gated on the complete frozen AMLGym matrix.
