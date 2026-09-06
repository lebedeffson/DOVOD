# DOVOD Q1 release status — 2026-09-05

## Executive status

Two separate research works now have executable theory, algorithms, tests, benchmark artifacts, and manuscript drafts.

- **Paper A:** method-complete at the recovered-core level; external AMLGym matrix is wired but must run in networked CI; authoritative real procedural labels remain an external scientific dependency.
- **Paper B:** mathematical/exact-planning core is complete at the finite static-world level; internal POMCP validation is present; a pinned real crowd-source calibration benchmark is wired for networked CI. Domain-specific acquisition costs and an independent external POMDP-library cross-check remain external scientific dependencies.

The frozen DOVOD `main` short-paper evidence is preserved separately and not overwritten.

## What is reproducible now

### Paper A

1. Positive-only non-identifiability: a prerequisite that is always true before observed successful executions cannot be proven necessary against a hypothesis class that also contains the model with that prerequisite removed.
2. Finite labelled version-space enumeration and mandatory/excluded/ambiguous predicate classification.
3. Minimum decision-equivalent prerequisite preservation = minimum hitting set.
4. Exact SciPy MILP cross-checked against exhaustive search on small instances.
5. Exact bidirectional contextual repair with exceptions and guards.
6. Exact *soft* bidirectional repair with weighted label-error slack for external data that cannot be perfectly fit by a small frozen vocabulary.
7. Learner-agnostic AMLGym applicability bridge using action-local state templates and a single upstream applicability bit.
8. PAC-Bayes-kl sparse-mask certificate mechanics with a sample-independent product prior.
9. Controlled benchmark: exact recovery of three planted contextual edits; zero error on 4096 independent certification draws; upper risk about 0.00538 at delta=0.05, rho=0.02.
10. AMLGym 1.0.11 contract and 20-domain matrix runner for SAM, OffLAM, NOLAM and ROSAME.

### Paper B

1. Symmetric persistent source-orientation non-identifiability for every finite direct-response sequence.
2. Direct mutual information exactly zero under the theorem assumptions.
3. Closed-form known-truth calibration result: Bayes error `2r(1-r)` and pre-cost gain `2(r-1/2)^2`.
4. Persistent latent reliability implies `Cov(C_i,C_j)=Var(R)` and positive repeated-evidence dependence when reliability is non-degenerate.
5. Integrated finite hidden-world model containing physical state, semantic model, persistent reliability and persistent orientation.
6. Numerical posterior-vector Bellman implementation with disclosed 12-digit cache canonicalization.
7. Independent ordered-history Bellman oracle with no belief rounding and no history merging.
8. Exact evidence-count Bellman DP. Under static conditional-independent query likelihoods, order is irrelevant and the count vector is sufficient.
9. Exact state-count formula `C(h+2Q,h)` for the full evidence-count lattice with `Q` binary repeatable queries and horizon `h`.
10. POMCP-style internal approximate planning baseline validated against exact root action values.
11. Pinned Blue Birds external-source protocol: gold-task hash split, per-worker orientation/reliability calibration, held-out worker validation, and calibration-ranked top-k source selection.

## Fresh local verification

Canonical evidence is stored in JSON rather than copied into prose. As of the current run:

- `python -m pytest -q`: **40 passed**.
- H3, three seeds: count DP = ordered-history oracle in value and first action on all cases.
- Current runtime H3 speedup over the numerical posterior-vector implementation, using three fresh repetitions per solver/case and per-case medians: mean **8.22x**, range **7.79x–8.46x**.
- H3 state compression: **6175 -> 1330** states, ratio 4.642857x.
- H4: 7315 count states.
- H5: 33649 count states.
- H6: 134596 count states; value `0.3922438401236031`.
- Count-DP memoization is per solver instance and stores Bellman values only; full posterior vectors are computed on visit and not retained per state. This removes the cross-instance/high-dimensional posterior-cache memory pathology found during release QA.
- `r=0.9`: six-direct-observation MI = `0`; closed-form calibration gain = `0.32`.
- Paper A synthetic certificate risk upper bound: `0.005383204915233359`.
- Paper B POMCP: 5/5 exact-optimal first actions; max absolute value error `0.017696393853929182`.

Timing is environment-specific. State counts, values, theorem identities and regression equivalence are the stronger reproducible quantities.

## Frozen public empirical evidence reused with provenance

The package includes the public DOVOD v12 evidence snapshot from `lebedeffson/DOVOD@aed82d6...`:

- Paper A short-paper line: 201/272 MECCANO and 259/272 IMPACT direct refutations, held-recording calibration/robustness/rarefaction evidence.
- Paper B short-paper line: 777 episodes, 187 mixed-uncertainty episodes, exact Bellman vs myopic cost evidence and reliability/cost stress results.

Those figures are inherited evidence. Raw third-party media are not redistributed here, so they are not re-described as newly rerun results.

## Historical evidence intentionally retired

The earlier 18–22x H3 speedup belongs to a lost ephemeral runtime. It is provenance only. The current publication candidate must cite only the fresh benchmark artifact unless that historical runtime is independently recovered.

## Remaining external gates before a strong journal submission

### Paper A

- Complete the frozen AMLGym matrix in a networked environment and retain all failed/unsupported cells.
- Analyze domains as statistical units; add repeated trajectory-generation seeds where AMLGym exposes them without test leakage.
- Compare against upstream learners under matched information assumptions and include negative results.
- Obtain authoritative positive and negative procedural admissibility labels for the real-procedure claim; observational counterexamples alone cannot establish positive necessity.
- If effects/problem-solving claims are added, implement and evaluate a PDDL-level repair rather than extrapolating from the current applicability-only decision layer.

### Paper B

- Execute and review the pinned Blue Birds external calibration/source-selection job; retain the exact source commit/blob provenance and the prespecified hash split.
- Measure source acquisition costs and error modes on a domain closer to the target procedural/perception setting; Blue Birds provides real worker reliability/reference items but not deployment-specific costs.
- Add an external POMDP implementation/library cross-check; the current POMCP is intentionally an independent in-package baseline, not an external implementation.
- Validate the exact/count quotient on at least one external sequential evidence dataset where stationary order-exchangeability is empirically defensible.

## Release interpretation

The engineering/research package itself is now coherent and reproducible. The remaining items are not missing local code disguised as theory; they are external-data / third-party-runtime scientific gates. Until they land, use **Q1-scale core / journal working paper**, not **Q1-ready accepted result**.
