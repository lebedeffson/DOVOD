# DOVOD Q1 2026 research package

This branch contains two separate research tracks and their reproducible implementation. The frozen `main` baseline is not modified.

## Paper A — decision-equivalent authorization repair

Working title: **From Sequence Regularity to Action Authorization: Decision-Equivalent and Certified Repair of Learned Preconditions**.

Paper A is a downstream, learner-agnostic repair layer for action applicability. It does **not** infer physical necessity or rewrite learned effects. The core implementation contains positive-only non-identifiability diagnostics, exact hitting-set analysis, contextual prerequisite exceptions and guards, exact/weighted-soft finite-vocabulary MILPs, independent holdout certification, paired tests, and finite-class bounds.

Controlled evidence:

- 128 frozen candidate edits over a 256-state support;
- 640 training observations;
- exact recovery of all three planted local edits;
- complete-support repaired risk `0.0` vs `0.125` for both the upstream rule and the best global-deletion baseline;
- independent holdout: `0/4096` errors, one-sided exact 95% upper risk `0.0007311`;
- 5% label-corruption stress at `n=640`: mean support risk `0.0015625`, maximum `0.0078125`.

Frozen AMLGym V4 confirmatory evidence covers `20 domains x 4 learners x 2 trace budgets = 160` prespecified cells. The protocol replays and excludes historically inspected pilot states by semantic fingerprint, separates repair/calibration/test by a fixed hash, and never uses test labels for fitting, deployment gating, or CI pass/fail. The canonical primary aggregate retains all outcomes: 5 failures/timeouts, 80 empty held-out subsets, and 75 usable cells. Among usable cells DOVOD has `7 wins / 68 ties / 0 losses` versus upstream applicability; domain means give `4 / 6 / 0`, exact two-sided sign-test `p=0.125`. This supports conservative selective deployment, not broad superiority.

A post-freeze seeded reproducibility replay is reported separately rather than substituted for the primary result. It yields 75 usable cells with `6 / 69 / 0` and domain `3 / 7 / 0`, `p=0.25`; the observed hosted-runner loss for `sokoban/ROSAME/10` is narrowly retained as infrastructure missing rather than silently converted into a scientific timeout.

V5 adds reviewer-facing evidence without changing the frozen primary claim:

- a full 160-cell post-confirmatory panel of random and frequency at-most-one-edit baselines; DOVOD beats each simple baseline in `6` usable cells, ties `69`, loses `0`, with domain means `3 / 7 / 0` (`p=0.25`); the simple baselines themselves improve only `2/75` usable cells over upstream;
- an independent Assembly101 ordering audit on 328 sequences / 3,964 annotated events. At predecessor-frequency threshold `0.90`, 18 training relations are mined and `5/18` are contradicted by held-out labeled-correct behavior; relation violations occur in `8/37` covered correct events and `2/5` covered explicit ordering mistakes. This is a falsification/claim-boundary audit: frequent temporal order is not a hard prerequisite and violation alone is not a reliable mistake detector.

Tracked V5 receipts are `results/paper_a_amlgym_simple_baselines_v5.json` and `results/paper_a_assembly101_ordering_audit_v5.json`; the narrative amendment is `docs/PAPER_A_V5_EVIDENCE.md`.

## Paper B — exact evidence-count acquisition under source uncertainty

Working title: **Exact Evidence-Count Dynamic Programming and Cost-Robust Information Acquisition for Static Procedural Decisions**.

Paper B asks which physical, semantic, or calibration evidence to acquire before acting when the hidden world is static during a short local decision. It does not claim a new POMDP paradigm. The implementation contains the persistent orientation construction, calibration result, shared-latent-reliability dependence analysis, exact ordered-history and posterior-vector Bellman references, exact evidence-count DP, a POMCP-style approximation, a source-typed procedural adapter, and a first-action cost-robustness diagnostic.

Current evidence:

- on three 256-world / 9-query / horizon-3 cases, count-DP exactly matches both exact reference representations in root value and first action;
- horizon-3 uses 1,330 count states vs 6,175 ordered histories; horizons 4/5/6 visit exactly `7,315 / 33,649 / 134,596` states, matching the combinatorial formula;
- on the frozen clean CI runner, count-DP is about `7.19x` faster than the posterior-vector Bellman implementation on the three horizon-3 cases; timing is engineering evidence, exact value/action/state identities are primary;
- on 12 independent 512-world cases, 15,000-simulation POMCP selects an exact-optimal root action in `12/12`; mean absolute value error `0.00291`, maximum `0.00645`;
- frozen MECCANO: 777 episodes, 187 mixed-uncertainty episodes; Bellman expected cost `1.6657369` vs myopic `1.7379769`, relative reduction `4.1566%`; only `44.92%` of mixed episodes have positive gain and the top 20% contribute `61.52%` of total positive gain;
- IMPACT PSR official split is a retained negative regime: exact horizon-3 and myopic horizon-1 tie on all `19/19` covered test transitions, while root action changes across the declared cost grid in `19/19`;
- controlled cost uncertainty: the minimax-regret first-action rule reduces worst-case regret in `8/8` acquisition-active cases, mean reduction `0.09093`, maximum `0.10430`;
- Blue Birds source calibration is kept separate from procedural planning: calibration-selected top-5 workers reach `0.90` held-out accuracy vs `0.75` raw majority, bootstrap difference `+0.15 [0.0625, 0.2375]`; naive orientation flipping is negative at `0.70` and is not promoted into a stronger claim.

The intended deployment hierarchy is explicit: exact count-DP for small static ambiguous decisions, approximation when exact enumeration is too large, myopic acquisition when one query is already sufficient, and robust stopping/abstention when plausible query costs cross policy boundaries.

## Reproducibility

From `research/q1_2026`:

```bash
python -m pip install -r requirements.txt
make release
```

Repository CI runs the full Python test suite, reference-result verification, and public-repository integrity checks on Python 3.10, 3.11, and 3.12. AMLGym is intentionally separate because of heavy external learner dependencies and uses `requirements-amlgym.txt`.

Important files:

- `paper_a/` — applicability repair and certification;
- `paper_b/` — source-aware acquisition and exact/approximate planning;
- `papers/PAPER_A_DRAFT.md`, `papers/PAPER_B_DRAFT.md` — separate manuscripts;
- `configs/amlgym_q1_contract.json` — frozen Paper A confirmatory contract;
- `results/` — tracked machine-readable evidence receipts;
- `docs/PAPER_A_V5_EVIDENCE.md` — post-confirmatory V5 evidence boundary.

## Claim boundaries

The two papers are complementary but not merged into one scientific claim. Paper A repairs learned authorization restrictions; Paper B decides which evidence to acquire when residual uncertainty remains. Synthetic mechanism validation, frozen procedural evidence, external benchmark evidence, and post-confirmatory diagnostics are reported separately. No absence-of-counterexamples result is interpreted as proof of physical necessity, no Blue Birds result is treated as procedural Bellman validation, and no post-confirmatory result is used to retrofit the frozen AMLGym significance claim.
