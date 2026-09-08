# Q1 2026 recovery and validation status

## Branch integrity

- Active research branch: `q1/full-rebuild-20260905`.
- Frozen `main` baseline: `aed82d667a4cf058fc32fb2a7fa131bb4b7a3cbb`.
- The research work remains branch-local; `main` is unchanged.

## Release state

The branch now contains a coherent two-paper research release: restored exact core, Paper A controlled/stress/certification code, frozen AMLGym confirmatory machinery, Paper A post-confirmatory reviewer evidence, Paper B exact/approximate acquisition code, real procedural workloads, source-calibration evidence, manuscripts, and tracked machine-readable receipts.

Repository acceptance is the normal `tests` workflow: full `pytest`, reference-result verification, and public-repository integrity checks on Python 3.10, 3.11, and 3.12. Heavy external studies are orchestrated separately and are not rerun merely to make documentation changes.

## Paper A

Paper A is finalized as a decision-layer applicability-repair paper with explicit claim boundaries. The controlled mechanism recovers the three planted contextual edits, achieves zero complete-support risk in the planted case, and has 0/4096 independent holdout errors with a one-sided 95% risk upper bound of about 0.0007311.

The frozen AMLGym V4 primary remains authoritative: 160/160 prespecified cells, 5 retained failures/timeouts, 80 empty held-out subsets, 75 usable cells, `7 wins / 68 ties / 0 losses` versus upstream applicability, domain means `4 / 6 / 0`, exact two-sided sign-test `p=0.125`. This supports selective deployment, not broad learner-wide superiority.

The post-freeze seeded replay remains a reproducibility diagnostic, not a replacement result: 75 usable cells, `6 / 69 / 0`, domain `3 / 7 / 0`, `p=0.25`, with only the observed `sokoban|ROSAME|10` hosted-runner artifact loss narrowly allowlisted as infrastructure missing.

Post-confirmatory practical drill-down remains descriptive: 216 operator surfaces -> 13 counterexample-flagged -> 7 non-empty calibration-approved repairs + 6 abstentions; across 4,448 held-out decisions, errors fall from 90 upstream to 20 under DOVOD. The coarse 80%-threshold global override improves 0/32 selected cells.

V5 reviewer hardening is complete without changing the primary confirmatory claim:

- full 160-cell simple-baseline panel: 75 usable cells; DOVOD vs random one-edit `6/69/0`, DOVOD vs frequency one-edit `6/69/0`, domain `3/7/0` against each (`p=0.25`); each simple baseline improves only 2 usable cells over upstream;
- Assembly101 held-out ordering audit: 328 sequences, 3,964 events. At threshold 0.90, 18 training predecessor relations are mined and 5 are refuted by held-out labeled-correct behavior; relation violations occur in 8/37 covered correct events and 2/5 covered explicit ordering mistakes. Toy-group and threshold-sensitivity analyses preserve the qualitative conclusion.

The Paper A manuscript now includes both V5 blocks and states explicitly that they are post-confirmatory/falsification diagnostics rather than new unbiased significance evidence.

## Paper B

Paper B is finalized as **Exact Evidence-Count Dynamic Programming and Cost-Robust Information Acquisition for Static Procedural Decisions**. The contribution is deliberately bounded to a static binary-evidence class plus a regime-selection deployment story; it does not claim a new POMDP or generic compression theorem.

Validated evidence includes:

- exact count-DP agreement with ordered-history and posterior-vector Bellman references on three 256-world / 9-query / horizon-3 cases;
- 1,330 count states vs 6,175 ordered histories at horizon 3; exact state counts 7,315 / 33,649 / 134,596 at horizons 4/5/6;
- about 7.19x mean count-vs-posterior-vector speedup on the frozen clean runner, treated as machine-dependent engineering evidence;
- POMCP exact-optimal root-action recovery in 12/12 independent 512-world cases, mean absolute value error 0.00291, maximum 0.00645;
- frozen MECCANO 4.1566% expected-cost reduction versus myopic on the declared model, with gain concentrated in ambiguous episodes rather than universal;
- IMPACT PSR negative transfer result: horizon-3 ties horizon-1 on all 19 covered test transitions while all 19 remain cost-policy sensitive;
- minimax-regret first-action reduction in 8/8 acquisition-active controlled cost-grid cases;
- Blue Birds held-out calibration selection 0.90 vs 0.75 raw majority, with naive orientation flipping retained as a negative 0.70 result. Blue Birds is not used as procedural Bellman validation.

## Manuscripts and evidence

- `papers/PAPER_A_DRAFT.md` — authorization repair, certification, frozen AMLGym result, post-confirmatory simple baselines, Assembly101 falsification audit.
- `papers/PAPER_B_DRAFT.md` — source-typed/static acquisition, exact count-DP, approximation, positive/negative procedural regimes, cost robustness, bounded source calibration.
- `docs/PAPER_A_V5_EVIDENCE.md` — explicit V5 provenance/claim-boundary note.
- `results/` — tracked compact evidence receipts plus frozen V4 summaries.

The joint system story is architectural only: learned candidate restrictions -> Paper A repair/certification -> residual uncertainty -> Paper B evidence acquisition -> act/review.
