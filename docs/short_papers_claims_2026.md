# DOVOD short papers 2026 — claim map

The final v12 manuscripts use 19 quantitative evidence groups. `B12` is preserved as secondary evidence but is deliberately not part of the final v12 source-selection thesis.

The compact snapshot is `results/paper_claims_2026/frozen_evidence.json`. It is checked by `python scripts/verify_short_paper_evidence_2026.py` and linked to executable recomputations where third-party raw data are not required.

## Paper A — prerequisite falsification

- **A1/A2:** 201/272 MECCANO and 259/272 IMPACT direct refutations.
- **A3:** nested held-recording authorization recall 0.8707 -> 0.8853 over 110 fit/calibration pairs; mean 3.85 restrictions removed by one calibration recording.
- **A4:** 162/201 and 184/259 refutations survive arbitrary loss of one carrier.
- **A5:** 16 MECCANO and 11 IMPACT refutations occur only in rework contexts.
- **A6:** hard >=2-carrier threshold reduces MECCANO LORO recall 0.9013 -> 0.7884.
- **A7:** 95% expected recovery of the observed refutation set needs 9/11 recordings and 11/13 participants.
- **A8:** zero-failure 95% upper bounds are 28.31% at n=9 and 23.84% at n=11; 59 observations are required for a bound below 5%.

## Paper B — source-aware information acquisition

- **B1:** 777 episodes, 187 mixed-uncertainty episodes.
- **B2/B3:** exact Bellman 1.6657 vs myopic 1.7380 (-4.16%), recording-cluster interval [-0.0799,-0.0655].
- **B4/B5:** fixed baselines 1.7845 / 2.0680; exact and myopic coincide at semantic cost 5.
- **B6/B7:** reliability stress (0.55,0.85) changes source type in 33.15%; semantic-first 6.95% -> 40.11%.
- **B8:** wrong semantic cost gives 24.76% / 44.29% mean relative regret at true costs 5 / 10 when assumed cost is 1.
- **B9:** runtime-only prevalidation 1.466206 -> 1.245739; after charging six one-time normalized units across 777 episodes, total is 1.253461 (-14.51%).
- **B10/B11:** k=10 exact solver uses 157,464..531,414 memoized states; analytical envelope `(g+1)*3^k` gives 531,441 at g=8.
- **B12:** IMPACT AUC 0.5915 -> 0.7462 is retained only as secondary semantic-resolution evidence; it is not an external replication of the Bellman policy and is not in the final v12 two-page thesis.

Raw-data Level-3 reproduction remains separate because MECCANO and IMPACT raw material is not redistributed.
