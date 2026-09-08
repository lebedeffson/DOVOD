# Practical-evidence amendment — 2026-09-08

This note records the post-freeze practical strengthening of the two Q1 tracks. It does **not** rewrite frozen evidence, and it does not silently replace negative results. Where this note conflicts with an older approximation sentence in `PAPER_B_DRAFT.md`, this note is the controlling interpretation for the practical-validation branch.

## Paper A — staged audit and carrier coverage

The full carrier-aware audit retains 201/272 MECCANO refutations and 259/272 IMPACT refutations. A carrier is an independent recording for MECCANO and an independent participant for IMPACT. For a relation supported by `k` of `m` carriers, uniformly sampling `s` carriers discovers at least one supporting counterexample with exact probability

`1 - C(m-k,s) / C(m,s)`.

Expected coverage of the full observed refutation set is:

| Dataset | 1 carrier | 2 carriers | 4 carriers | 90% milestone | 95% milestone | leave-one expected coverage |
|---|---:|---:|---:|---:|---:|---:|
| MECCANO, 11 recordings | 70.56% | 77.37% | 83.99% | 7/11 | 9/11 | 98.24% |
| IMPACT, 13 participants | 62.19% | 68.10% | 75.49% | 9/13 | 11/13 | 97.77% |

This supports a staged practical audit: early independent carriers recover a large fraction of the eventually observed counterexamples, while additional carriers continue to improve coverage.

Carrier multiplicity is evidence strength, **not** a safe hard acceptance threshold. The frozen downstream result is 0.90131 recall when one-carrier refutations are retained and 0.78836 when support from at least two carriers is required. Therefore `>=2 carriers` must not be promoted as the deployment rule.

## Paper B — what the new practical tests changed

### 1. The old POMCP success block is only an easy-case smoke test

The older controlled POMCP block reported exact-optimal root recovery on its selected random-prior cases. Those cases are weak evidence for information acquisition because the exact root is frequently immediate `DECIDE`.

A new mechanism-based validation keeps the first 12 balanced-prior cases whose exact horizon-three root is `QUERY`. On each case, POMCP is run at simulation budgets 250, 500, 1000, 2500, 5000, 10000, and 20000 with five fixed seeds. The chosen action is scored by its **true exact Bellman root-action value**, not by POMCP's own value estimate.

At 20,000 simulations:

- exact-optimal root action: 0/60 runs;
- QUERY-vs-DECIDE family accuracy: 0/60;
- mean true root-action regret: 0.02324;
- maximum true root-action regret: 0.05018.

The root execution rule was corrected to choose the maximum estimated mean root value rather than the most visited action. The hard-case failure remained. A bounded exploration-constant calibration selected `c=0.2` on 12 development cases, but on a separate 12-case confirmation block exact root recovery remained 0% and mean regret was 0.02238. POMCP is therefore retained as an approximation baseline, not as the practical default in this local static class.

### 2. Reliability misspecification has an operational failure mode

The exact planner was evaluated while only its assumed source reliability was distorted:

`r_assumed = clip(0.5 + alpha * (r_true - 0.5), 0.5, 1)`.

The complete assumed-model policy is then executed in the true observation model. At `alpha=1` the evaluator reproduces exact policy value. Underconfidence is the dangerous direction in this panel:

- `alpha=0.50`: mean expected query count 0; wrong-decision probability 0.50; mean regret 0.02324;
- `alpha=0.75`: mean expected query count 0.156; wrong-decision probability 0.490; mean regret 0.01905;
- `alpha=1.00`: mean expected query count 1.853; wrong-decision probability 0.406; mean regret approximately 0;
- `alpha=1.25`: mean expected query count 1.976; wrong-decision probability 0.397; mean regret 0.00254.

The practical interpretation is not that overconfidence is good. It is that systematic underestimation of information quality can make the planner stop acquiring evidence too early; calibration error can dominate approximation error.

### 3. Cheap confidence gating by one-step margin failed

A fresh unconditional 48-case panel compared horizon one with exact horizon three. Non-myopic value was positive in 39/48 cases (81.25%), with mean `h1-h3` value gap 0.01607 and maximum 0.04861.

A deterministic triage rule escalated to exact horizon three only when the horizon-one best-vs-second-best root margin was below a fixed threshold. Thresholds up to 0.005 escalated 0/48 cases and captured none of the non-myopic value; thresholds from 0.01 upward escalated all 48. The margin therefore degenerates into a none-vs-all gate in this panel and should not be used as a practical ambiguity detector.

### 4. Myopic top-K query screening is useful but source-family fragile

A first engineering approximation ranks the 10 queries by their exact horizon-one root query values, retains the best `K`, and solves exact horizon-three count-DP on the reduced vocabulary.

On the first 12 acquisition-active development cases, `K=6` gives:

- exact root action 11/12;
- QUERY-vs-DECIDE family 12/12;
- zero full-policy regret 10/12;
- mean full-policy regret 0.000641;
- mean state reduction 3.89x;
- mean wall-clock speedup including screening about 3.54x.

On the next 12 screening-unseen cases it gives 12/12 exact roots and 12/12 zero policy regret. That block had previously been used for a separate POMCP-calibration confirmation, so it is **not** described as a globally untouched holdout.

A fresh 48-case source-cost transfer panel (16 cases each at nominal, physical-expensive, and semantic-expensive costs) also gives zero full-policy regret in all 48 cases with the same 3.89x state reduction. However, that transfer panel contains 30 physical-root acquisitions and 18 immediate decisions, and no semantic-root acquisition.

A dedicated fresh source-family stress test exposes the limitation. Among eight exact physical-root cases, top-6 screening has 8/8 exact root actions and 8/8 zero policy regret. Among eight exact semantic-root cases, the full optimal semantic query ranks only 7th-10th myopically; top-6 contains the optimal query in 0/8 cases, exact root recovery is 0/8, and mean full-policy regret is 0.00578. Increasing K to 8 lowers mean regret to 0.000897 but still yields exact root recovery in only 4/8 cases.

Repeating the ranking with exact horizon-two rather than horizon-one query values does **not** fix the mechanism: on a new 8+8 source-diverse block, the semantic optimal queries retain the same low ranks and h2-top-6 again has 0/8 semantic exact-root recovery.

Thus top-K shallow value screening is a useful physical-root optimization, but it is not source-general.

### 5. Structural typed-core screening is the strongest current practical shortcut

The next screen is semantic rather than value-ranked. The full 10-query controlled vocabulary is:

- `calibrate-physical`;
- `calibrate-semantic`;
- four physical `state-*` queries;
- four semantic `model-feature-*` queries.

The frozen structural screen retains **all eight decision-bearing `state` and `model_feature` queries** and removes only the two explicit calibration queries. Exact horizon-three count-DP is then solved on this eight-query core.

On a completely fresh 32-case unconditional balanced panel:

- exact root action: 32/32;
- zero full-policy regret: 32/32;
- mean state reduction: 1.8277x;
- mean wall-clock speedup: about 1.82x.

A separate fresh source-diverse stress panel requires eight physical-root and eight semantic-root acquisition-active cases. The structural eight-query core gives:

- physical-root: 8/8 exact root, 8/8 zero policy regret;
- semantic-root: 8/8 exact root, 8/8 zero policy regret;
- mean state reduction in both panels: 1.8277x;
- mean wall-clock speedup: about 1.84x physical and 1.86x semantic.

This is currently the most defensible practical shortcut in the controlled static family because it preserves both decision-bearing source families instead of trusting shallow ranking.

It is **not an exactness theorem**. Calibration queries can have real decision value in other workloads, especially when source orientation or reliability must be learned online. The correct deployment rule is therefore: use the typed-core screen only after a local calibration-query ablation shows negligible policy regret; otherwise keep the full exact vocabulary.

## Practical regime-selection rule after the new evidence

1. For small static local problems, use exact evidence-count DP as the reference and preferred solver.
2. If the query vocabulary is repeatedly reused, test a semantics-preserving typed-core reduction offline. In the current controlled family, keeping all `state` and `model_feature` queries while dropping calibration queries is zero-regret in the tested fresh panels and reduces exact state work by about 1.83x.
3. Do not prune queries solely because their horizon-one or horizon-two root value is poor; semantic queries can be non-myopically valuable while ranking 7th-10th shallowly.
4. Do not use the horizon-one root margin as a confidence gate without workload-specific validation.
5. Treat POMCP as an approximation that must be calibrated on acquisition-active exact-solvable cases. The current implementation is not reliable enough on that hard block to replace exact DP.
6. Calibrate source likelihoods separately. Underestimating source informativeness can cause premature `DECIDE` and materially higher terminal error.
7. For Paper A, use independent-carrier coverage as a staged-audit planning tool, but do not turn carrier multiplicity into a hard refutation filter.

## Reproducibility receipts

Focused GitHub Actions runs on `q1/practical-strengthening-20260908` completed successfully for every analysis above. The main consolidated practical run is `34253533901` (artifact digest `sha256:fdd51597aa12c427cd89f4e697901d9ff71fe6dc7941ab8f434d0b02f821c7e1`). Additional fresh-screen receipts are:

- source-cost transfer: run `34254123103`, artifact digest `sha256:1b0b71a7477307c872d25e4a206c8b2509f6cbfd9ef3ce7cff7b1ad5c595ab35`;
- semantic-root stress: run `34254358784`, artifact digest `sha256:4ee114be179b46fdd85e51026645648dc3aba28ad270ed3fec67278d30705ac3`;
- h2 screening confirmation: run `34254617818`, artifact digest `sha256:83d0c6d2c5c5f6150463d741673f3520b9a1750c552e07be1cdba91c45376a06`;
- structural typed-core screen: run `34254915011`, artifact digest `sha256:f87c309093a444273de81c4d3abe314e8fc49a06f668e9e60038009d1392af31`.

The ordinary repository CI also remains green across Python 3.10, 3.11, and 3.12 on the practical branch.
