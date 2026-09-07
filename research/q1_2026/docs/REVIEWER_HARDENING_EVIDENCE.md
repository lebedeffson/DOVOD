# Reviewer-hardening evidence map — V4

This note records experiments added after the practical-value review. It separates real procedural evidence, controlled mechanism validation, post-confirmatory explanatory analyses, scalability checks, cost-robustness diagnostics, and negative results. No human-time, industrial-cost, or safety-certification claim is inferred from these results.

## Paper A — counterexample audit before local repair

Primary practical question: **how much of the candidate prerequisite-review queue can be falsified automatically from successful procedural evidence before any local optimizer is invoked?**

| Evidence | MECCANO | IMPACT | Allowed interpretation |
|---|---:|---:|---|
| Directed candidate prerequisite hypotheses | 272 | 272 | Fixed audit universe for the frozen component representation |
| Refuted by >=1 successful counterexample carrier | 201 (73.9%) | 259 (95.2%) | Universal empirical unary prerequisite claim is falsified |
| Refuted by >=2 independent carriers | 162 (59.6%) | 184 (67.6%) | Refutation is not supported by a single carrier only |
| Unordered pairs with both directions refuted at >=1 carrier | 65 / 136 | 123 / 136 | Strong evidence of observed order flexibility; not causal proof |
| Unordered pairs with exactly one direction refuted at >=1 carrier | 71 / 136 | 13 / 136 | One directed universal claim remains unresolved |

The absence of a counterexample does **not** validate necessity. MECCANO and IMPACT component identities are not assumed to match, so relation-level cross-dataset intersections are not computed.

A naive requirement of at least two independent carriers is not a free robustness improvement. In the frozen MECCANO LORO stress, next-action recall changes from 0.9013 at one carrier to 0.7884 at two carriers. Nested held-recording calibration instead changes recall from 0.8707 to 0.8853, gain 0.01454 with bootstrap 95% CI [0.01063, 0.01916] over 110 pairs.

The operational effect measured here is **queue compression**, not human-time saving: MECCANO reduces 272 directed hypotheses to 71 unresolved hypotheses (3.83x smaller queue), and IMPACT reduces 272 to 13 (20.92x smaller queue). No minutes-per-rule assumption is used.

Source artifact: `results/paper_a_real_audit.json`.

## Paper A — post-confirmatory AMLGym engineering drill-down

The frozen 160-cell confirmatory matrix remains the primary external result. A separate post-confirmatory explanatory study inspects barman, floortile, parking, and tpp, the four domains whose frozen domain-mean DOVOD improvement was positive. It runs all four AMLGym learners and both frozen trace budgets, giving 32 descriptive cells. Domain selection is therefore post hoc and this is **not** a second significance test.

All 32 cells executed successfully and had non-empty test decisions. Across 4,448 held-out decisions:

- upstream learners make 90 applicability errors: 62 false allows and 28 false blocks;
- DOVOD makes 20 errors: 11 false allows and 9 false blocks;
- cell-level comparison is 6 improvements, 26 ties, 0 worsenings;
- mean cell risk reduction is 0.0168341.

The operator-level audit surface contains 216 operator decision surfaces. Only 13 are flagged by observed applicability counterexamples. After calibration gating, 7 receive a **non-empty** contextual repair and 6 are left unresolved/abstained. Empty no-op gate decisions are reported separately and never counted as repairs.

A deliberately coarse non-contextual threshold comparator is fitted only on the repair split: if at least 80% of upstream-blocked repair observations are reference-allowed, all blocks for that operator are globally overridden. In these 32 cells both the raw and calibration-gated threshold policies produce 0 wins, 32 ties, 0 losses against the upstream decisions; aggregate test risk is unchanged. This is a negative baseline and is not presented as a model of human engineering practice.

Representative corrected case: in floortile with NOLAM and trace budget 3, the learned model blocks `(change_color robot1 black black)` while the reference model allows it. The deployed DOVOD repair changes the held-out decision to ALLOW using an exception and a guard over frozen local state features. In that cell the upstream model has 5 false blocks among 140 test decisions (risk 0.0357143); the repaired decision surface has 0 errors.

Other strong descriptive cells include barman/ROSAME/n10, where test risk changes from 0.1631944 to 0.0173611, and tpp/ROSAME/n3, where risk changes from 0.25 to 0.0. Parking is retained as a targeted domain with a tie in this seeded explanatory replay, preventing a success-only narrative.

Source artifact: `results/paper_a_amlgym_practical_summary_v4.json`; full row-level output is GitHub Actions artifact 10016874375 from run 34117056957.

## Paper B — exact / approximate validation and large-query stress

The expanded controlled validation contains 12 independently generated cases with 512 hidden worlds, Q=9 and horizon 3. POMCP selects an exact-optimal root action in 12/12 cases. Mean absolute value error is 0.002912; maximum error is 0.006451.

At horizon 6, the count-DP / ordered-history envelopes and 10k-simulation POMCP times are:

| Q | Count-state envelope | Ordered-history envelope | Mean POMCP time in CI |
|---:|---:|---:|---:|
| 12 | 593,775 | 199,411,801 | 0.439 s |
| 18 | 5,245,786 | 2,238,976,117 | 0.624 s |
| 24 | 25,827,165 | 12,490,815,793 | 0.747 s |
| 26 | 40,475,358 | 20,158,268,677 | 0.781 s |

No exact-optimality claim is made for Q>9. In all 12 large-query stress runs POMCP and the one-step myopic rule choose the same root action. This negative result is retained: a larger query set alone does not make non-myopic planning useful.

## Paper B — cost uncertainty and minimax-regret first action

The initial unconditional random-prior cost diagnostic was negative: all 8 cases chose immediate DECIDE throughout the tested cost grid. It is retained as a failed diagnostic.

The final cost-robust stress uses the first eight balanced-prior seeds whose exact nominal-cost root action is QUERY; selection does not inspect robust-policy performance. Nine joint physical/semantic cost scenarios span 0.5x to 10x nominal scale.

- optimal root action changes across the cost grid in 8/8 cases;
- optimal source kind changes in 8/8 cases;
- minimax-regret root selection reduces worst-case first-action regret in 8/8 cases;
- mean reduction is 0.09093; maximum reduction is 0.10430;
- under this deliberately broad uncertainty set the robust first action is DECIDE in all eight cases.

This is a controlled first-action robustness result, not a learned real-world cost distribution and not a full robust-POMDP theorem.

Source artifact: `results/paper_b_scalability_robustness_summary.json`; full row-level output is retained as the GitHub Actions artifact from reviewer-hardening run 34112086144.

## Paper B — where non-myopic planning helps on MECCANO

The frozen MECCANO replay contains 777 episodes, including 187 mixed physical/semantic uncertainty episodes. The original aggregate at normalized semantic cost 1 is Bellman 1.66574 versus myopic 1.73798, a 4.16% relative reduction. A post-reviewer decomposition explains where this average comes from.

At semantic cost 1 across the 187 mixed episodes:

- 44.92% have positive Bellman-vs-myopic expected-cost gain and 55.08% are exact ties;
- first intervention types differ in 13.37% of episodes;
- mean gain is 0.07224 overall, 0.26322 when first intervention types differ, and 0.04277 when they agree;
- the top 20% of episodes account for 61.52% of total positive gain;
- median gain is 0, 75th percentile 0.12099, maximum 0.38236.

The frozen difficulty indicator `mask_k` separates regimes:

| mask_k | Episodes | Mean gain | Positive-gain fraction | First-intervention disagreement |
|---:|---:|---:|---:|---:|
| 1 | 49 | 0.00000 | 0.0% | 0.0% |
| 2 | 68 | 0.09115 | 35.29% | 35.29% |
| 3 | 70 | 0.10444 | 85.71% | 1.43% |

The effect also depends on query price. At semantic cost 2 mean gain falls to 0.00537 and only 9.63% of episodes have positive gain. At cost 5 all 187 episodes are exact ties. At cost 10 positive gains reappear only sparsely (8.02%; mean 0.00709). The defensible conclusion is selective: non-myopic planning is useful in structurally ambiguous states under some cost regimes, not simply whenever a planner has a longer horizon.

Source artifact: `results/paper_b_meccano_gain_summary_v4.json`; full row-level output is GitHub Actions artifact 10016902244 from run 34117606357.

## Paper B — second real procedural distribution: IMPACT PSR

A second post-reviewer benchmark uses IMPACT v1.1 Procedure Step Recognition annotations pinned to commit `4fed5faa5f05f7aece55712e458defa1f372b248`. The empirical next-step prior is learned only from the official train split and evaluated on the official test split. Frames with simultaneous PSR labels are excluded rather than arbitrarily ordered.

The target is the next singleton PSR step. Three typed queries reveal deterministic annotated properties of the next step: component (`state_idx`), operation (install/remove), and quality (normal/incorrect installation). Costs 0.04/0.03/0.04 are normalized decision-loss units, not measured money or latency. Exact horizon 3 is compared with myopic horizon 1.

Both the prespecified minimum train-support threshold 5 and the predeclared sensitivity threshold 3 produce the same evaluated set: 19 of 25 official test transitions, 76% coverage across 18 test sequences after training on 64 train sequences.

On those 19 covered transitions:

- exact and myopic realized mean cost are both 0.0373684;
- exact is better on 0, tied on 19, worse on 0;
- first actions agree on all 19;
- paired exact-minus-myopic bootstrap interval is [0, 0].

This negative result is retained. In these covered IMPACT transitions, candidate sets are small enough and the first typed query is informative enough that additional lookahead supplies no value. The result complements MECCANO: non-myopic acquisition value is workload-structural, not universal.

Cost sensitivity is nevertheless strong: the optimal root action changes across a uniform 0.5x, 1x, 2x, 5x, 10x grid on 19/19 covered transitions. Representative transition: after `Remove anti_vibration_handle`, true next step is `Remove bearing_screw_topleft`. At 0.5x-5x the exact root action is `QUERY component`; at 10x it becomes `DECIDE` on step 29.

Source artifact: `results/paper_b_impact_psr_summary_v4.json`; full receipts are GitHub Actions artifact 10016729965 from run 34117056957.

## Practical deployment interpretation across the two Paper B datasets

The combined evidence supports a regime decision rather than a universal-solver claim:

1. if the hidden world changes materially during acquisition, the static count-DP model is not appropriate;
2. if one affordable query already resolves the relevant uncertainty, use the myopic policy — IMPACT is a real negative example where deeper lookahead adds nothing;
3. if multiple typed queries interact and ambiguity remains after one query, exact count-DP can be used for small static problems and POMCP for larger ones — MECCANO `mask_k=2/3` episodes characterize where lookahead becomes useful;
4. if query costs are themselves unreliable, use scenario/minimax-regret first-action selection or abstain/DECIDE rather than trusting one nominal price.

This checklist is derived from observed success/failure regimes; it is not a universal optimal threshold in Q or horizon.

## Reproducibility acceptance

Reviewer-hardening code, practical-v3 AMLGym/IMPACT evidence, and the MECCANO concentration analysis run in GitHub Actions against pinned or frozen inputs. Quantitative manuscript figures should be generated only from the saved JSON receipts listed above. The final clean two-paper test count must be read from release CI after these evidence files are committed; do not reuse the earlier 41-test count without verification.
