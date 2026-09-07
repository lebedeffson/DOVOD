# Reviewer-hardening evidence map

This note records experiments added after the practical-value review. It separates real procedural evidence, controlled mechanism validation, approximation/scalability checks, and negative diagnostics. No human-time or safety-certification claim is inferred from these results.

## Paper A: counterexample audit before local repair

Primary practical question: **how much of the candidate prerequisite-review queue can be falsified automatically from successful procedural evidence before any local optimizer is invoked?**

| Evidence | MECCANO | IMPACT | Allowed interpretation |
|---|---:|---:|---|
| Directed candidate prerequisite hypotheses | 272 | 272 | Fixed audit universe for the frozen component representation |
| Refuted by >=1 successful counterexample carrier | 201 (73.9%) | 259 (95.2%) | Universal empirical unary prerequisite claim is falsified |
| Refuted by >=2 independent carriers | 162 (59.6%) | 184 (67.6%) | Refutation is not supported by a single carrier only |
| Unordered pairs with both directions refuted at >=1 carrier | 65 / 136 | 123 / 136 | Strong evidence of observed order flexibility; not causal proof |
| Unordered pairs with exactly one direction refuted at >=1 carrier | 71 / 136 | 13 / 136 | One directed universal claim remains unresolved |

The absence of a counterexample does **not** validate necessity. MECCANO and IMPACT component identities are not assumed to match, so relation-level cross-dataset intersections are not computed.

A naive rule that requires at least two independent carriers is not a free robustness improvement. In the frozen MECCANO LORO stress, next-action recall changes from 0.9013 at the one-carrier rule to 0.7884 at the two-carrier rule. The nested held-recording calibration instead changes recall from 0.8707 to 0.8853, gain 0.01454 with bootstrap 95% CI [0.01063, 0.01916] over 110 pairs. This supports a two-stage design: counterexample triage followed by decision-level calibration/repair, rather than a universal hard carrier threshold.

Controlled exact-vs-greedy repair and AMLGym remain secondary evidence. They answer compactness and portability questions, not whether the real procedural problem exists.

Source artifact: `results/paper_a_real_audit.json`.

## Paper B: approximation scale and cost robustness

### Exact / approximate validation

The expanded controlled validation contains 12 independently generated cases with 512 hidden worlds, Q=9 and horizon 3. POMCP selects an exact-optimal root action in 12/12 cases. Mean absolute value error is 0.002912; maximum error is 0.006451. This validates root-action recovery only for this controlled tractable class.

### Large-query engineering stress

At horizon 6, the theoretical count-DP and ordered-history envelopes grow as follows:

| Q | Count-state envelope | Ordered-history envelope | Mean 10k-sim POMCP time in CI |
|---:|---:|---:|---:|
| 12 | 593,775 | 199,411,801 | 0.439 s |
| 18 | 5,245,786 | 2,238,976,117 | 0.624 s |
| 24 | 25,827,165 | 12,490,815,793 | 0.747 s |
| 26 | 40,475,358 | 20,158,268,677 | 0.781 s |

No exact-optimality claim is made for Q>9. In all 12 large-query stress runs, POMCP and the one-step myopic rule choose the same root action. This negative result is retained: a larger query set alone does not make non-myopic planning useful.

### Cost uncertainty

The first unconditional random-prior cost diagnostic was negative: all 8 cases chose immediate DECIDE throughout the tested cost grid because the priors were dominated by inadmissible worlds. It is retained as a failed diagnostic and is not used to claim robustness.

The final cost-robust stress uses a predeclared mechanism selection rule: take the first eight balanced-prior seeds whose exact nominal-cost root action is QUERY. Selection does not inspect robust-policy performance. Nine joint physical/semantic cost scenarios span 0.5x to 10x nominal scale.

- optimal root action changes across the cost grid in 8/8 cases;
- optimal source kind changes in 8/8 cases;
- minimax-regret root selection reduces worst-case first-action regret in 8/8 cases;
- mean reduction is 0.09093; maximum reduction is 0.10430;
- under this deliberately broad uncertainty set the robust first action is DECIDE in all eight cases.

This is a controlled first-action robustness result, not a learned real-world cost distribution and not a full robust-POMDP theorem. It provides an operational fallback when nominal source costs are too uncertain to justify immediate acquisition.

The real procedural evidence remains the frozen 777-episode MECCANO replay: Bellman 1.66574 versus myopic 1.73798 (4.16% reduction; paired 95% interval for Bellman-minus-myopic [-0.07992, -0.06551]). The preserved prevalidation calculation reduces amortized total cost by 14.51% under the frozen model, motivating explicit cost calibration for repeated workloads.

Source artifact: `results/paper_b_scalability_robustness_summary.json`; full row-level output is retained as the GitHub Actions artifact from reviewer-hardening run 34112086144.

## Reproducibility acceptance

At the implementation head used for these experiments, the clean two-paper release runs 41 tests, including regression tests for the reusable minimax-regret selector. Public CI and the reviewer-hardening workflow pass. Quantitative manuscript figures are generated from saved JSON artifacts.
