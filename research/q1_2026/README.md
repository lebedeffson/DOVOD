# DOVOD Q1 2026 research package

This branch contains two separate research tracks plus their reproducible implementation. The frozen `main` baseline is not modified.

## Paper A — decision-equivalent authorization repair

Working title: **From Sequence Regularity to Action Authorization: Decision-Equivalent and Certified Repair of Learned Preconditions**.

The paper asks which learned ordering restrictions should remain in the action-authorization surface. The implementation contains positive-only identifiability diagnostics, exact hitting-set analysis, contextual exception/guard repair, exact and soft finite-vocabulary MILPs, independent holdout certification, paired tests, finite-class bounds, and an external AMLGym confirmatory protocol.

Controlled benchmark facts:

- 128 frozen candidate edits over a 256-state support;
- 640 training observations;
- exact recovery of all three planted local edits;
- complete-support repaired risk 0.0 vs 0.125 for both the upstream rule and the best global-deletion baseline;
- independent holdout: 0/4096 errors, one-sided exact 95% upper risk 0.0007311;
- paired exact p-value vs upstream: 3.73e-155;
- stress at n=640 with 5% label corruption: mean support risk 0.0015625, max 0.0078125.

External AMLGym validation uses a frozen v4 confirmatory design over 20 domains x 4 learner families x 2 trace budgets = 160 prespecified cells. Pilot states are replayed exactly and excluded by semantic fingerprint before confirmatory ranking. Repair, calibration, and test are split by a fixed SHA-256 bucket; test labels do not affect fitting or deployment gating. Failed learner/tool cells and empty held-out test subsets are retained rather than silently removed.

The canonical frozen full-matrix result is stored in `results/paper_a_amlgym_confirmatory_matrix.json`: all 160 cells are present and the protocol is clean. 5 cells are retained as failures/timeouts; 80 successful cells have empty held-out test subsets; among 75 usable cells, 7 improve, 68 tie, and 0 worsen. Domain-level mean risk has 4 wins, 6 ties, and 0 losses (exact two-sided sign-test p=0.125). This is evidence for conservative selective deployment, not broad superiority across AMLGym.

A later clean replay exposed upstream learner-process nondeterminism because the primary run did not pin Python hash order and common Python/NumPy/PyTorch RNG seeds. The primary result above is not retroactively replaced. A post-freeze reproducibility amendment now pins `PYTHONHASHSEED=0`, Python/NumPy seed `20260906`, and the same PyTorch seed for ROSAME while preserving the frozen domains, budgets, semantic split, repair vocabulary, deployment gate, metrics, and 900-second per-case limit. Its aggregate is treated as a separate reproducibility diagnostic. In that seeded replay, 159 per-case artifacts were produced and the hosted runner for `sokoban/ROSAME/10` shut down before the frozen 900-second scientific timeout; that cell is retained explicitly as an infrastructure-missing outcome. The 75 usable seeded cells contain 6 improvements, 69 ties, and 0 worsenings, with domain-level 3 wins / 7 ties / 0 losses (p=0.25). A repeated seeded `barman/ROSAME/10` run reproduces all scientific fields exactly; only wall-clock timing fields differ.

Infrastructure accounting is narrowly allowlisted: only the observed `sokoban|ROSAME|10` artifact loss may be synthesized as `infrastructure_missing`; any other missing confirmatory cell remains fatal to the merge.

### Practical carrier coverage

The carrier-aware real-data audit now has an explicit staged-audit analysis. MECCANO has 201 observed refutations across 11 independent recordings; IMPACT has 259 across 13 independent participants. Exact hypergeometric coverage shows that the expected observed-refutation coverage reaches 90% after 7/11 MECCANO recordings and 9/13 IMPACT participants, and 95% after 9/11 and 11/13 respectively. Random leave-one expected coverage is 98.24% for MECCANO and 97.77% for IMPACT.

Carrier multiplicity is **not** a safe hard filter. Requiring support from at least two carriers reduces the frozen downstream recall from 0.90131 to 0.78836. The practical use of carrier counts is evidence-strength reporting and staged audit planning, not automatic deletion of one-carrier counterexamples.

## Paper B — source-typed evidence acquisition

Working title: **Source-Typed Information Acquisition for Procedural Action Decisions under Persistent Source Uncertainty**.

The paper asks which evidence source should be acquired before acting when physical state, semantic rule uncertainty, and persistent source behavior are jointly uncertain. The implementation contains the orientation non-identifiability construction, calibration result, latent-reliability correlation analysis, exact static-world Bellman solvers, exact evidence-count DP, ordered-history and posterior-vector cross-checks, a POMCP-style baseline, and a procedural prerequisite-acquisition adapter.

Controlled/recovered-core facts:

- count-DP exactly matches value and first action of both exact reference representations on all three 256-world / 9-query horizon-3 cases;
- history-to-count state ratio: 4.642857;
- mean count-vs-posterior-vector wall-clock speedup on the final clean CI runner: about 6.33x (runtime-specific engineering evidence);
- horizons 4/5/6 visit exactly 7,315 / 33,649 / 134,596 count states, matching the combinatorial formula.

The older random-prior POMCP success block is retained only as a smoke test. A harder validation keeps 12 balanced-prior cases whose exact horizon-3 root is `QUERY`, runs seven simulation budgets and five POMCP seeds, and scores the selected root action by its true exact Bellman value. At 20,000 simulations the current POMCP baseline recovers an exact-optimal root in 0/60 runs, matches the QUERY/DECIDE family in 0/60, and has mean true root-action regret about 0.02324. Exploration-constant tuning does not rescue the held-out hard block. Approximate planning is therefore not the recommended default for this local exact-solvable regime.

A full-policy likelihood-misspecification stress test shows why source calibration matters operationally. With reliability shrinkage `alpha=0.5`, the exact planner under its wrong model buys zero queries on average, has 0.50 wrong-decision probability, and incurs mean regret about 0.02324. At the correct model (`alpha=1`) mean expected query count is about 1.85 and policy regret is numerically zero. The tested underconfidence direction therefore causes premature stopping rather than merely noisier value estimates.

Cheap ambiguity gating also fails in the controlled panel: among 48 fresh balanced cases, horizon-3 non-myopic value is positive in 39/48. A horizon-1 best-vs-second-best margin threshold either escalates none of the cases (through 0.005) or all of them (from 0.01), so that margin is not used as a practical confidence gate.

### Practical exact-screening result

Shallow top-K query-value screening is useful but source-family fragile. A top-6 horizon-1 screen gives low regret and about 3.89x state reduction on physical-root acquisition cases, but a dedicated semantic-root stress shows 0/8 exact-root recovery because the truly optimal semantic queries rank only 7th-10th shallowly. Replacing horizon-1 ranking with horizon-2 ranking does not fix this mechanism.

The strongest current shortcut is instead a semantics-preserving **typed core**: keep all four physical `state-*` queries and all four semantic `model-feature-*` queries, and remove only `calibrate-physical` and `calibrate-semantic` before exact horizon-3 count-DP. On a fresh 32-case unconditional panel this eight-query core has 32/32 exact roots and 32/32 zero full-policy regret. On a separate source-diverse stress panel it has 8/8 zero-regret physical-root cases and 8/8 zero-regret semantic-root cases. Mean exact state reduction is 1.8277x and mean wall-clock speedup is about 1.82-1.86x on the CI runner.

This typed-core result is an engineering ablation, not an exactness theorem. Calibration queries can be decision-relevant in other workloads. Deployment should therefore use the eight-query core only after a workload-specific calibration-query ablation confirms negligible policy regret; otherwise use the full exact vocabulary.

Frozen procedural evidence contains 777 MECCANO episodes (187 mixed-source episodes): Bellman expected cost 1.6657369 vs 1.7379769 for the myopic comparator, relative reduction 4.1566%, with paired 95% interval for Bellman-minus-myopic cost [-0.0799234, -0.0655095]. This experiment uses controlled perfect reveals and does not empirically identify persistent source orientation.

A separate Blue Birds held-out source-calibration experiment supports the narrower reliability-selection claim: top-5 calibration-selected sources obtain 0.90 test accuracy vs 0.75 for raw majority voting, paired bootstrap difference +0.15 with 95% interval [0.0625, 0.2375]. Naive orientation flipping reduces majority accuracy to 0.70; this negative result is retained and the stronger orientation heuristic is not claimed.

The detailed practical amendment, including negative screening results, holdout boundaries, exact run IDs, and artifact digests, is in `papers/PRACTICAL_EVIDENCE_AMENDMENT_20260908.md`.

## Reproducibility

From `research/q1_2026`:

```bash
python -m pip install -r requirements.txt
make release
```

The ordinary repository CI remains green across Python 3.10, 3.11, and 3.12 on the practical branch. Focused GitHub Actions workflows separately reproduce the carrier-coverage analysis, acquisition-active POMCP budget curve, likelihood-misspecification evaluator, bounded POMCP calibration, shallow screening, source-cost transfer, semantic-root stress, h2-screen confirmation, and structural typed-core screen. AMLGym requires `requirements-amlgym.txt` and is run separately because of its heavy external learner dependencies.

Important files:

- `paper_a/` — decision-layer repair and certification;
- `paper_b/` — source-aware acquisition and exact/approximate planning;
- `benchmarks/run_paper_a.py`, `run_paper_a_stress.py`;
- `benchmarks/run_paper_a_carrier_coverage_curve.py`;
- `benchmarks/run_paper_b.py`, `run_paper_b_practical.py`;
- `benchmarks/run_paper_b_pomcp_budget_curve.py`;
- `benchmarks/run_paper_b_likelihood_misspecification.py`;
- `benchmarks/run_paper_b_screened_exact.py`;
- `benchmarks/run_paper_b_semantic_root_screen_stress.py`;
- `benchmarks/run_paper_b_h2_screen_confirmation.py`;
- `benchmarks/run_paper_b_structural_core_screen.py`;
- `benchmarks/run_core.py`;
- `benchmarks/run_amlgym_confirmatory_case.py`;
- `benchmarks/merge_amlgym_confirmatory_results.py`;
- `configs/amlgym_q1_contract.json`;
- `papers/PAPER_A_DRAFT.md`;
- `papers/PAPER_B_DRAFT.md`;
- `papers/PRACTICAL_EVIDENCE_AMENDMENT_20260908.md`.

## Claim boundaries

The two papers are complementary but not merged. Paper A repairs learned authorization restrictions. Paper B decides which evidence source to acquire when uncertainty remains. Synthetic mechanism validation, frozen procedural evidence, and external benchmark evidence are reported separately. No absence-of-counterexamples result is interpreted as proof of physical necessity, no external improvement is generalized beyond the statistical unit supported by the confirmatory data, and no query-screening result is promoted to an exactness theorem outside the tested workload class.
