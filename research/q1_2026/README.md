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

The canonical full-matrix result is stored in `results/paper_a_amlgym_confirmatory_matrix.json`: all 160 cells are present and the protocol is clean. 5 cells are retained as failures/timeouts; 80 successful cells have empty held-out test subsets; among 75 usable cells, 7 improve, 68 tie, and 0 worsen. Domain-level mean risk has 4 wins, 6 ties, and 0 losses (exact two-sided sign-test p=0.125). This is evidence for conservative selective deployment, not broad superiority across AMLGym.

## Paper B — source-typed evidence acquisition

Working title: **Source-Typed Information Acquisition for Procedural Action Decisions under Persistent Source Uncertainty**.

The paper asks which evidence source should be acquired before acting when physical state, semantic rule uncertainty, and persistent source behavior are jointly uncertain. The implementation contains the orientation non-identifiability construction, calibration result, latent-reliability correlation analysis, exact static-world Bellman solvers, exact evidence-count DP, ordered-history and posterior-vector cross-checks, a POMCP-style baseline, and a procedural prerequisite-acquisition adapter.

Controlled/recovered-core facts:

- count-DP exactly matches value and first action of both exact reference representations on all three 256-world / 9-query horizon-3 cases;
- history-to-count state ratio: 4.642857;
- mean count-vs-posterior-vector wall-clock speedup on the clean CI runner: about 8.51x;
- horizons 4/5/6 visit exactly 7,315 / 33,649 / 134,596 count states, matching the combinatorial formula;
- POMCP-style baseline chooses an exact-optimal root action in 5/5 controlled seeds; mean absolute value error 0.01307.

Frozen procedural evidence contains 777 MECCANO episodes (187 mixed-source episodes): Bellman expected cost 1.6657369 vs 1.7379769 for the myopic comparator, relative reduction 4.1566%, with paired 95% interval for Bellman-minus-myopic cost [-0.0799234, -0.0655095]. This experiment uses controlled perfect reveals and does not empirically identify persistent source orientation.

A separate Blue Birds held-out source-calibration experiment supports the narrower reliability-selection claim: top-5 calibration-selected sources obtain 0.90 test accuracy vs 0.75 for raw majority voting, paired bootstrap difference +0.15 with 95% interval [0.0625, 0.2375]. Naive orientation flipping reduces majority accuracy to 0.70; this negative result is retained and the stronger orientation heuristic is not claimed.

## Reproducibility

From `research/q1_2026`:

```bash
python -m pip install -r requirements.txt
make release
```

The two-paper release contains **36 tests** and regenerates Paper A controlled/stress verification, Paper B exact/POMCP/practical reports, and the recovered core benchmark. AMLGym requires `requirements-amlgym.txt` and is run separately because of its heavy external learner dependencies.

Important files:

- `paper_a/` — decision-layer repair and certification;
- `paper_b/` — source-aware acquisition and exact/approximate planning;
- `benchmarks/run_paper_a.py`, `run_paper_a_stress.py`;
- `benchmarks/run_paper_b.py`, `run_paper_b_practical.py`;
- `benchmarks/run_core.py`;
- `benchmarks/run_amlgym_confirmatory_case.py`;
- `benchmarks/merge_amlgym_confirmatory_results.py`;
- `configs/amlgym_q1_contract.json`;
- `papers/PAPER_A_DRAFT.md`;
- `papers/PAPER_B_DRAFT.md`.

## Claim boundaries

The two papers are complementary but not merged. Paper A repairs learned authorization restrictions. Paper B decides which evidence source to acquire when uncertainty remains. Synthetic mechanism validation, frozen procedural evidence, and external benchmark evidence are reported separately. No absence-of-counterexamples result is interpreted as proof of physical necessity, and no external improvement is generalized beyond the statistical unit supported by the confirmatory data.
