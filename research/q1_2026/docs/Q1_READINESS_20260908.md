# Q1 readiness review — 8 September 2026

## Decision

The practical package has crossed the point where another synthetic benchmark is likely to add more reviewer confidence than manuscript consolidation. The two studies are now **Q1-credible research packages**, not guaranteed Q1 acceptances. Paper B has the stronger empirical breadth; Paper A remains intentionally conservative in its external claim because its prespecified AMLGym domain-level test is not broadly significant.

The stop rule used for this review was simple: continue practical strengthening until the largest remaining threats were either removed or made explicit. The final two threats were (i) held-out metadata leakage in the Assembly101 adapter and (ii) pseudoreplication from treating adjacent transitions as iid. Both are now closed.

## Paper B: practical gate passed

The leakage-safe Assembly101 evaluation constructs the empirical next-action prior, candidate set, and verb/noun query metadata from the official training split only. The test split contains 68 unseen action IDs and 2,370 occurrences of them; none enters policy construction. The official test workload contains 258,825 adjacent transitions, of which 200,813 (77.5864%) are train-supported and evaluated.

On those transitions, exact h=2 mean realized cost is 0.078816 versus 0.318325 for myopic h=1, giving mean reduction 0.239509. Exact planning is better on fewer individual transitions than it is worse (57,104 vs 100,036), but its wins are much larger: mean gain 0.90451 when exact wins versus mean loss 0.03553 when exact loses. This asymmetry is reported explicitly rather than hidden behind the average.

The key final check resamples whole held-out videos instead of transitions. All 1,055 test videos contain eligible transitions and all 1,055 have lower mean exact cost than myopic cost. A 10,000-draw transition-weighted video-cluster bootstrap gives exact-minus-myopic 95% interval **[-0.24191,-0.23707]**. Equal-video weighting gives -0.23635 with interval [-0.23899,-0.23371]. The large Assembly101 result therefore survives the dependence correction.

The result is also stable across the declared query-cost grid: mean exact-over-myopic reductions are 0.25309, 0.23951, 0.21433, and 0.15427 at 0.5x,1x,2x,5x costs. Support thresholds 3,5,10 yield the same covered workload and mean reduction.

This complements rather than replaces the other practical regimes: MECCANO gives a 4.1566% positive non-myopic effect concentrated in ambiguous episodes, while IMPACT PSR gives a clean 19/19 tie negative regime. Controlled noise stress shows graceful decay of non-myopic value through epsilon=0.20. Hard acquisition-active POMCP validation remains negative (0/60 exact-optimal root actions at 20,000 simulations), which prevents an unjustified approximate-solver claim. The typed-core exact ablation preserves the exact root decision and zero regret in 48/48 fresh cases while reducing exact state count by about 1.83x.

**Paper B practical verdict:** no further benchmark expansion is justified before submission formatting/reviewer simulation. The empirical story is now stronger if it stays bounded than if more datasets are appended.

## Paper A: practical gate passed with a narrower claim

The original controlled eight-bit study has exact planted repair recovery, zero complete-support risk, 0/4096 independent holdout errors, and a one-sided exact 95% upper risk bound 7.31e-4. The new 16-bit two-action study adds two overlapping authorization surfaces with 160 candidate edits per action. At n=1024, all six action-by-seed fits recover the planted structure exactly with zero full-support risk. Smaller n values include incomplete recovery and nonzero risk, which exposes rather than hides sample complexity.

The frozen AMLGym matrix remains the external anchor: 160 prespecified cells are retained; 75 have usable held-out decisions; DOVOD improves 7, ties 68, and worsens 0 versus upstream. The prespecified domain summary is 4 wins, 6 ties, 0 losses with p=0.125, so the paper must not claim broad superiority. The seeded reproducibility amendment is slightly weaker (6/69/0; domain 3/7/0) and is retained instead of replacing the primary result.

The post-confirmatory drill-down is explanatory only but operationally useful: 216 operator surfaces reduce to 13 counterexample-flagged surfaces, seven calibration-approved non-empty repairs, and six abstentions. Across 4,448 held-out decisions in those selected positive domains, errors fall 90 to 20; a coarse 80%-threshold global override improves 0/32 cells.

Independent MECCANO/IMPACT auditing reduces 272 candidate prerequisite relations to 71 and 13 unresolved hypotheses. Carrier rarefaction reaches 95% of expected observed refutations by 9/11 MECCANO recordings and 11/13 IMPACT participants. A hard two-carrier counterexample rule is an important negative control: downstream recall falls from 0.90131 to 0.78836.

**Paper A practical verdict:** Q1-credible only under the selective-deployment / decision-equivalent-repair framing. Do not rewrite it as a broad superiority or causal prerequisite-discovery paper. Its external statistical breadth is the main residual reviewer risk, but adding post-hoc significance fishing would weaken rather than strengthen the paper.

## Final manuscript snapshots

The consolidated submission-oriented manuscripts are:

- `papers/PAPER_A_Q1_READY.md` — selective decision-equivalent authorization repair;
- `papers/PAPER_B_Q1_READY.md` — exact evidence-count planning and regime selection.

The older `PAPER_A_DRAFT.md` and `PAPER_B_DRAFT.md` remain as provenance snapshots and should not be used as the primary submission text after this finalization.

## Reproducibility receipts

The final practical workflow is GitHub Actions run `34258190161`, completed successfully. Its artifact `q1-final-practical-evidence-20260908` has SHA-256 digest `1f8f63e3af5cbde105956988caddb0ae1eb6b3a127da2e44a38f2c6b5131165c`.

Assembly101 source receipts recorded before evaluation:

- train.csv — 100,677,017 bytes; SHA-256 `748cd23d772d2ecf9c4506378401d26765afc5c096bc7b2c802e5733fcefa766`;
- test.csv — 47,993,475 bytes; SHA-256 `afcda4252eadc3d7ab1c4a59924da8fdff1dd76e9add249656218292181a7959`.

On the finalized code line, standard CI passes on Python 3.10, 3.11, and 3.12, and the Q1 research-core plus Blue Birds external jobs are green.

## What is deliberately not being added

No additional synthetic dataset, arbitrary larger horizon, extra post-hoc subgroup, or significance test is added merely to increase the experiment count. Remaining work should now be editorial: venue-specific length/LaTeX, figures/tables, reviewer-style claim audit, and citation/notation cleanup. A new practical experiment should be opened only if a concrete reviewer concern cannot be answered by the existing evidence.
