# Paper B draft

## Identifiability-Aware Bayesian Evidence Acquisition with Persistent Source Reliability and Orientation

Sequential information acquisition becomes qualitatively different when the source itself has a persistent latent mode. In the binary orientation model a source is honest or persistently inverted with equal prior probability. For every finite direct response sequence, marginalizing orientation produces exactly the same likelihood under both task truths. Consequently the posterior task probability remains one half and direct-only mutual information is zero for every finite horizon.

A known-truth calibration query breaks this symmetry. One calibration followed by one direct task query has Bayes error `2r(1-r)` and pre-cost improvement `2(r-1/2)^2`. A persistent reliability variable also makes repeated correctness indicators dependent: after marginalizing session reliability, their covariance is `Var(R)`.

We integrate task state, procedural-model uncertainty, source reliabilities and orientations into one finite static hidden-world model. Three solvers provide independent checks: a numerical posterior-vector Bellman DP with disclosed cache canonicalization, a no-merge ordered-history oracle, and an exact evidence-count DP. Under stationary conditional-independent likelihoods, history order is irrelevant given `(query,outcome)` counts, so the count vector is an exact sufficient statistic. The full count lattice has `C(h+2Q,h)` states, versus the exponential ordered-history tree.

For the frozen 256-world, nine-query cases, count DP matches the history oracle in value and first action on all three H3 seeds while using 1330 rather than 6175 states. H4/H5/H6 contain exactly 7315/33649/134596 count states. An independent POMCP-style root-sampling implementation is validated on a tractable oriented calibration case. These are mechanism and algorithmic results; real source prevalence, calibration probes and acquisition costs remain external empirical gates.
