# Formal results draft

## Paper A

**A. Positive-only prerequisite non-identifiability.** If prerequisite `p` is true in every observed successful pre-state, a model requiring `p` and the otherwise identical model omitting `p` agree on all observed positives but disagree on a counterfactual state. Necessity is not identified by those observations alone.

**B. Decision-equivalent compression.** For every observed blocked state, record the candidate prerequisites it violates. A retained prerequisite set preserves all blocks iff it hits every violation set. Minimum empirical decision-equivalent compression is minimum hitting set.

**C. Contextual containment.** Contextual exceptions that waive a prerequisite set `D` admit no state that would remain blocked after globally deleting `D`; guards only reduce permissions. This is a permissiveness result, not a correctness or safety guarantee.

**D. PAC-Bayes sparse-mask certificate.** For fixed vocabulary size `M`, Bernoulli inclusion prior `rho`, and a deterministic selected mask with `k` edits, the point-posterior KL is `k log(1/rho)+(M-k) log(1/(1-rho))`. Insert this into the standard PAC-Bayes-kl inequality under its IID/bounded-loss/prior-independence assumptions and invert binary KL numerically.

## Paper B

**E. All-finite direct non-identifiability.** Under equal prior mass on honest and inverted persistent orientation, changing binary task truth swaps the two mixture likelihood terms for any direct-response sequence; the marginal likelihood is unchanged, so posterior truth remains one half.

**F. Calibration threshold.** One known-truth calibration plus one direct query has optimal error `2r(1-r)`, hence pre-cost gain `2(r-1/2)^2` over the direct-only symmetric baseline.

**G. Count sufficiency.** In a static hidden world with stationary conditionally independent likelihoods, `P(H|W)=prod_{q,o} P(o|q,W)^{N_{q,o}(H)}`. Posterior and Bellman continuation value therefore depend on history only through evidence counts.

**H. Count-state cardinality.** The number of nonnegative `2Q`-category count vectors with total mass at most `h` is `C(h+2Q,h)`. For `Q=9`: H3=1330, H4=7315, H5=33649, H6=134596.

The count quotient fails when unmodeled source state evolves with order/time, likelihoods depend on unrecorded history, or reward contains an order-dependent term not represented in the state.
