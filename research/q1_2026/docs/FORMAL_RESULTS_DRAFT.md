# Formal results draft for the two journal manuscripts

This file is a compact proof-oriented companion to `THEOREM_LEDGER.md`. It intentionally avoids claims that depend on the pending external benchmark.

## Paper A

### Result A — positive-only prerequisite non-identifiability

For conjunctive prerequisite hypotheses, if predicate `p` is true in every observed successful pre-state, then the model that requires `p` and the otherwise identical model that omits `p` agree on all observed positives. They disagree on a counterfactual state where all other prerequisites hold and `p` does not. Necessity of `p` is therefore not identifiable from the positive observational protocol alone.

### Result B — decision-equivalent compression as hitting set

For every observed blocked state, record the set of candidate prerequisites it violates. A retained prerequisite subset preserves the block exactly when it hits that violation set. Simultaneous preservation of all blocks is therefore a hitting-set feasibility condition, and minimum-cardinality decision-equivalent compression is minimum hitting set.

### Result C — contextual repair containment

If contextual exceptions waive base prerequisites in a set `D`, globally deleting `D` waives at least the same prerequisites in every state. Contextual guards only remove additional permissions. Thus the action set admitted by the contextual repair is contained in the action set admitted by global deletion of `D`.

### Result D — PAC-Bayes sparse-mask certificate

For fixed vocabulary size `M` and Bernoulli inclusion prior `rho`, deterministic mask `h` with `k` selected edits has description KL

`k log(1/rho)+(M-k)log(1/(1-rho))`.

Insert this in the standard PAC-Bayes-kl inequality and numerically invert binary KL to obtain an upper bound on true bounded decision risk under the theorem's IID and sample-independent-prior assumptions.

## Paper B

### Result E — all-finite direct non-identifiability under symmetric source orientation

Marginal likelihood of a direct-response sequence is an equal mixture of the honest and inverted likelihoods. Flipping task truth swaps the two mixture terms, leaving the likelihood unchanged. The posterior task truth remains its prior after every finite direct history.

### Result F — calibration threshold

A known-positive calibration updates orientation. The next direct response can then be interpreted or inverted optimally. The resulting accuracy is `r^2+(1-r)^2`; error is `2r(1-r)`; improvement over the symmetric direct-only `1/2` error is `2(r-1/2)^2`.

### Result G — count sufficiency

With a static hidden world and stationary conditionally independent query likelihoods,

`P(H|W)=prod_{q,o} P(o|q,W)^{N_{q,o}(H)}`.

Therefore the posterior and Bellman continuation value are functions of the evidence-count vector, not the ordering of history. The exact history MDP admits an exact quotient by count vectors.

### Result H — count-state cardinality

The number of nonnegative `2Q`-dimensional count vectors with total count at most `h` is

`sum_{t=0}^h C(t+2Q-1,2Q-1)=C(h+2Q,h)`.

For `Q=9`: H3=1330, H4=7315, H5=33649, H6=134596, exactly the current cache-state counts.

## Assumption failure cases that must remain visible

- A prerequisite can become identifiable once counterexamples/interventions or trusted negative labels are available; A does not claim universal non-identifiability.
- Contextual repair containment compares permissions, not correctness or safety.
- PAC-Bayes risk certification requires its sampling/prior assumptions; clustered trajectory frames cannot be silently treated as IID.
- Count sufficiency fails if source state evolves with time/order, query likelihoods adapt to unrecorded history, or costs/rewards depend on order beyond the count/posterior state.
- The orientation theorem uses a symmetric honest/inverted prior; known-orientation ordinary noisy sensors do not have zero direct information.
