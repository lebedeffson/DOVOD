# Theorem and proposition ledger

## Paper A

**A1 Positive-only non-identifiability.** If candidate prerequisite `p` is true in every observed successful pre-state, the model requiring `p` and the otherwise identical model omitting it agree on all observed positives but disagree on a counterfactual state. Necessity is not identified by those positives alone.

**A2 Decision-equivalent compression.** For each observed blocked state let `V_s` be the set of candidate prerequisites it violates. A retained set preserves every observed block iff it hits every `V_s`; minimum preservation is minimum hitting set.

**A3 Contextual containment.** If contextual exceptions waive prerequisite set `D`, every state admitted by the contextual repair is also admitted after globally deleting `D`; guards can only shrink the contextual allow set. This is permissiveness containment, not a safety theorem.

**A4 Finite-vocabulary repair.** Exact and weighted-soft MILPs globally optimize the frozen contextual edit class. The soft objective is edit cost plus weighted false-allow/false-block errors.

**A5 PAC-Bayes-kl specialization.** For a fixed edit vocabulary and sample-independent Bernoulli subset prior, a deterministic selected mask has explicit KL description cost and a standard PAC-Bayes-kl upper risk bound under the stated IID bounded-loss assumptions.

## Paper B

**B1 All-finite direct non-identifiability.** With symmetric honest/inverted persistent orientation, marginalizing orientation gives identical likelihood for every finite direct response sequence under both binary task truths, hence MI is zero.

**B2 Calibration theorem.** One known-truth calibration followed by one direct query has Bayes error `2r(1-r)` and pre-cost gain `2(r-1/2)^2`.

**B3 Persistent reliability.** If session reliability `R` is sampled once and correctness indicators are conditionally IID Bernoulli(`R`), then `Cov(C_i,C_j)=Var(R)` for `i!=j`.

**B4 Exact count sufficiency.** In a static hidden world with stationary conditionally independent binary query likelihoods, posterior and Bellman value depend on history only through `(query,outcome)` counts.

**B5 Count-state cardinality.** With `Q` binary repeatable queries and horizon `h`, the full evidence-count lattice contains `C(h+2Q,h)` states. For `Q=9`, H3/H4/H5/H6 are exactly 1330/7315/33649/134596.
