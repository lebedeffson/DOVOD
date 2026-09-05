# Mathematical core of the DOVOD short papers

## 1. Falsifying empirical prerequisite hypotheses

For a candidate unary prerequisite `(p,a)`, a strict support learner proposes `p -> a` when every observed pre-action state for action `a` in the fit data has `p=1`.

A direct empirical counterexample is a successful observed transition of `a` from a state with `p != 1`. Let

`C(p,a) = {u : independent carrier u contains at least one such counterexample}`.

Then `C(p,a) != empty` falsifies the universal empirical relation in the observed state representation. `C(p,a)=empty` means only **unfalsified in the available evidence**; it does not prove mechanical necessity.

A relation carried by `c=|C(p,a)|` of `m` independent units survives deletion of any `d` carriers whenever `c>d`.

For finite-population rarefaction, a uniformly selected `k`-unit subset misses a relation with probability

`Pr(miss | c,m,k) = C(m-c,k) / C(m,k)`.

Averaging `1-Pr(miss)` over the already observed refuted-relation set gives the exact expected discovery fraction. This is saturation analysis of a finite observed set, not an estimator of unseen relation species.

For a separate zero-failure check, if `n` genuinely independent/exchangeable Bernoulli observations contain zero failures, the one-sided `1-alpha` upper bound is

`p_upper = 1 - alpha^(1/n)`.

Neither calculation is a safety certificate or proof of mechanical necessity.

## 2. Choosing what information to acquire

The exact short-paper model keeps a finite semantic version space and physical completion probabilities. Two perfect-reveal information actions are available:

- physical query: reveal one binary component state at cost `c_p`;
- semantic review: identify the target action's prerequisite alternative within the current version space at cost `c_s`.

For knowledge state `s`, exact planning follows

`V(s) = min_{z in Z(s)} [ c(z) + E_o V(T(s,z,o)) ]`.

The myopic baseline selects the query with maximum immediate expected uncertainty reduction per unit cost and repeats the rule after each observation.

The reliability experiment is deliberately a **separate noisy one-step stress**. The exact Bellman recursion above assumes perfect reveals; repeated correlated/noisy observations are not claimed as solved by that recursion.

### Prevalidation accounting

If one pre-runtime review costs `c_pre`, `|R|` ambiguities are reviewed once, and the result is reused for `H` episodes,

`C_amortized = C_runtime + |R| * c_pre / H`.

For the frozen 777-episode stress at semantic cost 2, three reviews give runtime cost 1.245739. With `c_pre=2` and reuse over 777 episodes, amortized total cost is 1.253461 versus baseline 1.466206, a 14.51% reduction. These are normalized model units, not measured time, money or expert workload.

### Exact-to-myopic guard

With `k` unresolved binary physical predicates, each predicate has three memoization statuses: unresolved, revealed 0, revealed 1. If the target action has `g` semantic alternatives, a conservative envelope for the current perfect-reveal model is

`N_upper(k,g) = (g+1) * 3^k`.

At `k=10,g=8`, the measured checkpoint uses 531,414 Bellman states and the envelope is 531,441. `HybridSourceSelector` uses the envelope as a cheap guard: exact Bellman below a configured threshold, gain-per-cost myopic fallback above it.

## Claim boundary

The current algorithms do not establish real sensor latency, empirical expert-review reliability, measured human workload, causal semantics, mechanical safety, or authoritative SOP truth. Those require separate domain validation.
