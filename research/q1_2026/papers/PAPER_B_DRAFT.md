# Exact Evidence-Count Dynamic Programming and Cost-Robust Information Acquisition for Static Procedural Decisions

## Abstract
Information acquisition before a procedural decision is a standard partially observable decision problem; this paper does not introduce a new POMDP paradigm. We study a restricted executable class in which the hidden world is static during acquisition and observations are binary. The main algorithmic contribution is an exact evidence-count dynamic program that merges order-equivalent histories without changing Bellman value. On the original nine-query family it visits 1,330 states at horizon three instead of 6,175 ordered histories, and exact state counts remain 7,315, 33,649, and 134,596 at horizons four, five, and six. On the final clean CI runner the horizon-three count implementation is about 7.19x faster than the posterior-vector Bellman implementation (7.05x-7.27x across the three frozen cases); timing is engineering evidence only, while value/action/state identities are primary. Approximation validation is expanded to 12 independent 512-world cases with the same Q=9,h=3 decision class: 15,000-simulation POMCP selects an exact-optimal root action in 12/12 cases with mean absolute value error 0.00291 and maximum 0.00645. The practical contribution is deliberately regime-specific. On 777 preserved MECCANO episodes, Bellman acquisition reduces expected cost by 4.16% relative to myopic planning, but only 44.92% of the 187 mixed-uncertainty episodes have positive gain and the top 20% account for 61.52% of total positive gain. A second real procedural benchmark on the official IMPACT PSR train/test split gives the opposite result: among 19 train-supported test transitions, exact h=3 and myopic h=1 tie in all 19, while the preferred root action still changes with query-cost scale in 19/19 transitions. Controlled cost-grid experiments further show that a minimax-regret first-action rule reduces worst-case regret in 8/8 acquisition-active cases. The resulting contribution is a deployment hierarchy rather than a universal solver claim: exact count-DP for small static ambiguous decisions, approximation when exact enumeration is too large, myopic acquisition when one query already resolves the decision, and robust stopping/abstention when plausible source costs cross policy boundaries.

## 1. Introduction
When an intelligent procedural assistant is unsure whether an action should be taken, “get more information” is not a complete policy. It may inspect a physical component, query a semantic source about the rule, calibrate a source on a known item, ask another observer, or stop and act. These choices have different costs and different information structures.

The research question is not whether a POMDP can model this situation—it can—but when finite-horizon lookahead adds measurable value over a one-step policy, how far exact state representation can be compressed for a static evidence class, and what the system should do when acquisition costs themselves are uncertain.

We ask:

**For a short static procedural decision, when should the system buy another physical, semantic, or calibration observation, and when should it stop because one-step acquisition is sufficient or the cost model is too uncertain?**

Our contributions are:

1. an exact evidence-count sufficient statistic for a declared static binary-evidence class, with exact Bellman equivalence to ordered history;
2. independent exact cross-checks against ordered-history and posterior-vector Bellman implementations and exact combinatorial state-count characterization;
3. approximation validation on 12 independent 512-world cases plus a Q=12-26 engineering stress that explicitly separates exact and approximate regimes;
4. two real procedural workloads with opposite outcomes: MECCANO, where non-myopic value is concentrated in structurally ambiguous episodes, and IMPACT PSR, where one informative query makes h=3 tie h=1 on all covered test transitions;
5. a cost-uncertainty diagnostic in which a minimax-regret first-action rule reduces worst-case regret in all eight acquisition-active controlled cases;
6. a bounded source-calibration component: persistent orientation/reliability is modeled explicitly and a held-out Blue Birds experiment validates calibration-based worker selection, but not the full procedural Bellman policy.

This paper is distinct from the authorization-repair paper in the same research program. Paper A decides which learned restrictions should remain. The present paper assumes residual uncertainty remains and decides what evidence to acquire before acting.

## 2. Hidden-world formulation
Let a hidden world w contain at least three components:

- physical procedure state X;
- semantic rule or version-space variable M;
- persistent source properties S.

A final action decision d incurs loss L(d,w). Query q has cost c_q and returns observation y according to P(y|w,q). Because source properties are part of w, two answers from the same source can remain dependent after marginalizing S.

For posterior belief b(w), the optimal finite-horizon value is

V_h(b) = min { min_d E_b[L(d,w)], min_q [c_q + sum_y P(y|b,q) V_{h-1}(b_{q,y})] }.

The stop/decide action is part of the same optimization. The static assumption means w does not change while evidence is collected. This is appropriate only for short local decisions; changing-state procedures require a dynamic model.

## 3. Persistent orientation and calibration
Consider a binary latent fact T and a source orientation O in {honest,inverted}. Conditional on orientation, the source reports the truth with reliability r>1/2 if honest and reports the complemented truth with the same reliability if inverted. Under symmetric priors P(T=0)=P(T=1)=1/2 and P(O=honest)=P(O=inverted)=1/2, a direct answer is marginally independent of T:

P(Y=1|T=1) = 0.5 r + 0.5(1-r) = 0.5,

and the same holds for T=0. Thus direct evidence alone does not resolve the truth/orientation symmetry in this construction.

A calibration query asks about a known item and supplies an anchor. With one calibration answer and one direct target answer, the Bayes error is

P(error) = 2r(1-r),

so the pre-cost reduction from the symmetric 1/2 error is

1/2 - 2r(1-r) = 2(r-1/2)^2.

This is an identifiability observation for the declared model, not a claim that real procedural sources are literally honest/inverted.

## 4. Shared latent reliability creates dependent evidence
Let R be a latent source reliability shared across two conditionally independent correctness indicators C1,C2 with P(C_i=1|R)=R. Then

Cov(C1,C2) = Var(R).

This is a standard consequence of a shared latent Bernoulli parameter and is used here only as a modeling warning: marginal accuracy is insufficient to determine the value of repeated evidence.

A two-bit controlled witness makes the decision consequence concrete. With independent bits, query cost 0.1 and horizon one, the optimal action is to stop. With prior mass 0.5 on 00 and 0.5 on 11, the same single-bit marginals hold, but one query reveals both bits and becomes optimal. Same marginals, different joint structure, different information action.

## 5. Source-typed procedural acquisition
The practical adapter constructs a joint finite problem from:

- a posterior over physical prerequisite completion states, possibly correlated;
- a finite semantic version-space over prerequisite rules;
- typed physical queries about component state;
- typed semantic queries about whether a component is required;
- optional source-calibration queries;
- query-specific costs and reliability/orientation models;
- false-allow and false-block decision losses.

A hidden world determines whether the candidate action is truly authorized. The planner can query, stop, or decide. A world-expansion guard prevents accidental combinatorial explosion in large version spaces.

## 6. Exact evidence-count dynamic programming
### 6.1 Sufficient statistic
For a static hidden world and conditionally independent binary emissions given w, the likelihood of an ordered history depends on the history only through how many times each query produced each outcome. If j indexes query-outcome pairs, p_j(w) is the corresponding emission probability, and n_j(H) is its count in history H, then

L(H;w) = product_j p_j(w)^{n_j(H)}.

Two ordered histories with the same count vector therefore induce the same posterior and the same remaining decision problem. The Bellman state can be indexed by the evidence-count vector rather than the ordered sequence.

For Q repeatable binary queries and horizon h, the full count lattice contains

N_count(Q,h) = C(h+2Q,h).

This exact equivalence is the main algorithmic result. It relies on the stated static likelihood structure and is not a generic POMDP compression theorem.

### 6.2 Exact cross-checks
We maintain two reference solvers:

- OrderedHistoryDP, which keeps exact ordered histories and performs no history merging;
- a posterior-vector Bellman solver, which caches numerically canonicalized posterior vectors.

On three independent 256-world, nine-query, horizon-three instances, count-DP matches the ordered-history oracle and posterior-vector implementation in root value and first action. It uses exactly 1,330 count states versus 6,175 ordered histories, a 4.642857x state reduction. At horizons 4, 5, and 6 the count solver visits exactly 7,315, 33,649, and 134,596 states, matching the theoretical formula.

On the clean CI runner for frozen V4 evidence, mean horizon-three count-vs-posterior-vector wall-clock speedup is 7.186x, with range 7.052x-7.271x. These timings are machine-dependent engineering measurements; exact state-count, value, and action identities are the primary result.

## 7. Approximation and intended scale
A second controlled block expands approximation validation to 12 independently generated 512-world, Q=9,h=3 cases. With 15,000 simulations, the POMCP-style solver selects an action in the exact optimal root-action set in 12/12 cases. Mean absolute value error is 0.002912 and maximum is 0.006451.

A separate large-query stress keeps h=6 and increases Q to 12, 18, 24, and 26. The corresponding count-state envelopes are 593,775; 5,245,786; 25,827,165; and 40,475,358. Ordered-history envelopes are approximately 199 million; 2.24 billion; 12.49 billion; and 20.16 billion. This block is an engineering scalability diagnostic, not an exact-optimality claim for Q>9.

Ten-thousand-simulation POMCP runs complete in roughly 0.44-0.78 seconds on the recorded CI environment. In all 12 stress runs (three seeds at each Q), the chosen root action matches the one-step myopic action. This negative result is retained: increasing the number of available queries does not by itself create non-myopic value. The value of lookahead must come from decision structure.

## 8. Frozen MECCANO procedural workload
The preserved MECCANO acquisition experiment contains 777 episodes, 187 of which have mixed physical/semantic acquisition structure. Query outcomes in this replay are controlled perfect reveals, so the experiment tests the acquisition policy and cost model, not real sensor reliability.

Under the frozen semantic-cost-1 model:

- Bellman expected cost: 1.6657369;
- myopic comparator: 1.7379769;
- relative reduction: 4.1566%;
- physical-first: 1.7845103;
- semantic-first: 2.0680209;
- paired 95% interval for Bellman-minus-myopic cost: [-0.0799234,-0.0655095].

The selected source changes in about 0.33155 of relevant episodes. This is a stable average advantage under the frozen workload, but it does not imply that every episode benefits from deeper planning.

### 8.1 When does non-myopic planning help?
A post-reviewer decomposition over the same frozen 187 mixed-uncertainty episodes characterizes where the gain concentrates. At semantic cost 1, 44.92% of episodes have positive myopic-minus-Bellman gain and 55.08% are exact ties. The top 20% of episodes account for 61.52% of total positive gain. When the two policies choose different first intervention types, mean gain is 0.26322; when they choose the same type, mean gain is 0.04277.

The ambiguity strata make the pattern more explicit. For the frozen mask-size groups k=1,2,3, positive-gain fractions are 0%, 35.29%, and 85.71%, with mean gains 0, 0.09115, and 0.10444 respectively. Thus the 4.16% aggregate is not a uniformly small improvement; it is a mixture of many ties and a smaller set of materially ambiguous decisions.

The cost regime can eliminate that value. At semantic cost 2, mean gain falls to 0.00537 and only 9.63% of episodes have positive gain; at cost 5 all 187 episodes tie; at cost 10 positive gain occurs in only 8.02% of episodes. This motivates explicit cost-robustness analysis rather than a universal lookahead recommendation.

## 9. Second procedural benchmark: IMPACT PSR
To test whether the MECCANO advantage transfers automatically, we add a second procedural distribution using IMPACT v1.1 Procedure Step Recognition annotations pinned to source commit `4fed5faa5f05f7aece55712e458defa1f372b248`. The next-step conditional prior is learned only from the official train split. Evaluation uses the official test split and excludes simultaneous multi-label frames rather than imposing an arbitrary order.

The target is the next singleton PSR step conditioned on the current singleton PSR step. Three typed queries reveal the annotated next-step component, install/remove operation, and normal/incorrect-installation quality. Query costs are normalized decision-loss units 0.04, 0.03, and 0.04; they are not measured money, human time, or sensor latency.

With the prespecified minimum train support of five, 19 of 25 official test transitions are covered (76%). On those 19 transitions, exact horizon-three and myopic horizon-one policies have identical mean realized cost 0.0373684 and the comparison is 0 exact wins / 19 ties / 0 losses. The horizon-three root action never differs from the horizon-one root action. A prespecified support-threshold-3 sensitivity run yields the same 19 covered transitions and the same 19/19 ties.

This is a real negative regime and is retained as such. The local candidate set is small enough and the first typed query is informative enough that deeper lookahead adds no value. Yet cost sensitivity remains strong: all 19 covered transitions change root action somewhere on the declared 0.5x,1x,2x,5x,10x cost grid. For example, after `Remove anti_vibration_handle`, with true next step `Remove bearing_screw_topleft`, the exact root action is QUERY component at 0.5x through 5x but becomes DECIDE at 10x. IMPACT therefore separates two questions that MECCANO alone could not: deeper lookahead can be unnecessary even while the acquisition decision remains highly cost-sensitive.

## 10. Query-cost misspecification and first-action robustness
The frozen MECCANO replay already exposes the danger of stale costs. If semantic cost is assumed to be 1 while its true value is 5, regret is 0.247639; at true cost 10, regret rises to 0.442896. These penalties are larger than the 4.16% Bellman-vs-myopic workload gain and therefore can dominate the benefit of non-myopic planning.

We add a controlled first-action robustness diagnostic over a predeclared joint physical/semantic cost grid: (0.5,0.5), (1,1), (2,1), (1,2), (5,1), (1,5), (5,5), (10,1), and (1,10). The evaluated cases are the first eight balanced-prior seeds whose exact nominal-cost root action is QUERY; selection does not inspect high/low-cost outcomes or minimax performance.

Across all 8 cases, the exact optimal root action changes somewhere on the grid, and the optimal source kind also changes in all 8. A minimax-regret first-action rule reduces worst-case first-action regret in 8/8 cases, with mean reduction 0.09093 and maximum 0.10430. Under this deliberately broad uncertainty set it chooses DECIDE rather than immediately buying evidence. This is a controlled first-action rule, not a full robust-POMDP solution and not an empirical distribution of industrial costs.

An unconditional random-prior diagnostic is also retained: all eight tested cases trivially choose immediate DECIDE under every cost. We do not use that negative block as evidence of robustness.

## 11. External Blue Birds source-calibration experiment
The procedural replay above uses controlled reveals and therefore does not estimate real persistent source reliability. To obtain separate held-out evidence for calibration-based source selection, we use the Blue Birds worker-label dataset distributed with `welinder/cubam`, pinned to commit `fe5ba700f1adbb489c69af311558d64370d73d36`. Tasks are deterministically split into calibration and held-out test partitions. Worker ranking uses calibration data only.

Selecting the top five workers by calibration accuracy yields test accuracy 0.90. Raw majority voting over all available workers yields 0.75. The paired accuracy difference is +0.15 with bootstrap 95% interval [0.0625,0.2375]. This supports the narrow claim that calibration can identify more useful sources for held-out decisions in this dataset.

A naive orientation-correction heuristic, which flips workers classified as inverted by calibration behavior, achieves only 0.70 versus 0.75 for raw majority voting. We retain this as a negative result. Blue Birds is not a procedural-action benchmark and does not validate the complete Bellman acquisition policy.

## 12. Practical deployment guidance
The combined evidence supports a regime-selection hierarchy rather than a solver-centric recommendation:

1. If hidden state changes materially during acquisition, do not use the static count-DP; use a dynamic model.
2. If one affordable typed query already resolves the local ambiguity, use the myopic policy; the IMPACT PSR regime is an example.
3. If several typed queries interact after one observation and the local static model is tractable, use exact count-DP as an auditable oracle; the more ambiguous MECCANO strata motivate this regime.
4. If the latent/query space is too large for exact enumeration, switch to POMCP or another factorized/approximate planner and validate root decisions against exact cases where possible.
5. If plausible acquisition costs cross a policy boundary, evaluate a declared cost scenario set and use a robust first-action rule or DECIDE/abstain rather than blindly executing the nominal-cost plan.
6. Treat real sensor/source reliability as a separate calibration problem. The current procedural replay does not estimate it.

The myopic policy should remain the mandatory engineering baseline. Non-myopic planning is justified only when replay under the same cost model shows a material reduction. For repeated workloads, estimating query costs from observable resource use and comparing calibration expense with amortized benefit is essential; the frozen MECCANO prevalidation calculation yields a 14.51% amortized reduction under its declared assumptions, but this number is model-specific.

## 13. Related work
The formulation belongs to the established POMDP, controlled-sensing, value-of-information, and active feature acquisition literature. Kaelbling, Littman, and Cassandra provide the classical planning-under-uncertainty framework; Silver and Veness introduce Monte-Carlo planning in large POMDPs. Li and Oliva, *Towards Cost Sensitive Decision Making* (AISTATS 2025, PMLR 258), study cost-sensitive information acquisition. Dong et al., *Value of Information: A Framework for Human-Agent Communication* (ACL 2026), formalize information value in communication.

We do not claim to introduce POMDPs, value of information, latent-mixture dependence, or POMCP. The paper-specific algorithmic claim is the exact evidence-count sufficient statistic for the stated static binary query class, together with a procedural evaluation that explicitly identifies both positive and negative lookahead regimes and a bounded first-action cost-robustness rule.

## 14. Limitations
The exact count DP relies on a static hidden-world likelihood structure and finite binary evidence; changing-state or history-dependent sensors need a richer sufficient statistic. The practical adapter can expand exponentially in physical and semantic latent variables, hence the explicit world guard. Exact optimality is demonstrated only on the local Q=9 family; Q=12-26 is an approximation/scalability stress with no exact-optimality claim. POMCP root-action recovery is validated on 12 controlled 512-world cases, not arbitrary large POMDPs.

MECCANO query responses are perfect reveals in replay, so real sensor/expert reliability remains unmeasured. IMPACT PSR covers 19/25 official test transitions under the train-supported conditional prior; uncovered transitions are not imputed. Its query outcomes are deterministic annotation properties and its costs are normalized decision-loss units, not measured latency or money. The IMPACT result is intentionally negative for lookahead and therefore bounds rather than weakens the claim.

The minimax-regret extension addresses uncertainty over a declared discrete scenario set and only the first action; it is not a full robust-POMDP solution or a learned real-world cost model. Blue Birds validates held-out source selection but not procedural action choice, and naive orientation flipping is negative. No claim is made that non-myopic acquisition is preferable when the workload-specific gain does not justify additional engineering complexity.

## 15. Reproducibility
The repository separates controlled exact/approximation tests, frozen MECCANO evidence, the post-reviewer MECCANO gain decomposition, the official-split IMPACT PSR negative benchmark, large-query stress, cost-uncertainty diagnostics, and Blue Birds source-calibration validation. On the frozen V4 evidence head `08366db201ddcd1f68adcd138ca8f21376ed505a`, the clean CI release passes 46 automated tests. Machine-readable receipts include `paper_b_meccano_gain_summary_v4.json`, `paper_b_impact_psr_summary_v4.json`, `paper_b_scalability_robustness_summary.json`, and `paper_b_bluebirds_external_summary.json`. External-data sources are pinned to commits, and synthetic, procedural, and source-calibration evidence remain explicitly separated.

## 16. Conclusion
The paper makes four bounded claims. First, for static binary evidence, count sufficient statistics give an exact Bellman representation that is substantially smaller than ordered histories and can serve as a local oracle. Second, a POMCP-style approximation recovers an exact-optimal root action in 12/12 controlled 512-world cases and remains computationally light in the recorded Q=26,h=6 stress, although exact optimality is not claimed there. Third, non-myopic value is workload-structural: on 187 mixed MECCANO episodes at semantic cost 1, 44.92% show positive gain and the top 20% account for 61.52% of all positive gain, while on a second real IMPACT PSR test distribution exact h=3 and myopic h=1 tie on all 19 covered transitions. Fourth, acquisition remains cost-sensitive even when deeper lookahead is unnecessary; IMPACT changes root action across the declared cost grid in 19/19 covered transitions, and the controlled minimax-regret selector reduces worst-case first-action regret in 8/8 acquisition-active cases. The practical contribution is therefore a regime-selection framework rather than a universal solver: exact for small static ambiguous decisions, approximate when enumeration is too large, myopic when one query already resolves the decision, and robust/abstaining when the dominant uncertainty is acquisition cost itself.

## References
1. Krishnamurthy, V. Partially Observed Markov Decision Processes: From Filtering to Controlled Sensing. Cambridge University Press, 2016.
2. Li, Y., Oliva, J. Towards Cost Sensitive Decision Making. AISTATS, PMLR 258, 2025.
3. Zhao, G. et al. Bayesian Active Learning by Soft Mean Objective Cost of Uncertainty. AISTATS, 2021.
4. Dong et al. Value of Information: A Framework for Human-Agent Communication. ACL, 2026. doi:10.18653/v1/2026.acl-long.1987.
5. Kaelbling, L. P., Littman, M. L., Cassandra, A. R. Planning and Acting in Partially Observable Stochastic Domains. Artificial Intelligence, 1998.
6. Silver, D., Veness, J. Monte-Carlo Planning in Large POMDPs. NeurIPS, 2010.
7. Li, Y., Oliva, J. Active Feature Acquisition with Generative Surrogate Models. ICML, 2021.
8. Li, Y., Shan, S., Liu, Q., Oliva, J. B. Towards Robust Active Feature Acquisition. arXiv:2107.04163, 2021.
9. MacDonald, R. A., Smith, S. L. Active sensing for motion planning in uncertain environments via mutual information policies. International Journal of Robotics Research, 2019.
10. Ragusa, F., Furnari, A., Farinella, G. M. MECCANO: A multimodal egocentric dataset for humans behaviour understanding in the industrial-like domain. Computer Vision and Image Understanding, 235, 103764, 2023.
11. De Finetti, B. Theory of Probability. Wiley, 1974.
