# Exact Evidence-Count Planning for Cost-Sensitive Procedural Information Acquisition

## Q1-ready manuscript snapshot — 8 September 2026

## Abstract
Before a procedural system acts, it may need to buy information: inspect a component, query a semantic attribute, calibrate a source, or stop and decide. This is naturally a partially observable decision problem, but the practical question is not whether a POMDP can represent it; it is when non-myopic information acquisition is worth its cost and which solver regime is defensible. We study a finite static hidden-world class with binary evidence and derive an exact evidence-count dynamic program whose state is a sufficient statistic for ordered observation history. On three independent 256-world, nine-query, horizon-three instances it exactly matches ordered-history and posterior-vector Bellman solvers while using 1,330 count states instead of 6,175 ordered histories; on the frozen clean CI runner it is about 7.19x faster than the posterior-vector reference. The empirical evaluation deliberately contains positive and negative regimes. On 777 preserved MECCANO episodes, exact lookahead reduces expected cost by 4.16% relative to a myopic comparator, with the gain concentrated in ambiguous episodes. On the official IMPACT PSR split, exact horizon three and myopic horizon one tie on all 19 train-supported test transitions, showing that deeper planning is unnecessary when one typed query already resolves local ambiguity. A third held-out procedural distribution, Assembly101, is evaluated with a leakage-safe train-only policy construction: 200,813 of 258,825 official test transitions are covered (77.59%). Exact horizon-two acquisition has mean realized cost 0.07882 versus 0.31832 for horizon-one myopic acquisition, a reduction of 0.23951. A whole-video cluster bootstrap over all 1,055 test videos gives a 95% interval [-0.24191,-0.23707] for exact-minus-myopic cost, and every evaluated video has lower mean exact cost. The advantage remains positive under 0.5x-5x query-cost scaling. Controlled observation-noise stress shows a graceful decay of non-myopic value rather than a perfect-reveal-only effect. Approximation results are intentionally mixed: a 15,000-simulation POMCP-style solver recovers an exact-optimal root action in 12/12 easy Q=9 cases, but on 12 acquisition-active hard cases it recovers 0/60 exact-optimal roots at 20,000 simulations across five seeds per case. We therefore treat exact count-DP as the auditable reference for tractable static decisions, myopic acquisition as the mandatory engineering baseline, and approximation as a workload-validated fallback rather than a default. A structural typed-core ablation further preserves the exact root decision in 48/48 fresh cases while reducing count states by about 1.83x. The paper’s contribution is a regime-selection framework backed by exact equivalence, negative controls, cost/reliability misspecification stress, and three real procedural distributions; it does not claim a universal POMDP compression theorem or universal superiority of non-myopic acquisition.

## 1. Introduction
“Get more information before acting” is not an actionable policy unless the system can choose which information to acquire, how much it costs, how source quality affects its value, and when to stop. In procedural assistance, the uncertainty can mix physical state, semantic rule uncertainty, and persistent source properties. A one-step value-of-information rule is attractive because it is cheap and interpretable, but it can miss interactions between observations. Full lookahead is more principled but can be computationally expensive and may add no value at all on easy local decisions.

We therefore ask a bounded question:

**For a short static procedural decision, when does finite-horizon non-myopic acquisition materially improve the decision, and which exact or approximate solver should be used in each workload regime?**

The paper makes six contributions.

1. We derive an exact evidence-count sufficient statistic for a declared static binary-emission class and verify Bellman equivalence against two independent exact representations.
2. We characterize exact state growth and show a reproducible engineering speed advantage over a posterior-vector Bellman reference.
3. We evaluate non-myopic value on three distinct procedural workloads with opposite regimes: positive MECCANO, negative IMPACT PSR, and strongly positive held-out Assembly101.
4. We harden Assembly101 against split leakage and transition pseudoreplication: priors, candidates, and verb/noun query metadata come from the official training split only, while uncertainty is additionally quantified by resampling whole held-out videos.
5. We retain approximation failures rather than selecting favorable cases. POMCP succeeds on the original easy exact-solvable family but fails on deliberately acquisition-active hard cases even at 20,000 simulations.
6. We study deployment-relevant misspecification: query costs, likelihood quality, observation noise, and safe query screening. The resulting recommendation is a regime selector, not a universal solver claim.

This paper is separate from the companion authorization-repair study. That work changes which learned restrictions remain in an action-authorization surface. Here the authorization decision remains uncertain and the problem is which evidence to acquire before acting.

## 2. Static hidden-world formulation
Let a hidden world w contain physical state X, a semantic model or rule variable M, and optional persistent source properties S. A terminal decision d incurs loss L(d,w). Query q costs c_q and returns y according to P(y|w,q). Source properties may be included in w, so repeated answers can remain dependent after marginalizing S.

For belief b and finite horizon h,

V_h(b) = min { min_d E_b[L(d,w)], min_q [ c_q + sum_y P(y|b,q)V_{h-1}(b_{q,y}) ] }.

Stopping is part of the optimization: the planner is never forced to buy evidence. The central modeling restriction is that w is static during the short acquisition episode. If the physical process changes materially while evidence is collected, the sufficient statistic below is no longer valid and a dynamic model is required.

## 3. Exact evidence-count dynamic programming
### 3.1 Sufficient statistic
Suppose each repeatable query has a binary observation and emissions are conditionally independent given the static hidden world. Index query-outcome pairs by j. If p_j(w) is the corresponding emission probability and n_j(H) is the number of times it appears in ordered history H, then

L(H;w) = product_j p_j(w)^{n_j(H)}.

Hence two histories with the same count vector induce the same posterior and the same remaining decision problem. Ordered history can therefore be replaced exactly by an evidence-count vector.

For Q repeatable binary queries and horizon h, the complete count lattice contains

N_count(Q,h) = C(h+2Q,h).

This is an exact result for the declared static likelihood class. It is not claimed for arbitrary POMDPs, changing-state sensors, or history-dependent emissions.

### 3.2 Independent exact cross-checks
Three independently generated 256-world, Q=9, h=3 cases are solved with:

- `OrderedHistoryDP`, which performs no history merging;
- a posterior-vector Bellman implementation;
- the evidence-count DP.

All three implementations agree in root value and first action. Count-DP visits 1,330 states versus 6,175 ordered histories, a 4.642857x reduction. At horizons 4, 5, and 6 the exact count solver visits 7,315, 33,649, and 134,596 states, matching the combinatorial formula.

On the frozen clean V4 CI runner, count-DP is 7.186x faster on average than the posterior-vector reference, with a range of 7.052x-7.271x over the three cases. Wall-clock timing is treated only as engineering evidence; value, action, and exact state-count identities are the scientific invariants.

## 4. Approximation is a fallback, not an automatic replacement
An initial approximation check uses 12 independent 512-world Q=9,h=3 cases. At 15,000 simulations, the POMCP-style solver selects an action from the exact optimal root-action set in 12/12 cases, with mean absolute value error 0.002912 and maximum 0.006451. This establishes that the implementation can behave well on the original exact-solvable family.

That result is not sufficient for deployment. A separate mechanism-based selection constructs balanced hidden-world priors and retains the first 12 cases whose exact nominal-cost root action is genuinely `QUERY`. For each case we run five fixed POMCP seeds at budgets up to 20,000 simulations and score the chosen action using the exact Bellman root action-value table. At 20,000 simulations, 0/60 runs choose an exact-optimal root action, 0/60 even recover the correct QUERY-vs-DECIDE action family, mean true root-action regret is 0.02324, and maximum regret is 0.05018. Tuning the exploration constant on a development set does not rescue the conclusion: the selected constant 0.2 yields 0 exact-root recovery on confirmation, QUERY-vs-DECIDE accuracy 0.4444, and mean true regret 0.02238.

This positive/negative pair is important. POMCP is retained as an approximation baseline and a possible large-state fallback, but it is not recommended merely because exact enumeration becomes inconvenient. Approximation must be validated against exact cases that are acquisition-active and structurally similar to the intended workload.

## 5. Practical solver reduction without losing exactness
A shallow “top-K query” screen appears attractive on development cases: h1 top-6 achieves about 3.89x state reduction and low average regret. However, a targeted semantic-root stress exposes a severe failure: the exact optimal semantic query ranks 7th-10th under the shallow score, so h1 top-6 obtains 0/8 exact root matches. Ranking with h2 instead of h1 still gives 0/8 exact root matches. A scalar h1-margin gate is similarly unhelpful: thresholds up to 0.005 escalate no cases, whereas thresholds from 0.01 escalate all cases on a 48-case panel.

We therefore keep a simpler structural ablation. The typed-core rule retains all `state` and `model_feature` queries and removes only `calibrate_physical` and `calibrate_semantic`. On 32 fresh unconditional cases, eight physical-root stress cases, and eight semantic-root stress cases, this eight-query exact problem matches the full exact root decision in 48/48 cases and has zero full-policy regret in all 48. Mean count-state reduction is 1.8277x and mean CI wall-clock speedup is approximately 1.82-1.86x across the three panels.

This is an engineering ablation, not a theorem that calibration queries are always removable. The rule is enabled only after workload-specific ablation shows that calibration queries are decision-irrelevant for the target family.

## 6. Real procedural workloads
### 6.1 MECCANO: positive but concentrated non-myopic value
The preserved MECCANO workload contains 777 episodes; 187 have mixed physical/semantic acquisition structure. The replay uses controlled perfect reveals, so it evaluates policy structure and cost—not real sensor reliability.

At the frozen semantic-cost-1 setting, exact Bellman expected cost is 1.6657369 versus 1.7379769 for the myopic comparator, a 4.1566% relative reduction. The paired 95% interval for Bellman-minus-myopic cost is [-0.0799234,-0.0655095]. Physical-first and semantic-first fixed policies are worse at 1.7845103 and 2.0680209.

The average hides strong heterogeneity. Among the 187 mixed episodes, 44.92% have positive exact-over-myopic gain and 55.08% are ties. The top 20% of episodes account for 61.52% of all positive gain. For ambiguity groups k=1,2,3, positive-gain fractions are 0%, 35.29%, and 85.71%. When the two policies choose different first intervention types, mean gain is 0.26322; when they choose the same type, mean gain is 0.04277. This supports selective non-myopic planning rather than universal deeper lookahead.

### 6.2 IMPACT PSR: a real negative regime
The second workload uses IMPACT v1.1 Procedure Step Recognition annotations pinned to source commit `4fed5faa5f05f7aece55712e458defa1f372b248`. The next-step prior is learned from the official train split only and evaluated on the official test split. Simultaneous multi-label frames are excluded rather than arbitrarily ordered.

With minimum training support five, 19 of 25 official test transitions are covered. Exact h=3 and myopic h=1 have identical mean realized cost 0.0373684: 0 exact wins, 19 ties, 0 losses, and no root-action disagreements. A support-threshold-3 sensitivity gives the same result. Thus one typed query is already sufficient and deeper planning adds no value.

The acquisition decision is nevertheless cost-sensitive. All 19 covered transitions change root action somewhere on the declared 0.5x,1x,2x,5x,10x cost grid. IMPACT therefore separates “lookahead depth” from “information cost”: non-myopic planning can be unnecessary while information acquisition remains highly sensitive to cost assumptions.

### 6.3 Assembly101: large held-out positive regime
The third workload uses Assembly101 fine-grained action annotations. The target is the next fine-grained action conditioned on the current action; typed queries reveal the next action’s verb or noun/object attribute. These are annotation-derived information channels, not physical-vs-semantic source labels.

The evaluation was explicitly hardened against split leakage. The empirical next-action prior, candidate next actions, and action-to-verb/noun metadata are all constructed from the official training annotations only. The official test split is used only for held-out transition evaluation and consistency checking for action IDs already seen in training. The test split contains 68 action IDs not present in training (2,370 row occurrences); none enters policy construction, and the report records zero evaluated transitions using test-only action metadata.

The official training set yields 2,514 sequences and the test set 1,055 sequences. There are 258,825 held-out adjacent test transitions. The prespecified minimum-support-five adapter covers 200,813 transitions, or 77.5864%, across 4,496 unique eligible transition pairs. The main uncovered block is 53,493 transitions whose true next action was never observed after the same current action in training; these cases are not imputed.

On covered transitions, exact h=2 mean realized cost is 0.078816 versus 0.318325 for myopic h=1, a mean reduction of 0.239509. Root actions differ on 25.32% of evaluated transitions. Exact planning is better on 57,104 transitions, tied on 43,673, and worse on 100,036. This count asymmetry is not hidden: when exact is better, mean reduction is 0.90451; when it is worse, mean increase is only 0.03553. Thus the aggregate gain is driven by fewer high-consequence wins rather than a higher per-transition win frequency.

A transition-level paired bootstrap gives a 95% interval [-0.24147,-0.23762] for exact-minus-myopic cost. Because adjacent transitions from the same video are not iid, we additionally resample whole official test videos. All 1,055 test videos contain evaluated transitions, and every video has lower mean exact cost than myopic cost. The transition-weighted whole-video cluster bootstrap with 10,000 draws gives [-0.24191,-0.23707]; an equal-video-weighted estimate is -0.23635 with interval [-0.23899,-0.23371]. The cluster analysis therefore confirms that the large effect is not an artifact of treating adjacent transitions as independent bootstrap units.

Support-threshold sensitivity at 3, 5, and 10 produces the same 77.5864% coverage and the same mean reduction. Joint query-cost multipliers retain a positive exact-over-myopic mean reduction throughout the declared grid: 0.25309 at 0.5x, 0.23951 at 1x, 0.21433 at 2x, and 0.15427 at 5x.

Assembly101 materially changes the empirical picture: non-myopic value is not limited to the smaller preserved MECCANO replay. At the same time, the result is deliberately bounded to train-supported next-action ambiguity under verb/noun information channels and normalized decision-loss costs.

## 7. Observation noise, cost error, and source calibration
### 7.1 Noisy evidence
A controlled known-noise sweep evaluates binary observation flip probabilities epsilon in {0,0.05,0.10,0.20}. Mean Bellman advantage over myopic planning decreases from 0.18645 to 0.16070, 0.09297, and 0.01463 as noise increases. All four cases retain positive advantage through epsilon=0.10; three of four remain positive at epsilon=0.20. This is evidence of graceful degradation and rules out the explanation that non-myopic value exists only under perfect reveals.

### 7.2 Query-cost misspecification
The frozen MECCANO replay already shows that stale costs can dominate solver improvements: assuming semantic cost 1 when true cost is 5 gives regret 0.247639, rising to 0.442896 at true cost 10. A controlled joint physical/semantic cost grid further selects eight acquisition-active cases without inspecting robust-policy outcomes. In all eight, the optimal root action and optimal source kind change somewhere on the grid. A minimax-regret first-action selector lowers worst-case first-action regret in 8/8 cases, with mean reduction 0.09093 and maximum 0.10430. This is a bounded first-action rule over a declared scenario set, not a full robust POMDP.

### 7.3 Likelihood-quality misspecification
On the acquisition-active exact family, underestimating evidence informativeness produces premature stopping. With likelihood scale alpha=0.50, the planner buys an expected 0 queries, has mean wrong-decision probability 0.50, and mean full-policy regret 0.02324. At alpha=0.75 it buys 0.156 queries with wrong-decision probability 0.4902 and regret 0.01905. At the correctly specified alpha=1.0 it buys 1.853 queries, wrong-decision probability falls to 0.4057, and regret is zero by construction. Mild overconfidence at alpha=1.25 buys 1.976 queries and introduces small regret 0.00254. Source calibration therefore changes stopping behavior, not merely reported probabilities.

### 7.4 Held-out source-selection evidence
The Blue Birds worker-label dataset, pinned to `welinder/cubam` commit `fe5ba700f1adbb489c69af311558d64370d73d36`, supplies a separate source-quality experiment. Tasks are deterministically split into calibration and held-out test partitions. Selecting the top five workers by calibration accuracy achieves 0.90 test accuracy versus 0.75 for raw majority voting over all workers; the paired difference is +0.15 with bootstrap 95% interval [0.0625,0.2375]. A naive orientation-flipping heuristic falls to 0.70 and is retained as a negative result. This supports calibration-based source selection only; it does not validate the complete procedural Bellman policy or a persistent orientation model in MECCANO/Assembly101.

## 8. Deployment hierarchy
The combined evidence supports the following engineering hierarchy.

1. If hidden state changes materially during acquisition, do not use this static count-DP; use a dynamic model.
2. Always evaluate a myopic policy. IMPACT demonstrates a regime in which deeper lookahead adds exactly no held-out benefit.
3. For small or medium static ambiguous decisions, use exact count-DP as the auditable reference. MECCANO and Assembly101 demonstrate workloads where lookahead matters.
4. Remove query types only after workload-specific ablation. The typed-core exact rule is useful on the tested family; shallow top-K ranking is not reliable under semantic-root stress.
5. Do not switch to POMCP solely because the nominal query count is large. Validate approximate root decisions on acquisition-active exact-solvable cases first; the hard-case failure here is substantial.
6. If plausible query costs cross policy boundaries, use explicit scenario analysis, robust first-action selection, or abstention rather than executing a nominal plan blindly.
7. Calibrate source likelihoods when they affect stopping. Misspecified informativeness can convert an evidence-acquisition policy into immediate decision-making.

The practical recommendation is therefore not “always plan deeper.” It is “identify the workload regime, keep a strong myopic baseline, retain an exact oracle where tractable, and validate every approximation or screen against acquisition-active exact cases.”

## 9. Limitations
The exact count representation assumes a static hidden world and finite binary emissions that are conditionally independent given the world. It is not valid for changing-state procedures or arbitrary history-dependent sensors without extending the sufficient statistic.

The exact solver remains combinatorial as Q and h grow. The structural typed-core screen is an empirical ablation for the tested family, not a universal safe reduction. Likewise, the POMCP hard-case result concerns this implementation and controlled acquisition-active family; it does not establish a universal impossibility result for Monte Carlo POMDP planning.

MECCANO uses controlled perfect reveals, so it cannot estimate real sensor/expert reliability. IMPACT and Assembly101 query outcomes are deterministic annotation properties and query costs are normalized decision-loss units, not measured money, latency, or human effort. Assembly101 covers only transitions supported by the training conditional prior; 22.41% of official test transitions remain outside the evaluated adapter. Verb/noun channels do not validate physical-vs-semantic source typing or persistent source orientation.

The whole-video Assembly101 bootstrap handles within-video transition dependence at the resampling level, but it does not establish transfer to unseen institutions, procedures, or annotation ontologies. Blue Birds validates held-out source selection in a non-procedural crowdsourcing dataset and should not be conflated with procedural action choice.

## 10. Reproducibility
The repository separates exact-equivalence tests, controlled approximation checks, acquisition-active approximation stress, MECCANO and IMPACT workloads, leakage-safe Assembly101 evaluation, whole-video cluster uncertainty, observation-noise stress, query-cost robustness, source-likelihood misspecification, structural screening, and external source calibration.

The Assembly101 evaluation records exact file receipts before analysis:

- `train.csv`: 100,677,017 bytes; SHA-256 `748cd23d772d2ecf9c4506378401d26765afc5c096bc7b2c802e5733fcefa766`;
- `test.csv`: 47,993,475 bytes; SHA-256 `afcda4252eadc3d7ab1c4a59924da8fdff1dd76e9add249656218292181a7959`.

The final practical workflow (`34258190161`) completed all focused tests, controlled Paper A scaling, Paper B noise/deep-horizon stress, leakage-safe Assembly101 evaluation, 10,000-draw whole-video cluster bootstrap, and artifact publication successfully. The artifact digest is SHA-256 `1f8f63e3af5cbde105956988caddb0ae1eb6b3a127da2e44a38f2c6b5131165c`. Standard CI passes on Python 3.10, 3.11, and 3.12, and the full Q1 research core plus Blue Birds external jobs pass on the same finalized code line.

## 11. Conclusion
Exact evidence-count dynamic programming gives a compact, auditable Bellman representation for a useful but explicitly restricted class of static procedural acquisition problems. Its practical value is workload-dependent. MECCANO contains concentrated non-myopic gains; IMPACT PSR is a clean negative regime in which deeper lookahead is unnecessary; and leakage-safe held-out Assembly101 provides a large positive regime, with exact h=2 reducing mean realized cost from 0.31832 to 0.07882 across 200,813 covered transitions and a whole-video 95% exact-minus-myopic interval [-0.24191,-0.23707]. Observation noise attenuates rather than immediately destroys non-myopic value. Cost and likelihood misspecification can dominate solver choice. Approximation is not automatically safe: POMCP succeeds on easy exact-solvable cases yet fails decisively on acquisition-active hard cases at the tested budgets. The resulting contribution is a regime-selection methodology: myopic when one query is enough, exact count-DP when local ambiguity is tractable, screened exact planning only after ablation, robust stopping when costs are uncertain, and approximation only after acquisition-active validation.

## References
1. Krishnamurthy, V. *Partially Observed Markov Decision Processes: From Filtering to Controlled Sensing*. Cambridge University Press, 2016.
2. Kaelbling, L. P., Littman, M. L., Cassandra, A. R. Planning and Acting in Partially Observable Stochastic Domains. *Artificial Intelligence*, 1998.
3. Silver, D., Veness, J. Monte-Carlo Planning in Large POMDPs. *NeurIPS*, 2010.
4. Li, Y., Oliva, J. Active Feature Acquisition with Generative Surrogate Models. *ICML*, 2021.
5. Li, Y., Shan, S., Liu, Q., Oliva, J. B. Towards Robust Active Feature Acquisition. arXiv:2107.04163, 2021.
6. Li, Y., Oliva, J. Towards Cost Sensitive Decision Making. *AISTATS*, PMLR 258, 2025.
7. Zhao, G. et al. Bayesian Active Learning by Soft Mean Objective Cost of Uncertainty. *AISTATS*, 2021.
8. Dong et al. Value of Information: A Framework for Human-Agent Communication. *ACL*, 2026. doi:10.18653/v1/2026.acl-long.1987.
9. MacDonald, R. A., Smith, S. L. Active sensing for motion planning in uncertain environments via mutual information policies. *International Journal of Robotics Research*, 2019.
10. Ragusa, F., Furnari, A., Farinella, G. M. MECCANO: A multimodal egocentric dataset for humans behaviour understanding in the industrial-like domain. *Computer Vision and Image Understanding* 235, 103764, 2023.
11. De Finetti, B. *Theory of Probability*. Wiley, 1974.
