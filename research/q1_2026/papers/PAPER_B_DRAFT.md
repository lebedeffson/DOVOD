# Source-Typed Information Acquisition for Procedural Action Decisions under Persistent Source Uncertainty

## Abstract
Procedural decision support often faces two different uncertainties at once: uncertainty about the physical state of the procedure and uncertainty about the rule that determines whether an action is appropriate. Evidence can also come from qualitatively different sources whose reliability is persistent rather than independent across queries. We study the resulting acquisition problem: before acting, which physical, semantic, or calibration query should be purchased, and when should the system stop asking and decide? We formulate a finite static hidden-world model containing physical state, a finite semantic rule/version-space variable, and latent source properties. A persistent honest/inverted orientation is not identifiable from direct answers alone under a symmetric prior; a calibration query breaks the symmetry. For a source with reliability r, one calibration plus one direct answer has Bayes error 2r(1-r), yielding risk gain 2(r-1/2)^2 before query costs. Shared latent reliability also induces positive correlation across answers, Cov(C1,C2)=Var(R), so marginal accuracies are insufficient for planning. We derive an exact evidence-count dynamic program for static binary evidence and cross-check it against ordered-history and posterior-vector Bellman implementations. On three 256-world, 9-query horizon-3 instances, the count DP matches exact value and first action while using 1,330 states instead of 6,175 ordered-history states; on the clean CI runner its mean wall-clock speedup over the posterior-vector implementation is about 8.51x. Exact state counts continue to match the combinatorial formula at horizons 4, 5, and 6: 7,315, 33,649, and 134,596 states. A POMCP-style approximate baseline selects an exact-optimal root action in 5/5 controlled seeds with mean absolute value error 0.01307. Frozen MECCANO procedural evidence over 777 episodes shows a 4.16% expected-cost reduction for Bellman acquisition relative to a myopic comparator, but uses controlled perfect reveals and therefore does not identify real persistent source orientation. A separate held-out Blue Birds experiment supports the narrower source-reliability claim: selecting the top five workers from calibration data yields 0.90 test accuracy versus 0.75 for raw majority voting, with paired bootstrap difference +0.15 and 95% interval [0.0625, 0.2375]. Naive orientation flipping lowers majority accuracy to 0.70 and is retained as a negative result. The resulting contribution is a source-typed acquisition model with explicit identifiability boundaries, exact finite planning, and reproducible separation between theoretical source uncertainty, procedural acquisition value, and external source-calibration evidence.

## 1. Introduction
When an intelligent procedural assistant is unsure whether an action should be taken, “get more information” is not a complete policy. It may inspect a physical component, query a semantic source about the rule, calibrate a source on a known item, ask another observer, or stop and act. These choices have different costs and different information structures.

A particularly important failure mode is to model every answer as conditionally independent with a known accuracy. Real sources can have persistent characteristics: one observer may be systematically reliable, unreliable, or orientation-inverted across many questions. Repeated observations from such a source are correlated through the latent source state. If the decision model ignores this dependence, it can overvalue redundant evidence or fail to purchase calibration when calibration is precisely what makes later answers interpretable.

We ask:

**Given uncertainty about both procedural state and procedure rule, and sources with persistent latent behavior, which evidence source should be acquired next before acting?**

Our contributions are:

1. a source-orientation non-identifiability construction showing why direct answers alone can carry zero information under a symmetric honest/inverted prior;
2. a closed-form calibration result and a latent-reliability correlation identity;
3. a finite hidden-world formulation combining physical state, semantic rule/version-space uncertainty, source properties, query costs, and stop/decide actions;
4. an exact evidence-count dynamic program with exact combinatorial state-count characterization for static binary evidence;
5. independent cross-checks against ordered-history and posterior-vector Bellman implementations plus a POMCP-style approximation;
6. a practical prerequisite-acquisition adapter and reproducible evidence spanning controlled cases, frozen procedural episodes, and an external held-out source-calibration dataset.

This paper is distinct from the authorization-repair paper in the same research program. Paper A decides which learned restrictions should remain. The present paper assumes residual uncertainty remains and decides what evidence to acquire before acting.

## 2. Hidden-world formulation
Let a hidden world w contain at least three components:

- physical procedure state X;
- semantic rule or version-space variable M;
- persistent source properties S.

A final action decision d incurs loss L(d,w). Query q has cost c_q and returns observation y according to P(y|w,q). Because source properties are part of w, two answers from the same source can remain dependent after marginalizing S.

For posterior belief b(w), the optimal finite-horizon value is

V_h(b) = min { min_d E_b[L(d,w)], min_q [c_q + sum_y P(y|b,q) V_{h-1}(b_{q,y})] }.

This is a finite POMDP/Bayesian experimental-design object, but the paper focuses on the special structure created by typed procedural evidence and persistent source uncertainty.

## 3. Persistent orientation is not identifiable from direct answers alone
Consider a binary latent fact T and a source orientation O in {honest,inverted}. Conditional on orientation, the source reports the truth with reliability r>1/2 if honest and reports the complemented truth with the same reliability if inverted. Under symmetric priors P(T=0)=P(T=1)=1/2 and P(O=honest)=P(O=inverted)=1/2, a direct answer is marginally independent of T.

For example,

P(Y=1|T=1) = 0.5 r + 0.5(1-r) = 0.5,

and the same holds for T=0. Therefore I(T;Y)=0. Repeated direct answers can reveal a persistent pattern, but without an anchor they do not resolve the truth/orientation symmetry.

A calibration query asks the source about a known item and supplies the missing anchor. This is why calibration is not merely another noisy measurement of the target; it changes identifiability.

## 4. Closed-form value of calibration
Suppose one calibration answer and one direct target answer are obtained from a source with known reliability r and unknown symmetric orientation. The orientation estimate is wrong with probability 1-r, and the target answer itself is wrong with probability 1-r. The final target classification is wrong exactly when one of these two binary steps is wrong and the other is correct, giving

P(error) = r(1-r) + (1-r)r = 2r(1-r).

Without useful evidence under the symmetric construction, Bayes error is 1/2. The risk gain before query cost is therefore

1/2 - 2r(1-r) = 2(r-1/2)^2.

The result gives a simple threshold: calibration plus direct querying is useful only when its total cost is below this decision-risk gain.

## 5. Persistent reliability creates correlated evidence
Let R be a latent source reliability shared across two conditionally independent correctness indicators C1,C2, with

P(C_i=1|R)=R.

Then

E[C1 C2] = E[R^2],
E[C1]E[C2] = E[R]^2,

so

Cov(C1,C2) = Var(R).

Thus marginal source accuracy does not determine the value of repeated queries. Two models with identical one-query marginals can imply different optimal actions.

Our controlled witness uses two physical bits, each marginally 0.5. With independent bits, query cost 0.1 and horizon 1, the optimal action is to stop and decide. With perfectly correlated prior mass 0.5 on 00 and 0.5 on 11, either bit perfectly reveals both, so the optimal action is to query. Same marginals, different correlation, different information action.

## 6. Source-typed procedural acquisition
The practical adapter constructs a joint finite problem from:

- a posterior over physical prerequisite completion states, possibly correlated;
- a finite semantic version-space over prerequisite rules;
- typed physical queries about component state;
- typed semantic queries about whether a component is required;
- optional source-calibration queries;
- query-specific costs and reliability/orientation models;
- false-allow and false-block decision losses.

A hidden world determines whether the candidate action is truly authorized. The planner can query, stop, or decide. A world-expansion guard prevents accidental combinatorial explosion in large version spaces.

The adapter supports physical-only and semantic-only uncertainty, expensive-query stopping, correlated priors, and persistent source-orientation cases. These regression cases are mechanism tests rather than substitutes for real data.

## 7. Exact evidence-count dynamic programming
### 7.1 Why ordered history is wasteful
For a static hidden world and binary query outcomes, the posterior after querying source i depends on how many positive and negative outcomes have been observed from i, not on their temporal order. An ordered-history DP therefore distinguishes histories that induce the same sufficient statistic.

For q available binary queries and horizon h, an evidence-count state records for each query the counts needed by the static likelihood model. The exact implementation caches these count states and computes posterior likelihoods from the sufficient statistic.

### 7.2 Exact cross-checks
We maintain two reference solvers:

- OrderedHistoryDP, which keeps exact ordered histories and performs no history merging;
- a posterior-vector Bellman solver, which caches numerically canonicalized posterior vectors.

The count DP is checked against both in value and first action on tractable instances. The ordered-history implementation is the exact no-merge oracle; the posterior-vector implementation supplies an independent Bellman representation.

### 7.3 State-count result
For the 9-query controlled family, observed count-state numbers exactly match the combinatorial formula implemented in the benchmark. At horizon 3 the count DP visits 1,330 states versus 6,175 ordered-history states, ratio 4.642857. At horizons 4, 5, and 6 it visits exactly 7,315, 33,649, and 134,596 states respectively.

The primary claim is exact state compression under the stated static-evidence assumptions. Wall-clock speed is secondary and machine-dependent.

## 8. Controlled solver results
We generate three independent 256-world, 9-query, horizon-3 cases. In all three cases:

- count-DP value matches OrderedHistoryDP to numerical precision;
- first action matches the ordered-history oracle;
- value and first action also match the posterior-vector Bellman solver;
- theoretical and observed state counts agree exactly.

On the clean CI runner, median-per-case timing gives a mean count-DP versus posterior-vector speedup of approximately 8.51x, with minimum about 8.47x and maximum about 8.55x. These are engineering measurements, not hardware-independent complexity claims.

The deeper horizon-4/5/6 runs remain exact and complete, with root values approximately 0.42358, 0.40450, and 0.39224 in the frozen controlled instance.

## 9. POMCP-style approximate baseline
To separate exact-planner correctness from implementation-specific Bellman recursion, we also implement a POMCP-style tree-search approximation over the same hidden-world generative model. Across five controlled seeds with 20,000 simulations each, the approximate solver chooses an action belonging to the exact optimal root-action set in 5/5 cases. Mean absolute value error is 0.013065 and maximum error is 0.017696.

This is an independently implemented approximation inside the package, not an external library cross-check.

## 10. Frozen procedural evidence
The preserved MECCANO acquisition experiment contains 777 episodes, 187 of which have mixed physical/semantic source structure. Under the frozen cost model:

- Bellman expected cost: 1.6657369;
- myopic comparator: 1.7379769;
- relative reduction: 4.1566%;
- physical-first: 1.7845103;
- semantic-first: 2.0680209;
- paired 95% interval for Bellman-minus-myopic cost: [-0.0799234,-0.0655095].

The selected source changes in about 0.33155 of relevant episodes. Source-cost misspecification produces measurable regret, and an amortized prevalidation calculation reduces expected cost by about 0.14510 under the frozen assumptions. Secondary IMPACT discrimination measurements are noun AUC 0.5915 and action AUC 0.7462.

The evidence boundary is crucial: these procedural experiments use controlled perfect reveals for the acquisition simulation. They demonstrate the value of non-myopic typed acquisition under the frozen episode model, but they do not empirically estimate a real persistent reliability/orientation process.

## 11. External Blue Birds source-calibration experiment
To obtain genuinely external evidence for source calibration, we use the Blue Birds worker-label dataset distributed with `welinder/cubam`, pinned to commit `fe5ba700f1adbb489c69af311558d64370d73d36`. Tasks are deterministically split into calibration and held-out test partitions. Worker ranking uses calibration data only.

Selecting the top five workers by calibration accuracy yields test accuracy 0.90. Raw majority voting over all available workers yields 0.75. The paired accuracy difference is +0.15 with bootstrap 95% interval [0.0625,0.2375]. This supports the narrow claim that calibration can identify more useful sources for held-out decisions in this dataset.

A naive orientation-correction heuristic, which flips workers classified as inverted by calibration behavior, achieves only 0.70 versus 0.75 for raw majority voting. We retain this as a negative result. It directly warns against claiming that the theoretical honest/inverted latent model has been empirically validated by a simple flip rule.

Blue Birds is not a procedural-action dataset. It validates a source-selection component, not the complete procedural Bellman policy.

## 12. Related work
The acquisition problem is related to value-of-information and cost-sensitive decision making. Li and Oliva, *Towards Cost Sensitive Decision Making* (AISTATS 2025, PMLR 258), study learning to acquire information under feature costs. Dong et al., *Value of Information: A Framework for Human-Agent Communication* (ACL 2026), formalize information value in communication. Our contribution is narrower and structural: source-typed procedural acquisition with a finite rule/version-space variable, persistent latent source behavior, calibration/identifiability analysis, and an exact evidence-count DP for static evidence.

The planning formulation also sits within the broader POMDP literature. Kaelbling, Littman, and Cassandra provide the classical planning-under-uncertainty framework; Silver and Veness introduce Monte-Carlo planning in large POMDPs. We do not claim to introduce POMDPs, generic value of information, or active feature acquisition.

## 13. Limitations
The exact count DP relies on static hidden-world likelihood structure and finite binary evidence; changing-state or history-dependent sensors need a richer sufficient statistic. The practical adapter can expand exponentially in physical and semantic latent variables, hence the explicit world guard. The POMCP baseline is approximate and locally implemented. Frozen procedural evidence uses perfect reveals. Blue Birds supports reliability-based held-out source selection but not procedural action choice, and naive orientation correction is negative. Real persistent orientation/reliability estimation in procedural sources remains future empirical work.

## 14. Reproducibility
The repository contains orientation and correlation derivations as executable regressions, static-world likelihood code, exact posterior-vector and ordered-history solvers, count-DP, POMCP-style approximation, the procedural adapter, practical regression scenarios, the recovered core benchmark, frozen procedural evidence, and a deterministic Blue Birds benchmark pinned to a source commit. Synthetic, frozen procedural, and external source-calibration evidence are reported separately.

## 15. Conclusion
Information acquisition before procedural action depends on more than marginal uncertainty. Physical facts, semantic rules, query costs, correlation, and persistent source properties jointly determine which observation is worth buying. Calibration can be decision-relevant because it resolves a structural identifiability symmetry, not merely because it adds another noisy datum. Under static evidence, count sufficient statistics yield an exact and substantially smaller Bellman state representation. Procedural and external experiments then support two separate empirical statements: non-myopic typed acquisition can improve frozen procedural decision cost, and calibration can improve held-out source selection. The negative orientation result prevents a stronger empirical claim than the data support.

## References
1. Li, Y., Oliva, J. Towards Cost Sensitive Decision Making. AISTATS, PMLR 258, 2025.
2. Dong et al. Value of Information: A Framework for Human-Agent Communication. ACL, 2026. doi:10.18653/v1/2026.acl-long.1987.
3. Kaelbling, L. P., Littman, M. L., Cassandra, A. R. Planning and Acting in Partially Observable Stochastic Domains. Artificial Intelligence, 1998.
4. Silver, D., Veness, J. Monte-Carlo Planning in Large POMDPs. NeurIPS, 2010.
5. DeGroot, M. H. Optimal Statistical Decisions. McGraw-Hill, 1970.
6. Howard, R. A. Information Value Theory. IEEE Transactions on Systems Science and Cybernetics, 1966.
