# From Sequence Regularity to Action Authorization: Decision-Equivalent and Certified Repair of Learned Preconditions

## Abstract
Action models learned from procedural demonstrations can inherit ordering regularities that are predictive of observed behavior without being justified as hard authorization conditions. Treating every recurrent predecessor as mandatory can false-block valid actions, whereas global deletion of suspicious preconditions can create false allows. We study a downstream, learner-agnostic repair problem: which learned restrictions should remain in the authorization surface, where should local exceptions be admitted, and where are additional local guards needed? Positive procedural traces alone do not identify mechanical necessity. For a finite candidate prerequisite family, minimum empirical decision-equivalent prerequisite selection reduces to Minimum Hitting Set and is NP-hard. We introduce a finite vocabulary of contextual prerequisite exceptions and guards and optimize exact or weighted-soft mixed-integer objectives over authorization decisions. The repair changes applicability decisions only; learned effects remain untouched. In a controlled eight-bit benchmark, the exact solver recovers three planted local edits, obtains zero complete-support risk versus 0.125 for both the upstream rule and the best global-deletion baseline, and makes 0 errors on an independent 4,096-decision holdout, giving a one-sided exact 95% risk bound of 7.31e-4. With 5% label corruption and 640 observations, mean complete-support risk is 0.00156 and maximum risk is 0.00781. We additionally run a frozen AMLGym 1.0.11 confirmatory protocol over 20 IPC-style domains, four learner families, and two trace budgets. All 160 prespecified cells are retained. Five are upstream failures/timeouts, 80 have empty held-out decision subsets, and 75 contain usable test decisions; among these, 7 improve, 68 tie, and 0 worsen against the upstream applicability decision. At the prespecified domain level the comparison is 4 wins, 6 ties, and 0 losses (exact two-sided sign-test p=0.125). Against a calibration-gated global-override baseline, the domain summary is 1 win, 8 ties, and 1 loss (p=1.0). We therefore claim conservative selective deployment and independently auditable decision repair, not broad superiority across action-model learners.

## 1. Introduction
A repeated action order is not the same thing as a necessary precondition. A technician may always inspect before tightening because of habit, training, interface layout, or data-collection policy. When a learner converts this regularity into a hard applicability restriction, a downstream planner or assistant may reject a valid action. Conversely, deleting the restriction globally may authorize the action in states for which the original restriction was useful.

We separate action-model learning from action authorization repair. An upstream learner remains responsible for producing a symbolic or symbolic-like action model. DOVOD operates downstream and asks a narrower decision question:

**Which learned restrictions deserve to remain in the authorization surface, and in which contexts should they be locally relaxed or strengthened?**

The distinction matters for both novelty and evaluation. We do not claim to introduce planning-domain repair, nor do we claim to infer causal mechanical necessity from demonstrations. Our target is a decision-equivalent authorization layer whose edits are explicit, local, optimized within a frozen finite vocabulary, and independently evaluated.

Our contributions are:

1. a positive-only non-identifiability result for prerequisite necessity;
2. an exact reduction of finite minimum decision-equivalent prerequisite selection to Minimum Hitting Set, with mandatory/optional/redundant optimum-role diagnostics;
3. a contextual edit language containing local prerequisite exceptions and local guards;
4. exact and weighted-soft MILP repair over a predeclared finite vocabulary;
5. independent decision-risk evidence using exact binomial bounds, paired comparisons, and finite-class certificates;
6. a frozen AMLGym confirmatory protocol that separates repair, calibration-only deployment gating, and test evaluation by semantic-state fingerprint.

## 2. Positive demonstrations do not identify necessity
Let action a have candidate prerequisites P={p1,...,pm}. A positive observation is a state x in which a is executed successfully. If every positive observation satisfies pi, then the data are compatible with at least two worlds: one in which pi is mechanically necessary and one in which the demonstrated policy simply never visits states violating pi before executing a. These worlds induce the same positive observations.

Therefore positive-only sequence evidence cannot, without additional assumptions, identify mechanical prerequisite necessity. Absence of a counterexample is not proof that no counterexample exists. This motivates a weaker operational object: equivalence of authorization decisions on an observed state distribution.

If a false prerequisite has counterexample mass at least q, n independent positive observations miss all counterexamples with probability at most (1-q)^n. To make this at most delta,

n >= log(delta) / log(1-q).

For q=0.05 and delta=0.05, n=59 is sufficient. Without a positive lower bound on q there is no finite absence guarantee.

## 3. Decision-equivalent prerequisite selection
For each negative authorization state x_j, define V_j as the set of candidate prerequisites violated in that state. A retained prerequisite set S continues to block every observed negative state iff S intersects every V_j. Hence minimum decision-equivalent selection is

min sum_i z_i
subject to sum_{p_i in V_j} z_i >= 1 for all j,
z_i in {0,1}.

This is Minimum Hitting Set and is NP-hard. The family of minimum solutions gives three useful roles relative to the observed decision problem:

- mandatory-optimal: present in every minimum solution;
- optional-optimal: present in some but not all minimum solutions;
- redundant-relative-to-observed-decisions: present in no minimum solution.

These roles are not claims of physical necessity.

## 4. Contextual repair language
Global deletion is too coarse for many procedural errors. We use two local edit types over a fixed context-literal vocabulary.

An exception E=(p,C) waives learned prerequisite p only when context conjunction C holds. A guard G=C blocks the action when C holds even if upstream prerequisites otherwise allow it. Context width is bounded, and the entire edit vocabulary is generated before fitting labels are inspected.

For a selected edit set R, the repaired authorization function is evaluated directly on each labeled state. The upstream effects and transition model are left unchanged. This makes DOVOD a decision-layer repair rather than a new action-model learner.

## 5. Exact and soft optimization
Let e_k denote candidate edits and z_k their binary selection variables. In the noiseless case we minimize weighted edit count subject to exact agreement with observed authorization labels. In the noisy case we add observation-error indicators xi_j and solve a weighted objective

sum_k w_k z_k + lambda_+ sum_{j:y_j=1} xi_j + lambda_- sum_{j:y_j=0} xi_j.

Different penalties can encode asymmetric false-allow and false-block costs. The resulting solution is globally optimal relative to the declared finite vocabulary, not necessarily relative to an unknown causal rule class.

## 6. Certification
We separate fitting evidence from independent evaluation.

### 6.1 Independent exact holdout
For k errors among n independent holdout decisions we use a one-sided Clopper-Pearson upper confidence bound. With zero errors,

R <= 1 - alpha^(1/n).

At n=4096 and alpha=0.05 this is 0.0007311.

### 6.2 Paired comparison
Baseline and repair are evaluated on the same decisions, so discordant outcomes support an exact McNemar/binomial comparison rather than an unpaired test.

### 6.3 Finite-class training certificate
If the vocabulary of M edits is fixed before labels and at most s edits may be selected, then the number of candidate subsets is bounded by

H = sum_{i=0}^s C(M,i).

In the controlled benchmark M=128 and s=3, giving H=349,633 and a zero-training-error uniform risk upper bound of 0.0243248 at the declared confidence level. A PAC-Bayes-kl calculation is retained as a secondary synthetic certificate; the independent exact holdout is primary.

## 7. Controlled benchmark
The controlled problem uses eight binary state variables and all 256 states. The upstream rule requires variables 0,1,2. Ground truth differs through three local edits: two prerequisite exceptions and one guard. A width-two vocabulary over four context variables contains 128 candidate edits.

The exact MILP is fitted on 640 sampled authorization observations. It selects exactly the three planted edits and fits all training decisions. On complete support, repaired risk is 0.0, while both the upstream rule and the best globally deleted prerequisite subset have risk 0.125.

On an independently sampled 4,096-decision holdout, the repair makes 0 errors. The exact one-sided 95% upper risk bound is 0.0007311. The upstream baseline makes 514 errors, yielding a paired exact p-value of 3.73e-155. Both the false-allow and false-block holdout subsets contain zero repaired errors.

## 8. Stress experiment
Training size is varied over 64, 128, 256, and 640 under 0% and 5% label corruption across five seeds. Small samples sometimes fail to identify all local contexts, which is expected. At 256 noise-free observations all five seeds recover the planted repair exactly and have zero complete-support risk. The same holds at 640 noise-free observations.

At 640 observations with 5% corruption, the weighted-soft solver has mean complete-support risk 0.0015625 and maximum 0.0078125; four of five seeds recover the exact planted structure. This is a mechanism/robustness result, not external validation.

## 9. AMLGym confirmatory evaluation
### 9.1 Frozen protocol
The external study uses AMLGym 1.0.11 with 20 domains, learners SAM, OffLAM, NOLAM, and ROSAME, and trace budgets 3 and 10: 160 prespecified cells.

An earlier pilot was used only to design protocol v4. Before confirmatory labels are used, CI replays the historical pilot-state selector and removes every semantic duplicate of every inspected pilot state. Remaining candidate states are ranked by a new SHA-256 semantic-state key. The problem-scoped semantic-state fingerprint is the split unit.

A fixed hash partitions selected states into repair [0,500), calibration [500,750), and test [750,1000). Repairs are fitted only on repair observations. Deployment is chosen per operator using calibration observations only. A nonempty repair is deployed only if it strictly lowers calibration risk and does not increase calibration false allows; ties revert to upstream. Test labels never influence fitting, gate selection, hyperparameters, or CI pass/fail.

A label-free preflight verifies zero pilot/confirmatory semantic-state overlap in all 20 domains.

### 9.2 Execution and retained failures
The canonical matrix contains all 160 cells and is protocol-clean. Four upstream learner failures are reproducible on childsnack: OffLAM raises KeyError 'kitchen' for both budgets, while NOLAM hits an upstream PDDL parser SyntaxError on `(xist ?param_1)` for both budgets. ROSAME at budget 10 on sokoban exceeds the frozen 900-second per-case limit and is retained as a timeout. These outcomes are not repaired or silently dropped.

Among successful cells, 80 have an empty held-out decision subset under the frozen semantic split and are reported as empty rather than converted into zero-risk successes. Seventy-five cells contain usable test decisions.

### 9.3 Scientific result
Among the 75 usable cells, DOVOD improves 7, ties 68, and worsens 0 relative to the upstream applicability decision. The unweighted mean cell-level risk reduction is 0.00829365. At the prespecified domain level the mean-risk comparison gives 4 wins, 6 ties, and 0 losses with exact two-sided sign-test p=0.125. Nonzero domain mean improvements are barman 0.0208333, floortile 0.0178571, parking 0.03125, and tpp 0.0078125.

Against the calibration-gated global-override baseline, the domain summary is 1 win, 8 ties, and 1 loss with p=1.0; DOVOD is slightly worse in barman and better in floortile. Consequently, the confirmatory result does not support a broad claim that contextual repair dominates the global gated comparator.

The appropriate interpretation is conservative: the calibration gate mostly abstains, no usable held-out cell is worsened relative to the upstream decision, and a small subset of domains benefits. Domain-level evidence is not statistically broad at conventional thresholds. This is evidence for selective deployment behavior, not universal superiority.

### 9.4 Post-freeze reproducibility diagnostic
A later clean replay of the same v4 split exposed an execution-level limitation in the upstream learner layer: although DOVOD state selection, semantic fingerprints, repair/calibration/test partitioning, repair fitting, and deployment gating are deterministic, AMLGym learner processes had originally been launched without a global `PYTHONHASHSEED` and common Python/NumPy/PyTorch seed. The replay therefore produced a different set of learned upstream models and a different aggregate scientific summary while the protocol preflight remained clean.

We do not replace the frozen primary result with whichever replay is more favorable. The 7/68/0 result above remains the prespecified primary confirmatory analysis. A post-freeze reproducibility amendment now pins `PYTHONHASHSEED=0`, Python and NumPy RNG seed 20260906, and the same PyTorch seed for ROSAME. It leaves domains, budgets, split fingerprints, repair vocabulary, calibration gate, metrics, and the 900-second scientific case limit unchanged. Amended runs are reported as reproducibility diagnostics and compared against the frozen result rather than used for retrospective model or result selection.

## 10. Procedural falsification evidence
Frozen MECCANO/IMPACT analyses motivate why such a repair layer is useful. Of 272 candidate directed component relations, 201 are refuted in MECCANO TRAIN and 259 in IMPACT Reassembly-A. In a nested held-recording MECCANO evaluation, calibration raises recall from 0.8707348 to 0.8852784, gain 0.0145437 with bootstrap interval [0.0106327, 0.0191601]. These are falsification and operational-risk results, not proof that unrefuted restrictions are mechanically necessary.

## 11. Related work
Safe action-model learning under partial observability, including Le, Juba, and Stern (AAAI 2024), studies guarantees during model learning. Aineto and Scala (KR 2024) similarly study action-model learning with guarantees. DOVOD is downstream and learner-agnostic.

Planning-domain repair is the closest neighboring line. Counterexample-based correction (Lin et al., AAAI 2025), the IJCAI 2025 survey on model repair, trajectory-guided repair in Robotics and Computer-Integrated Manufacturing (2026), and the ICAPS 2026 constraint formulation for domain repair show that planning domains can be revised from external evidence. Our narrower contribution is decision-equivalent applicability repair with local exception/guard semantics, finite exact optimization, calibration-only deployment, and independent decision-risk evidence. We do not claim to introduce domain repair itself.

## 12. Limitations
Decision equivalence is distribution-relative and should not be interpreted as causal necessity. The finite edit vocabulary can omit useful repairs. AMLGym evaluation concerns applicability decisions only; effects and plan-solving performance are outside the external claim. Many frozen cells have empty held-out decision subsets, and five cells are failures/timeouts. The confirmatory sign test does not establish broad superiority, and the global gated baseline is not broadly beaten. The primary AMLGym run also predates explicit process-level RNG/hash pinning for the upstream learner implementations; this is now exposed and handled as a post-freeze reproducibility diagnostic rather than hidden by retrospective replacement.

## 13. Reproducibility
The repository contains the exact/soft contextual MILPs, hitting-set routines, identifiability diagnostics, certificates, controlled benchmark, stress suite, AMLGym bridge, frozen v4 contract, label-free preflight, per-cell runner, outcome-independent merger, and a regression test for duplicate case records embedded in shard summaries. The local release contains 36 tests. Heavy ROSAME/n=10 cases are orchestrated independently but retain the same frozen 900-second case limit. The post-freeze deterministic replay additionally pins process hash and learner RNG seeds; this amendment is deliberately separated from the primary frozen scientific aggregate.

## 14. Conclusion
Sequence regularity is not authorization necessity. A downstream repair layer can preserve an upstream model while making explicit, local, independently testable changes to its authorization surface. Controlled experiments establish exact recovery and certification properties. The external study gives a deliberately narrower result: conservative calibration-gated repair can avoid worsening observed held-out applicability decisions while producing improvements in a small subset of domains, but current domain-level evidence does not justify a universal performance claim.

## References
1. Le, T., Juba, B., Stern, R. Learning Safe Action Models with Partial Observability. AAAI, 2024.
2. Aineto, D., Scala, E. Action Model Learning with Guarantees. KR, 2024.
3. Lin, S. et al. Told You That Will Not Work: Optimal Corrections to Planning Domains Using Counter-Example Plans. AAAI, 2025.
4. Bercher, P., Sreedharan, S., Vallati, M. A Survey on Model Repair in AI Planning. IJCAI, 2025. doi:10.24963/ijcai.2025/1152.
5. Gösgens, R., Jansen, N., Geffner, H. Learning Lifted Action Models. KR, 2026. doi:10.24963/kr.2026/87.
6. Liu et al. Bridging the semantic gap: Trajectory-guided domain repair for reliable planning. Robotics and Computer-Integrated Manufacturing 101 (2026), 103290. doi:10.1016/j.rcim.2026.103290.
7. A Constraint Formulation for Domain Repair with Ground or Lifted Test Plans. ICAPS, 2026. doi:10.1609/icaps.v36i1.42810.
8. Clopper, C. J., Pearson, E. S. The Use of Confidence or Fiducial Limits Illustrated in the Case of the Binomial. Biometrika, 1934.
9. McNemar, Q. Note on the Sampling Error of the Difference between Correlated Proportions or Percentages. Psychometrika, 1947.
