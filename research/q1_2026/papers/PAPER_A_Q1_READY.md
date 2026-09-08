# Decision-Equivalent and Certified Repair of Learned Action Preconditions

## Q1-ready manuscript snapshot — 8 September 2026

## Abstract
Action models learned from procedural demonstrations can convert recurrent ordering patterns into hard applicability restrictions even when the data do not establish mechanical necessity. Deleting such restrictions globally can repair false blocks but can also create false allows. We study a downstream, learner-agnostic authorization-repair problem: which learned restrictions should remain, where should context-local exceptions be admitted, and where are local guards needed? Positive-only traces do not identify mechanical necessity. For a finite prerequisite family, minimum empirical decision-equivalent prerequisite selection reduces exactly to Minimum Hitting Set. We then introduce a bounded edit language of contextual prerequisite exceptions and guards and optimize exact or weighted-soft mixed-integer objectives over authorization decisions, without modifying learned action effects. In a controlled eight-bit benchmark, the exact solver recovers all three planted local edits, achieves zero complete-support risk versus 0.125 for both the upstream rule and the best global-deletion baseline, and makes 0 errors on an independent 4,096-decision holdout, giving a one-sided exact 95% upper risk bound of 7.31e-4. Under 5% label corruption at 640 observations, mean complete-support risk is 0.00156 and maximum risk is 0.00781. A harder 16-bit two-action study uses overlapping action contexts and 160 candidate edits per action. At 1,024 training observations, all six action-by-seed fits recover the planted repair structure exactly and have zero full-support risk; smaller samples intentionally expose incomplete structural recovery rather than being hidden. External validation uses a frozen AMLGym 1.0.11 protocol over 20 IPC-style domains, four learner families, and two trace budgets. All 160 prespecified cells are retained; 75 contain usable held-out decisions, with 7 improvements, 68 ties, and 0 degradations relative to upstream applicability. At the prespecified domain level the result is 4 wins, 6 ties, and 0 losses (two-sided sign-test p=0.125), so we do not claim broad learner-wide superiority. A post-confirmatory engineering drill-down on the four domains with positive frozen domain means shows the selective mechanism: 216 operator surfaces reduce to 13 counterexample-flagged surfaces, seven calibration-approved non-empty repairs, and six explicit abstentions; over 4,448 held-out decisions, errors fall from 90 upstream to 20 under the deployed local repairs, whereas a deliberately coarse 80%-threshold global override improves 0/32 cells. Independent MECCANO and IMPACT audits reduce a 272-relation prerequisite review queue to 71 and 13 unresolved hypotheses. Carrier-coverage analysis shows that 95% of expected observed refutations are reached by 9/11 MECCANO recordings and 11/13 IMPACT participants, while a seemingly conservative two-carrier hard threshold lowers downstream recall from 0.9013 to 0.7884. The contribution is an auditable selective-repair workflow with explicit abstention and independent decision-risk evidence; it is not causal prerequisite discovery or autonomous safety authorization.

## 1. Introduction
A repeated action order is not the same as a necessary precondition. A technician may inspect before tightening because of policy, training, convenience, interface layout, or data-collection bias. If a learner converts that regularity into a hard precondition, a planner can false-block a valid action. Conversely, globally deleting the precondition can authorize unsafe or unsupported states.

We separate action-model learning from authorization repair. The upstream learner remains unchanged. The downstream layer asks a narrower question:

**Which learned applicability restrictions should remain in the authorization surface, and in which observed contexts should they be locally relaxed or strengthened?**

This framing avoids two overclaims. First, positive demonstrations do not identify mechanical necessity. Second, the goal is not to relearn complete action models or effects. The goal is decision equivalence relative to a declared state distribution and finite edit vocabulary, with independent evaluation and the option to abstain.

Our contributions are:

1. a positive-only non-identifiability argument for prerequisite necessity and an explicit absence-of-counterexamples sample bound under a declared minimum counterexample mass;
2. an exact Minimum Hitting Set formulation for finite decision-equivalent prerequisite selection, including mandatory/optional/redundant optimum-role diagnostics;
3. a context-local edit language containing prerequisite exceptions and guards, optimized by exact or weighted-soft MILP;
4. independent decision-risk certification using exact binomial bounds, paired tests, and finite-class certificates;
5. controlled recovery, noise, and two-action 16-bit scaling studies that expose sample complexity rather than reporting only favorable sizes;
6. a frozen external AMLGym protocol that separates repair fitting, calibration-only deployment gating, and held-out test evaluation by semantic-state fingerprint;
7. a practical audit layer over MECCANO and IMPACT that quantifies falsification, unresolved residue, carrier coverage, and the cost of overly conservative multi-carrier thresholds.

## 2. Positive demonstrations do not identify necessity
Let action a have candidate prerequisites P={p1,...,pm}. A positive observation is a state in which a is executed successfully. If all positive observations satisfy pi, the same observations are compatible with at least two worlds: one where pi is mechanically necessary and another where the demonstrator policy simply never visits a violating state before executing a. Positive-only sequence evidence therefore cannot identify necessity without extra assumptions or interventions.

This motivates a weaker operational target: authorization decisions on observed or explicitly sampled states. A recorded successful state violating pi is a counterexample to the universal empirical prerequisite claim. The absence of such a state is not proof of necessity.

If a false prerequisite has counterexample probability mass at least q under the observation distribution, n independent positive observations miss every counterexample with probability at most (1-q)^n. To make this probability at most delta,

n >= log(delta) / log(1-q).

For q=0.05 and delta=0.05, n=59 is sufficient. Without a positive lower bound on q, no finite absence guarantee exists.

## 3. Decision-equivalent prerequisite selection
For each observed negative authorization state x_j, let V_j be the set of candidate prerequisites violated in that state. A retained prerequisite subset S preserves every observed block iff S intersects every V_j. Minimum empirical decision-equivalent selection is therefore

min sum_i z_i
subject to sum_{p_i in V_j} z_i >= 1 for all j,
z_i in {0,1}.

This is Minimum Hitting Set and is NP-hard. The family of minimum solutions yields three decision-relative roles:

- mandatory-optimal: present in every minimum solution;
- optional-optimal: present in at least one but not all minimum solutions;
- redundant-relative-to-observed-decisions: present in no minimum solution.

These roles describe equivalence to observed authorization decisions, not physical causality.

## 4. Contextual repair language
Global deletion is often too coarse. We use two edit types over a finite, predeclared context-literal vocabulary.

An exception E=(p,C) waives learned prerequisite p only when context conjunction C holds. A guard G=C blocks the action when C holds even if the upstream prerequisites otherwise permit it. Context width is bounded and the candidate edit vocabulary is frozen before fitting labels are inspected.

For selected edit set R, authorization is evaluated directly on each labeled state. Action effects, transition semantics, and the upstream learner remain untouched. The method is therefore a decision-layer repair, not a new action-model learner.

## 5. Exact and soft optimization
Let e_k be candidate edits and z_k binary selection variables. In a noiseless setting we minimize weighted edit count subject to exact agreement with observed authorization labels. Under label noise, error indicators xi_j are introduced and the objective is

sum_k w_k z_k + lambda_+ sum_{j:y_j=1} xi_j + lambda_- sum_{j:y_j=0} xi_j.

Different penalties can encode asymmetric false-allow and false-block costs. The optimization is globally exact relative to the declared finite vocabulary, not relative to an unknown causal rule class.

## 6. Certification
### 6.1 Independent exact holdout
With k errors in n independent held-out decisions we report a one-sided Clopper-Pearson upper bound. For zero errors,

R <= 1 - alpha^(1/n).

At n=4096 and alpha=0.05 this is 0.0007311.

### 6.2 Paired baseline comparison
Upstream and repaired rules are evaluated on the same decisions. Discordant outcomes are therefore compared with an exact paired McNemar/binomial calculation rather than an unpaired test.

### 6.3 Finite-class certificate
If a frozen vocabulary contains M edits and at most s can be selected, the number of candidate subsets is bounded by

H = sum_{i=0}^s C(M,i).

For the main controlled benchmark, M=128 and s=3, so H=349,633. The resulting finite-class zero-training-error bound is retained as a secondary certificate; independent held-out evidence is primary.

## 7. Controlled recovery and sample complexity
### 7.1 Eight-bit planted benchmark
The base controlled problem uses eight binary state variables and the complete 256-state support. The upstream rule requires variables 0,1,2. Ground truth differs by three local edits: two contextual prerequisite exceptions and one guard. A width-two edit language over four context variables creates 128 candidate edits.

The exact MILP is fitted on 640 sampled authorization observations. It recovers exactly the three planted edits and fits all training decisions. On complete support, repaired risk is 0, while the upstream rule and the best global-deletion subset both have risk 0.125.

On an independently sampled 4,096-decision holdout, the repaired rule makes 0 errors. The one-sided exact 95% upper risk bound is 0.0007311. The upstream rule makes 514 errors; the paired exact p-value is 3.73e-155. Both held-out false-allow and false-block subsets contain zero repaired errors.

### 7.2 Noise stress
Training size is varied over 64,128,256,640 under 0% and 5% label corruption across five seeds. Small samples sometimes miss local contexts, which is the expected failure mode. At 256 and 640 noise-free observations all five seeds recover the planted structure exactly and have zero complete-support risk. At n=640 with 5% corruption, the weighted-soft solver has mean complete-support risk 0.0015625 and maximum 0.0078125; four of five seeds recover the exact planted structure.

### 7.3 Two-action 16-bit scaling study
A stronger controlled study increases the state description to 16 bits, introduces two overlapping action-authorization problems, assigns each action a 160-edit candidate vocabulary, and plants four contextual edits per action. Each action is fitted independently because the method deliberately repairs authorization surfaces rather than learning a joint transition model.

The study reports all three fixed seeds at training sizes 256,512,1,024. For action A, mean contextual full-support risk is 0 at n=256, 0.003906 at n=512, and 0 at n=1,024; exact structural recovery rates are 1.0, 2/3, and 1.0. For action B, mean risk is 0.010417 at n=256 and 0 at n=512 and n=1,024; exact recovery rates are 1/3, 1.0, and 1.0. At n=1,024, all six action-by-seed fits recover the planted edit structure exactly with zero full-support risk. Mean fit time is approximately 1.23 seconds for action A and 1.16 seconds for action B on the recorded CI runner.

The smaller-n imperfections are scientifically useful: they show finite-sample structural ambiguity rather than an artificially monotone success story. This remains an action-local finite-vocabulary benchmark and is not claimed as general large-domain MILP scalability.

## 8. Frozen AMLGym external validation
### 8.1 Protocol
The external study uses AMLGym 1.0.11 with 20 IPC-style domains, four learner families (SAM, OffLAM, NOLAM, ROSAME), and two trace budgets (3 and 10): 160 prespecified cells.

An earlier pilot is used only to design the protocol. Before confirmatory labels are consumed, historical pilot states are replayed and removed by semantic fingerprint. Remaining states are ranked by a new SHA-256 key. A fixed semantic-state hash partitions states into repair [0,500), calibration [500,750), and test [750,1000). Repairs are fitted only on repair states. Deployment is selected per operator using calibration data only: a non-empty repair is deployed only if it strictly reduces calibration risk and does not increase calibration false allows; ties revert to upstream. Held-out test labels do not affect fitting, gating, hyperparameters, or CI success.

A label-free preflight verifies zero semantic overlap between historical pilot states and confirmatory states in all 20 domains.

### 8.2 Retained failures and empty test cells
All 160 cells remain in the canonical matrix. Four upstream learner failures are retained on `childsnack`: OffLAM raises `KeyError: kitchen` at both budgets, and NOLAM encounters an upstream PDDL parser syntax error on `(xist ?param_1)` at both budgets. `sokoban/ROSAME/10` exceeds the frozen 900-second scientific limit in the primary run and is retained as a timeout.

Among successful cells, 80 have empty held-out decision subsets under the frozen split and remain explicitly empty. Seventy-five cells contain usable held-out authorization decisions.

### 8.3 Prespecified scientific result
Across the 75 usable cells, DOVOD improves 7, ties 68, and worsens 0 relative to upstream applicability. Mean cell-level risk reduction is 0.00829365. At the prespecified domain level, mean-risk comparison yields 4 wins, 6 ties, and 0 losses, with exact two-sided sign-test p=0.125. Positive frozen domain means occur in `barman`, `floortile`, `parking`, and `tpp`.

Against a calibration-gated global-override baseline, the domain comparison is 1 win, 8 ties, and 1 loss with p=1.0. DOVOD is slightly worse in `barman` and better in `floortile`. The correct interpretation is therefore conservative selective deployment: no usable held-out cell is worsened versus upstream, but the confirmatory study does not establish broad superiority across learner/domain combinations.

### 8.4 Reproducibility amendment
A later clean replay showed that the upstream AMLGym learner processes had originally been launched without globally pinned Python hash order and common Python/NumPy/PyTorch seeds. The frozen primary result is not replaced by a more convenient replay. A post-freeze amendment pins `PYTHONHASHSEED=0`, Python/NumPy seed 20260906, and the same PyTorch seed for ROSAME while preserving domains, budgets, split fingerprints, repair vocabulary, deployment gate, metrics, and the 900-second scientific limit.

The seeded replay produces 159 per-case artifacts; the hosted runner executing `sokoban/ROSAME/10` shuts down before the scientific limit, so that cell is recorded narrowly as `infrastructure_missing`. The 75 usable seeded cells contain 6 improvements, 69 ties, and 0 worsenings; domain level is 3 wins, 7 ties, 0 losses with p=0.25. A repeated nontrivial `barman/ROSAME/10` seeded run reproduces every scientific field exactly; only wall-clock fields differ. Only the observed sokoban cell is allowlisted for infrastructure accounting; any other missing cell remains fatal.

### 8.5 Post-confirmatory mechanism drill-down
After the primary confirmatory result was frozen, the four domains with positive frozen domain means were selected for explanatory engineering analysis. Because this selection uses confirmatory outcomes, the block is descriptive and is not a second significance test.

Across 32 learner/budget cells, there are 216 operator applicability surfaces. Only 13 are flagged by an observed applicability counterexample. Seven receive a calibration-approved candidate containing at least one edit; six remain unresolved and abstain to upstream. Across 4,448 held-out decisions, upstream makes 90 errors (62 false allows, 28 false blocks), whereas deployed DOVOD decisions make 20 (11 false allows, 9 false blocks). At the cell level this is 6 improvements, 26 ties, and 0 degradations, with mean risk reduction 0.01683.

A deliberately coarse comparator globally overrides a restriction when at least 80% of repair observations support the action. Both raw and calibration-gated variants yield 0 wins, 32 ties, and 0 losses versus upstream and no aggregate held-out reduction. Thus the local-repair mechanism in this selected block is not reproduced by this simple global threshold rule.

## 9. Real procedural prerequisite auditing
The repair method is motivated by a separate question: how many apparently universal prerequisite relations survive counterexample search in real procedural traces?

For 17 procedural components there are 272 directed unary candidate relations. A successful counterexample only falsifies the universal empirical same-state prerequisite claim; a relation without a counterexample remains unresolved rather than validated.

MECCANO refutes 201/272 relations (73.9%) and leaves 71 unresolved. Of the 201 refutations, 162 are supported by at least two independent recordings. IMPACT refutes 259/272 (95.2%) and leaves 13 unresolved; 184 are supported by at least two independent participants. The unresolved review queue therefore shrinks by factors of 3.83x and 20.92x relative to reviewing all 272 candidates. These are counts of hypotheses removed from review, not estimates of human hours saved.

### 9.1 Carrier coverage and staged auditing
A carrier-level rarefaction analysis asks how many independent recordings or participants are needed to recover most of the refutations observed in the full corpus. For MECCANO, 90% of expected observed refutations is reached by 7 of 11 recordings and 95% by 9 of 11. For IMPACT, 90% is reached by 9 of 13 participants and 95% by 11 of 13. Leaving out one random carrier retains an expected 98.24% of observed MECCANO refutations and 97.77% of IMPACT refutations.

This supports staged auditing: many candidate universal restrictions can be falsified before every carrier is processed, while the remaining unresolved set is explicitly retained for further evidence or expert review.

### 9.2 Why “require two carriers” is not automatically safer
A natural conservative heuristic accepts a counterexample only when at least two independent carriers support it. On the frozen downstream MECCANO leave-one-recording-out decision task, however, recall falls from 0.901308 under the one-carrier rule to 0.788363 under the two-or-more rule. This is a useful negative result: increasing evidential redundancy can hurt the downstream decision objective if rare but valid exceptions are suppressed.

A nested held-recording calibration analysis provides a better pattern. Across 110 pairs, recall rises from 0.870735 before calibration to 0.885278 after calibration, a gain of 0.014544 with bootstrap interval [0.010633,0.019160], while pruning an average of 3.85 candidate restrictions per pair. Falsification and deployment calibration should therefore remain separate stages.

## 10. Practical workflow
The evidence supports a selective workflow rather than autonomous repair.

1. Treat sequence regularity as a candidate restriction, not a necessity fact.
2. Search for successful counterexamples and eliminate only universal empirical claims that are actually contradicted.
3. Keep unresolved restrictions unresolved; absence is not validation.
4. Generate a bounded contextual edit vocabulary and solve exact or weighted-soft repair only on the repair split.
5. Deploy per operator only when calibration data improve decision risk without increasing the declared false-allow constraint; otherwise abstain to the upstream model.
6. Evaluate held-out applicability decisions independently and report empty cells, upstream tool failures, and infrastructure failures rather than silently removing them.
7. Use carrier coverage to plan data/audit effort, but do not impose a hard multi-carrier threshold unless its downstream effect has been validated.

## 11. Relation to prior work
Safe action-model learning under partial observability, including Le, Juba, and Stern (AAAI 2024), and action-model learning with guarantees such as Aineto and Scala (KR 2024), act at the model-learning stage. DOVOD is downstream and learner-agnostic.

Planning-domain repair is the closest neighboring line. Counterexample-based correction, planning-model repair surveys, trajectory-guided repair, and constraint formulations demonstrate that symbolic models can be revised from external evidence. Our narrower contribution is decision-equivalent applicability repair with context-local exception/guard semantics, a finite exact optimization target, calibration-only deployment, explicit abstention, and independent held-out decision-risk evidence. We do not claim to introduce domain repair itself.

## 12. Limitations
Decision equivalence is distribution-relative and must not be interpreted as causal or mechanical necessity. A finite edit vocabulary can omit useful repairs. The method changes applicability decisions only; external AMLGym evaluation does not establish improvement in learned effects or complete plan-solving performance.

The frozen AMLGym result is conservative: only 75/160 cells contain usable held-out decisions, five cells are failures/timeouts, and the prespecified domain-level sign test is p=0.125. The global gated comparator is not broadly beaten. The 32-cell mechanism drill-down is selected after observing positive frozen domain means and is explanatory, not an unbiased generalization estimate.

The 16-bit two-action benchmark remains action-local with finite candidate vocabularies and does not establish industrial-scale MILP performance. MECCANO/IMPACT audit reductions count falsified/unresolved hypotheses rather than expert time or safety impact. Carrier rarefaction describes recovery of the observed corpus refutations, not an unseen-population guarantee.

Finally, the primary AMLGym run predates process-level hash/RNG pinning for upstream learners. The seeded replay is preserved as a post-freeze reproducibility diagnostic rather than used to replace the prespecified primary result.

## 13. Reproducibility
The repository contains the hitting-set analysis, exact/soft contextual MILPs, identifiability and certificate code, controlled eight-bit benchmark, noise stress, 16-bit two-action benchmark, AMLGym bridge and frozen v4 contract, label-free pilot-overlap preflight, outcome-independent per-cell merge logic, seeded replay diagnostics, post-confirmatory drill-down, and MECCANO/IMPACT audit receipts.

The final Q1 practical workflow successfully runs the two-action 16-bit benchmark together with the companion Paper B final validation. Standard repository CI passes on Python 3.10, 3.11, and 3.12. The full Q1 research-core workflow and its external Blue Birds job are green on the finalized code line. Machine-readable result files preserve favorable and unfavorable sample sizes, upstream learner failures, empty held-out subsets, abstentions, and post-freeze diagnostics rather than collapsing them into a single success metric.

## 14. Conclusion
Procedural sequence regularity should not be promoted automatically to action-authorization necessity. A downstream decision-equivalent repair layer can preserve the upstream learner while making explicit, local, independently testable changes to applicability decisions. Exact controlled studies establish recoverability and certification properties; the two-action 16-bit extension shows that exact structural recovery reappears consistently once the sample supports the planted local contexts. Real MECCANO and IMPACT traces remove large fractions of universal prerequisite hypotheses before expert adjudication, and carrier analysis shows that most observed refutations can be recovered without processing every carrier. The frozen AMLGym confirmatory study supports a bounded claim: calibration-gated local repair improves a small subset of held-out cells and degrades none of the 75 usable cells, but domain-level evidence is not broad enough to claim universal superiority. The post-confirmatory drill-down explains the mechanism without upgrading that claim: only 13 of 216 surfaces need intervention, seven receive non-empty repairs, six abstain, and a coarse global threshold fails to reproduce the local improvements. The contribution is therefore a conservative, auditable transition from sequence-derived hypotheses to a smaller unresolved set and selectively deployed context-local authorization repairs—not causal prerequisite discovery and not autonomous safety authorization.

## References
1. Le, T., Juba, B., Stern, R. Learning Safe Action Models with Partial Observability. *AAAI*, 2024.
2. Aineto, D., Scala, E. Action Model Learning with Guarantees. *KR*, 2024.
3. Lin, S. et al. Told You That Will Not Work: Optimal Corrections to Planning Domains Using Counter-Example Plans. *AAAI*, 2025.
4. Bercher, P., Sreedharan, S., Vallati, M. A Survey on Model Repair in AI Planning. *IJCAI*, 2025. doi:10.24963/ijcai.2025/1152.
5. Gösgens, R., Jansen, N., Geffner, H. Learning Lifted Action Models. *KR*, 2026. doi:10.24963/kr.2026/87.
6. Liu et al. Bridging the semantic gap: Trajectory-guided domain repair for reliable planning. *Robotics and Computer-Integrated Manufacturing* 101 (2026), 103290. doi:10.1016/j.rcim.2026.103290.
7. A Constraint Formulation for Domain Repair with Ground or Lifted Test Plans. *ICAPS*, 2026. doi:10.1609/icaps.v36i1.42810.
8. Clopper, C. J., Pearson, E. S. The Use of Confidence or Fiducial Limits Illustrated in the Case of the Binomial. *Biometrika*, 1934.
9. McNemar, Q. Note on the Sampling Error of the Difference between Correlated Proportions or Percentages. *Psychometrika*, 1947.
