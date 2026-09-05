# DOVOD short papers — 2026 public context

This document records the scientific and submission context for the two final DOVOD short papers without storing manuscript DOCX/PDF files or private author data in Git.

## Target venue

Both final v12 manuscripts are prepared for the **All-Russian Scientific and Technical Conference “Zolotov Readings 2026”**, Tver State Technical University, Tver, Russia, **Section 3 — Artificial Intelligence and Decision Making**.

Official conference page checked on 2026-09-05: `https://new.tstu.tver.ru/science/conference/Zolotov2026/`.

Current public requirements:

- application deadline: **10 September 2026**;
- conference dates: **15–16 October 2026**;
- venue: Tver State Technical University, Tver, Russia;
- application is sent to V. K. Kemaykin with the note/subject “Zolotov Readings” at `vk-kem@mail.ru`;
- application information: participant names, workplace/position, phone/e-mail, title, abstract, **theses up to 2 pages**, and lodging requirement;
- Section 3 is “Artificial Intelligence and Decision Making”;
- subsequent journal publication is planned in *Software & Systems* or the *Herald of Tver State Technical University* under their separate post-conference requirements.

The repository deliberately excludes conference forms, phone numbers, private author data and administrative correspondence.

## Paper A — prerequisite falsification

**Russian title:** “Отличение обязательных условий от привычного порядка действий в технологических процедурах”.

### Problem

A learner of procedural demonstrations can confuse frequent action order with a mandatory prerequisite. That may be useful for next-step prediction but is too strong for authorization: an empirically common order can falsely block a successful action.

### Core rule

For a candidate unary same-state prerequisite `p -> a`, DOVOD searches for a successful observed execution of action `a` while predicate `p` is absent. Let `C(p,a)` be the set of independent evidence carriers containing such a counterexample.

- `C(p,a) != empty` falsifies the universal empirical prerequisite hypothesis.
- `C(p,a) = empty` means only **unfalsified in the available evidence**.
- A positive claim of mechanical necessity still requires an authoritative manual/SOP, engineering constraint, expert adjudication, or controlled experiment.

### Final v12 evidence

- 272 candidate directed relations among 17 component-state variables;
- MECCANO TRAIN: **201/272** directly refuted relations;
- IMPACT Reassembly-A: **259/272** directly refuted relations;
- nested MECCANO calibration: held-recording action-authorization recall **0.8707 -> 0.8853**, gain about **0.0145**, over 110 fit/calibration pairs;
- one calibration recording removes **3.85** candidate restrictions on average;
- one-carrier deletion robustness: **162/201** MECCANO and **184/259** IMPACT;
- rework-only refutations: **16** MECCANO and **11** IMPACT;
- a hard `>=2` carrier rule is too conservative in the MECCANO LORO stress: **0.9013 -> 0.7884** recall;
- finite-population rarefaction: 95% expected recovery of the already observed refutation set needs **9/11** MECCANO recordings and **11/13** IMPACT participants;
- under an independent zero-failure Bernoulli assumption, the one-sided 95% upper bound is about **28.31%** at `n=9` and **23.84%** at `n=11`; **59** zero-failure observations are needed for the bound to fall below 5%.

### Novelty and boundary

The contribution is an asymmetric evidence rule for procedural authorization: a direct successful counterexample can falsify a universal restriction, while lack of a counterexample is deliberately prevented from becoming proof of necessity. Carrier identity, nested calibration, rework evidence and rarefaction make the result auditable.

The current short paper concerns unary, unconditional, same-state prerequisites. It does not claim to solve conjunctive, conditional, temporal, resource, causal or authoritative mechanical prerequisite learning.

## Paper B — source-aware information acquisition

**Russian title:** “Выбор дополнительной информации перед следующим действием: состояние объекта или правило процедуры”.

### Problem

Low confidence can have two different causes:

1. the **physical state** is unknown, so another observation/sensor may help;
2. the **procedure semantics** are uncertain, so another frame may be useless and the system needs a rule/manual/expert review.

The decision-support problem is therefore not simply “request more information”, but “choose the information source that can resolve the current decision”.

### Core exact model

The confirmatory Bellman model uses perfect binary reveal for physical queries and a finite semantic version space for rule alternatives. Each information action has a normalized cost. For knowledge state `s`:

`V(s) = min_{z in Z(s)} [ c(z) + E_o V(T(s,z,o)) ]`.

The strong myopic baseline maximizes immediate expected uncertainty reduction per unit cost. Source reliability errors are tested in a **separate one-step stress**, not silently inserted into the exact recursion.

### Final v12 evidence

- 777 controlled MECCANO episodes;
- 187 episodes contain simultaneous physical and semantic uncertainty;
- semantic cost 1: exact Bellman **1.6657** vs gain-per-cost myopic **1.7380**, improvement **4.16%**;
- fixed baselines: physical-first **1.7845**, semantic-first **2.0680**;
- recording-cluster 95% interval for Bellman-minus-myopic expected cost: **[-0.0799, -0.0655]**;
- at semantic cost 5, exact and myopic expected costs coincide;
- reliability stress `r_physical=0.55`, `r_semantic=0.85`: first-source type changes in **33.15%** of mixed episodes and semantic-first rate rises **6.95% -> 40.11%**;
- cost misspecification: assuming semantic cost 1 when the true cost is 5 or 10 gives mean relative regret **24.76%** and **44.29%**;
- prevalidation at semantic cost 2: downstream runtime cost **1.466206 -> 1.245739**; charging 6 one-time normalized units for three reviews and amortizing over 777 episodes gives total **1.253461**, still **14.51%** below baseline;
- exact-planner scaling at `k=10`: **157464..531414** memoized states for 2..8 semantic alternatives;
- analytical envelope for the current perfect-reveal model: `N_upper(k,g)=(g+1)*3^k`; at `k=10,g=8`, **531414 <= 531441**.

### Hybrid policy, novelty and boundary

The maintained short-paper layer includes an exact-to-myopic guard: use exact Bellman planning while the conservative state envelope is below a configured threshold; otherwise fall back to the gain-per-cost myopic selector.

The novelty is not the Bellman equation itself. It is the explicit factorization of uncertainty by **source**, information acquisition as the action space, cost/reliability stress around that decision, and a bounded exact-to-myopic implementation regime tied to the semantic version space and physical knowledge state.

The exact recursion does not yet contain repeated correlated/noisy observations, empirically calibrated sensor/expert reliability, measured human cost, or a semantic reviewer that can introduce a rule outside the frozen version space. External replication of the full source-selection policy on another procedure remains future work.

## Reproducibility boundary

The compact short-paper release maps the final v12 claims to code and frozen evidence. Raw MECCANO/IMPACT material is not redistributed. Full raw-data Level-3 reruns remain a separate gate requiring local access to the original third-party datasets.
