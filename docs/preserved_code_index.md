# Preserved research code index

The maintained API is intentionally small. Research implementations that are useful for audit, continuity or future consolidation live under `research/reference_impl/` and `prototypes/runtime/` instead of being mixed into `src/procedural_ai/`.

This page distinguishes **code currently checked into the repository** from **research directions tracked by the roadmap**. A direction is not described as implemented here unless the corresponding module or runtime artifact is actually present on `main`.

## Checked-in reference implementations

### Constraint and semantic evidence

- `hardening_loro_graph.py` — dataset-native procedure graph utilities and frozen dependency extraction.
- `q1_decision_equivalent_graph.py` — comparison of dependency models at the decision level.
- `q1_semantic_sample_complexity.py` — evidence/sample-complexity analysis for semantic relations.
- `q1_semantic_version_space.py` — finite semantic model alternatives.
- `recording_bootstrap.py` — recording-level resampling utilities.

### Source-aware uncertainty and planning

- `q1_uncertainty_source_aware.py` — source-aware uncertainty analysis.
- `source_aware_resolution_planner.py` — exact finite-state resolution planner.
- `joint_semantic_physical_uncertainty.py` — joint physical/semantic uncertainty model.
- `p4_observation_model.py` — observation/reliability model used by preserved experiments.
- `e4_real_state_calibrated_observation.py` — calibrated observation experiment boundary.

### Certificates and authorization

- `evidence_certificate.py` — evidence-linked authorization certificate logic.
- `semantic_version_authorization.py` — authorization over alternative procedure models.

### Risk and runtime

- `risk_model.py` — preserved action-risk model boundary.
- `prototypes/runtime/vr_compiler/procedure_compiler.py` — procedure compiler/reachability work.
- `prototypes/runtime/v24_runtime/playable_runtime.py` — compact playable runtime.
- `prototypes/runtime/v24_runtime/playable_cli.py` — runtime CLI.
- `prototypes/runtime/replay_v24/PLAYABLE_ACTION_MAP_V24.json` — action mapping artifact.
- `prototypes/runtime/replay_v24/IMPACT_PLAYABLE_PROCEDURE_V24.json` — playable procedure artifact.
- `prototypes/runtime/replay_v22/PROGRESSIVE_SEMANTIC_RISK_CATALOG_V22.json` — compact semantic risk catalogue.

## Maintained experiment entry points

### Dependency validation

- `experiments/constraints/run_nested_calibration.py`
- `experiments/constraints/run_carrier_robustness.py`
- `experiments/constraints/run_prospective_role_transfer.py`

### Information-source selection

- `experiments/information_selection/run_bellman_vs_myopic.py`
- `experiments/information_selection/run_noisy_sources.py`
- `experiments/information_selection/run_cost_sensitivity.py`
- `experiments/information_selection/run_exact_scaling.py`
- `experiments/information_selection/run_reliability_sweep.py`

## Research directions retained in the roadmap

The project roadmap also tracks follow-on work on:

- richer rarefaction and rework-conditioned relation analysis;
- additional certificate soundness/context binding;
- real visual evidence routing and expensive-model escalation;
- online reliability/session adaptation;
- risk-aware selective assistance;
- cross-procedure transfer;
- progressive intent;
- edge/OpenXR execution and human-study instrumentation.

These directions are documented in `docs/research_roadmap.md`. They should be migrated into `research/reference_impl/` only with their actual source code and reproducibility contract; placeholder modules are intentionally not created.

## Rule for future migration

New reusable primitives go to `src/procedural_ai/`. Benchmark-specific experiment scripts go to `experiments/`. Historical or exploratory implementations may be preserved under `research/reference_impl/`, but the index must never claim that a module exists unless it is present in the repository.
