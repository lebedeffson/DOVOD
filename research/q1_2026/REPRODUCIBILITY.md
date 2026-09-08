# Reproducibility contract

This directory is the working Q1 research release for two separate DOVOD papers. It lives on `q1/full-rebuild-20260905`; the frozen `main` baseline is not modified.

## Local two-paper core

Python 3.11+ is recommended.

```bash
python -m pip install -r requirements.txt
make release
```

`make release` runs the complete local unit suite and regenerates the controlled Paper A benchmark/stress verification, Paper B exact/POMCP/practical reports, and the recovered exact count-DP core. The repository-level `tests` workflow additionally runs full `pytest`, reference-result verification, and public-repository integrity checks on Python 3.10, 3.11, and 3.12.

Wall-clock values are machine-dependent. Exact reproducibility targets are values, selected actions, state counts, planted-edit recovery, theorem/formula checks, deterministic splits, and provenance hashes rather than absolute seconds.

## Paper A frozen AMLGym confirmatory matrix

The confirmatory contract is `configs/amlgym_q1_contract.json` (schema `dovod-q1-amlgym-contract-v4`). It pins AMLGym 1.0.11, 20 IPC-style domains, four learner families, two trace budgets, the semantic-state split, the repair vocabulary, and the calibration-only deployment gate.

The canonical workflow is `.github/workflows/q1-amlgym-confirmatory.yml`. It runs a label-free preflight that replays the historically inspected pilot selector and verifies zero semantic-state overlap before confirmatory test labels are used. Heavy ROSAME/n=10 cases are independently orchestrated under the same frozen 900-second scientific case limit. Orchestration changes do not alter the scientific contract.

The merge is outcome-agnostic: CI requires structural completeness and protocol integrity, never favorable scientific performance. The canonical frozen aggregate is `results/paper_a_amlgym_confirmatory_matrix.json` and retains improved, tied, worsened, failed/timeout, and empty-test cells.

Primary frozen result: 160/160 cells, 5 retained failures/timeouts, 80 empty-test cells, 75 usable cells, `7/68/0` usable-cell wins/ties/losses versus upstream, domain means `4/6/0`, exact two-sided sign-test `p=0.125`. This primary result is never replaced by a later replay.

### Post-freeze RNG reproducibility amendment

The DOVOD state/action selection and repair/calibration/test split are SHA-256 deterministic, but the first AMLGym run predated explicit process hash and common Python/NumPy/PyTorch RNG pinning. The reproducibility amendment therefore pins `PYTHONHASHSEED=0` plus the declared process seeds while leaving domains, budgets, fingerprints, repair vocabulary, deployment gate, metrics, and scientific timeout unchanged.

The seeded replay is reported as a diagnostic: 75 usable cells, `6/69/0`, domain `3/7/0`, `p=0.25`. The observed `sokoban|ROSAME|10` hosted-runner artifact loss is narrowly allowlisted as `infrastructure_missing`; any other absent confirmatory cell remains fatal. A repeated seeded `barman/ROSAME/10` case reproduces all scientific fields exactly while runtime fields vary as expected.

### Paper A V5 post-confirmatory evidence

V5 does not change or reopen the frozen confirmatory analysis.

`results/paper_a_amlgym_simple_baselines_v5.json` is a compact tracked receipt for workflow run `34137791383` / artifact `10025537361`. The full 160-cell panel compares DOVOD with random and frequency at-most-one-edit heuristics under the same repair/calibration/test partition and calibration gate. It is explicitly post-confirmatory and descriptive.

`results/paper_a_assembly101_ordering_audit_v5.json` is a compact tracked receipt for Assembly101 workflow run `34139277922` / artifact `10025229855`, pinned to source commit `6f3a953267ffb86cdeabf6751af05a75a011a4f8`. It tests whether high-frequency predecessor relations survive held-out labeled-correct behavior and whether their violations are selective for explicit ordering mistakes. It is a falsification/claim-boundary audit, not validation of mechanical necessity or the full DOVOD repair layer.

Narrative provenance and interpretation are in `docs/PAPER_A_V5_EVIDENCE.md`.

## Paper B evidence separation

Paper B keeps four evidence regimes distinct:

1. controlled exact/approximation validation for the static binary-evidence model;
2. frozen MECCANO procedural acquisition evidence with controlled perfect reveals;
3. official-split IMPACT PSR negative lookahead benchmark and cost sensitivity;
4. Blue Birds held-out source-calibration evidence.

Machine-readable receipts include `paper_b_meccano_gain_summary_v4.json`, `paper_b_impact_psr_summary_v4.json`, `paper_b_scalability_robustness_summary.json`, and `paper_b_bluebirds_external_summary.json`.

The Blue Birds result supports calibration-based source selection only. It is not a procedural-action benchmark and does not validate the full Bellman planner. The MECCANO acquisition replay uses controlled perfect reveals and therefore does not empirically identify real persistent source orientation/reliability.

## External-source pinning

- AMLGym: version 1.0.11 plus frozen contract;
- IMPACT PSR: source commit `4fed5faa5f05f7aece55712e458defa1f372b248`;
- Blue Birds / `welinder/cubam`: commit `fe5ba700f1adbb489c69af311558d64370d73d36`;
- Assembly101 mistake annotations: commit `6f3a953267ffb86cdeabf6751af05a75a011a4f8`.

Raw third-party datasets are not redistributed unless their licensing/source contract permits it; tracked receipts preserve source commits and evaluation provenance.

## Integrity

The authoritative validation is Git + CI + generated/trackable evidence receipts. `MANIFEST.sha256` is generated from a concrete release tree with:

```bash
python scripts/build_manifest.py
```

It must be regenerated whenever release files change rather than treated as immutable source truth.
