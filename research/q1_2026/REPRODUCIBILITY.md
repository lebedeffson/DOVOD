# Reproducibility contract

This directory is the working Q1 research release for two separate DOVOD papers. It lives on `q1/full-rebuild-20260905`; the frozen `main` baseline is not modified.

## Local two-paper core

Python 3.11+ is recommended.

```bash
python -m pip install -r requirements.txt
make release
```

`make release` runs the complete unit suite and regenerates the controlled Paper A benchmark/stress verification, the Paper B exact/POMCP/practical reports, and the recovered exact count-DP core. The clean release currently contains 36 tests.

Wall-clock values are machine-dependent. Exact reproducibility targets are values, selected actions, state counts, planted-edit recovery, theorem/formula checks, deterministic splits, and provenance hashes rather than absolute seconds.

## AMLGym confirmatory matrix

The confirmatory contract is `configs/amlgym_q1_contract.json` (schema `dovod-q1-amlgym-contract-v4`). It pins AMLGym 1.0.11, 20 IPC-style domains, four learner families, two trace budgets, the semantic-state split, the repair vocabulary, and the calibration-only deployment gate.

The canonical workflow is `.github/workflows/q1-amlgym-confirmatory.yml`.

It first runs a label-free preflight that replays the historically inspected pilot selector and proves zero semantic-state overlap before test labels are opened. Seven learner/budget combinations run as ordinary 20-domain shards. ROSAME with trace budget 10 is executed as 20 independent per-domain jobs because the sequential shard can exceed a hosted-runner wall-time limit. This changes orchestration only; every scientific argument and frozen split remains identical.

The ROSAME/n=10 jobs use the original frozen 900-second per-case limit and emit a case receipt whenever the process remains under workflow control. Thus a normal timeout or upstream failure remains one of the 160 outcomes rather than disappearing as a missing artifact. A hosted-runner cancellation can still preempt the shell before a receipt is written; such infrastructure cancellation is not re-labelled as a scientific timeout.

The merge step is outcome-agnostic: CI requires execution completeness and protocol integrity, never favorable scientific performance.

Canonical frozen aggregate:

`results/paper_a_amlgym_confirmatory_matrix.json`

The aggregate retains improved, tied, worsened, failed/timeout, and empty-test cells. Broad interpretation uses domain means and the exact domain-level sign test.

### Post-freeze RNG reproducibility amendment

A clean replay after the primary v4 run exposed an upstream reproducibility limitation: the DOVOD state/action selection and repair/calibration/test split are SHA-256 deterministic, but AMLGym learner execution was launched without an explicit process hash seed or common Python/NumPy/PyTorch RNG seed. Repeated learner executions can therefore produce different learned domains even when the scientific split and downstream DOVOD code are unchanged.

This is not repaired retroactively by choosing a more favorable replay. The original frozen v4 aggregate remains the primary confirmatory result. The post-freeze reproducibility amendment pins:

- `PYTHONHASHSEED=0` before the Python interpreter starts;
- `DOVOD_CONFIRMATORY_SEED=20260906` for Python and NumPy via opt-in `sitecustomize.py`;
- the same seed for PyTorch in ROSAME jobs.

The amendment changes neither domains, trace budgets, state fingerprints, repair/calibration/test buckets, repair vocabulary, deployment gate, metrics, nor the 900-second scientific case limit. It is reported as a reproducibility diagnostic rather than silently substituted for the frozen primary analysis. Any amended scientific aggregate is compared with, rather than selected over, the primary aggregate.

## Blue Birds external validation

`benchmarks/run_paper_b_bluebirds.py` pins `welinder/cubam` commit `fe5ba700f1adbb489c69af311558d64370d73d36` and deterministically hashes tasks into calibration/test partitions. Test ground truth is used only for evaluation.

The external result supports calibration-based reliability selection on held-out tasks; naive orientation flipping is retained as a negative result. It is not a procedural-action benchmark and does not by itself validate the full Bellman planner.

## Frozen procedural evidence boundary

The MECCANO/IMPACT numbers in `external_evidence/dovod_short_papers_v12.json` are provenance-preserving snapshots of previously frozen DOVOD evidence. Raw third-party datasets are not redistributed here. The MECCANO source-acquisition study uses controlled perfect reveals and therefore does not empirically identify persistent source orientation/reliability.

## Integrity

The authoritative clean validation is Git + CI plus the generated evidence JSONs. `MANIFEST.sha256` is generated from a concrete release tree with:

```bash
python scripts/build_manifest.py
```

It must be regenerated whenever release files change rather than treated as an immutable source of truth.
