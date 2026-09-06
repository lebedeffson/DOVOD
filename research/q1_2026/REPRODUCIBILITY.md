# Reproducibility contract

This directory is the working Q1 research release for two separate DOVOD papers. It is designed to be copied into `research/q1_2026/` of the parent DOVOD repository without modifying the frozen short-paper evidence on `main`.

## Local core

Python 3.11+ is recommended.

```bash
python -m pip install -r requirements.txt
make release
```

`make release` runs the full unit suite, regenerates the Paper A controlled benchmark, the Paper B orientation/planning benchmark, the reconstructed core scaling benchmark, and finally checks the fixed regression values and claim boundaries.

The wall-clock timing values are machine-dependent. The exact reproducibility targets are values, actions, state counts, synthetic planted edits, theorem-formula checks, and provenance hashes, not absolute seconds.

## External AMLGym gate

The external benchmark contract is frozen in `configs/amlgym_q1_contract.json` before external outcomes are read. It pins AMLGym 1.0.11, 20 IPC-derived domains, four learner families (SAM, OffLAM, NOLAM, ROSAME), two trace budgets, and a hash-defined repair/calibration/test split.

```bash
python -m pip install -r requirements.txt -r requirements-amlgym.txt
make amlgym-matrix
```

A single-domain smoke run is available as `make amlgym-smoke`. The parent repository workflow `.github/workflows/q1-research.yml` runs the full matrix in eight independent shards and merges all case JSON files while preserving failures and timeouts. A failed learner/domain cell is evidence about the protocol execution and must not be silently dropped.

The DOVOD repair layer is evaluated only on action applicability. It does not alter learned PDDL effects and it must not be described as improving plan solving without a separate experiment.

## External data boundary

The MECCANO/IMPACT numbers in `external_evidence/dovod_short_papers_v12.json` are a provenance-preserving snapshot of previously frozen DOVOD evidence. Raw third-party datasets are not redistributed in this package and those numbers are not presented as a fresh rerun.

## Integrity

`MANIFEST.sha256` hashes every release source/document/result file except itself and ephemeral caches. Regenerate it with:

```bash
python scripts/build_manifest.py
```
