# Reproducibility

## Quick verification without third-party data

`python scripts/verify_reference_results.py`

This validates the frozen result snapshots shipped with the repository.

## Full rerun

Obtain the required datasets under their original terms, then place the MECCANO PSR archive under `data/external/` (or set `PROCEDURAL_AI_MECCANO_PSR_ZIP`).

Run:

```bash
bash scripts/reproduce_core.sh
```

The experiment scripts are retained from the reviewer-hardened research implementation. The stable reusable API lives under `src/procedural_ai/`; historical research implementations and preserved directions live under `research/reference_impl/`.
