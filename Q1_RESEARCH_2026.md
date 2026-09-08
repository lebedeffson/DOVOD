# DOVOD Q1 research 2026

The frozen `main` baseline remains unchanged. The reproducible Q1 expansion is under [`research/q1_2026/`](research/q1_2026/README.md) on branch `q1/full-rebuild-20260905`.

It contains two separate research lines:

1. **Paper A — From Sequence Regularity to Action Authorization: Decision-Equivalent and Certified Repair of Learned Preconditions.** Downstream applicability repair, certification, frozen AMLGym confirmatory evidence, post-confirmatory simple baselines, and an Assembly101 held-out ordering audit.
2. **Paper B — Exact Evidence-Count Dynamic Programming and Cost-Robust Information Acquisition for Static Procedural Decisions.** Exact static evidence-count Bellman planning, approximation validation, positive/negative procedural regimes, cost robustness, and separately bounded source-calibration evidence.

Run the local two-paper core with:

```bash
cd research/q1_2026
python -m pip install -r requirements.txt
make release
```

The normal repository CI additionally runs full tests, reference-result verification, and public-repository integrity checks on Python 3.10/3.11/3.12. Heavy AMLGym and other external-data studies remain separately orchestrated and provenance-pinned.

The two manuscripts are maintained separately in `research/q1_2026/papers/`. Their joint story is architectural only: Paper A repairs learned authorization restrictions; Paper B decides what evidence to acquire when residual uncertainty remains.
