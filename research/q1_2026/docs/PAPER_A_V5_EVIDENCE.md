# Paper A V5 evidence amendment

This amendment adds two reviewer-facing evidence blocks **after** the frozen V4 confirmatory analysis. Neither block changes the frozen AMLGym primary result (`7/68/0` usable-cell win/tie/loss versus upstream; domain `4/6/0`, exact two-sided sign-test `p=0.125`).

## 1. Full-matrix simple baselines (post-confirmatory)

Workflow run: `34137791383`, head `10acbd00e52d8c1d2052d3a3966d43a70e3bd54d`.

The matrix covers all 160 AMLGym cells structurally and retains the same five upstream failure/timeout cells and 80 empty held-out cells, leaving 75 usable cells. The baselines are intentionally weak, at-most-one-edit heuristics using the same repair/calibration/test split and conservative calibration gate as DOVOD.

- DOVOD vs upstream on this seeded post-freeze panel: `6 wins / 69 ties / 0 losses`, mean risk reduction `0.00718254`.
- Random one-edit gated baseline vs upstream: `2 / 73 / 0`, mean risk reduction `0.00398148`.
- Frequency one-edit gated baseline vs upstream: `2 / 73 / 0`, mean risk reduction `0.00203704`.
- Direct DOVOD vs random: `6 wins / 69 ties / 0 losses`; mean random-minus-DOVOD risk `0.00320106`.
- Direct DOVOD vs frequency: `6 / 69 / 0`; mean frequency-minus-DOVOD risk `0.00514550`.
- Domain means: DOVOD has `3 wins / 7 ties / 0 losses` against each simple baseline; exact two-sided sign-test `p=0.25`.

Interpretation: these post-confirmatory baselines answer only whether the observed local-repair gains can be reproduced by a trivial one-edit choice. They are descriptive and cannot be used to retrofit significance into the frozen primary study.

Tracked receipt: `results/paper_a_amlgym_simple_baselines_v5.json`.

## 2. Assembly101 held-out ordering audit

Dataset: `assembly-101/assembly101-mistake-detection`, pinned commit `6f3a953267ffb86cdeabf6751af05a75a011a4f8`, CC BY-NC 4.0. The executable audit sees the 328 CSV annotation files reported by the dataset README and 3,964 annotated events (`2,927 correct`, `330 correction`, `707 mistake`). Workflow evidence head: `dcff9be0e0bea1aa328a046fc6c50716642c0485`.

Primary split (sequence-level): `257` train sequences / `71` held-out sequences.

At predecessor-frequency threshold `0.90`:

- 18 candidate relations are learned from training sequences;
- 5/18 are directly refuted by held-out labeled-correct behavior;
- 8/37 covered held-out correct events violate at least one relation;
- 2/5 covered explicit ordering mistakes violate a relation;
- violation precision when contrasting only correct vs explicit-ordering-mistake events is `0.20`.

Toy-id robustness split: `87` train toy groups / `14` held-out toy groups (`277/51` sequences). At threshold `0.90`, 18 relations are again learned, 5/18 are refuted by held-out correct behavior, and violations occur in `11/31` covered correct events and `3/5` covered ordering mistakes.

Threshold sensitivity preserves the qualitative conclusion. Sequence split correct-behavior refutations are `24/59` at `0.80`, `5/18` at `0.90`, and `1/8` at `0.95`; toy-group split refutations are `14/52`, `5/18`, and `5/14` respectively.

Interpretation: high-frequency temporal order is not mechanical prerequisite evidence, and relation violation alone is not a reliable mistake detector. This is a falsification/claim-boundary audit, not external validation of the full DOVOD repair policy.

Tracked receipt: `results/paper_a_assembly101_ordering_audit_v5.json`.
