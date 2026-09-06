from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from itertools import product
from pathlib import Path
from random import Random

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_a.contextual_repair import (
    RepairEdit,
    RepairSample,
    build_frozen_vocabulary,
    repaired_allows,
    solve_contextual_repair_soft_milp,
)

OUT = ROOT / "results" / "paper_a_stress.json"


def planted_edits():
    return (
        RepairEdit("exception", ((4, 1), (5, 0)), prerequisite=0),
        RepairEdit("exception", ((6, 1),), prerequisite=1),
        RepairEdit("guard", ((4, 0), (7, 1))),
    )


def pred(states, base, edits=()):
    return tuple(int(repaired_allows(s, base, edits)) for s in states)


def weighted_vocab(vocab, weight=0.25):
    return tuple(RepairEdit(e.kind, e.context, e.prerequisite, weight) for e in vocab)


def best_global_subset(base, samples):
    best = None
    for mask in product((0, 1), repeat=len(base)):
        keep = tuple(p for p, bit in zip(base, mask) if bit)
        guesses = pred((s.state for s in samples), keep)
        errors = sum(g != s.allow for g, s in zip(guesses, samples))
        candidate = (errors, len(keep), keep)
        if best is None or candidate < best:
            best = candidate
    return best[2]


def mean(xs):
    return sum(xs) / len(xs)


def main() -> None:
    start = time.perf_counter()
    base = (0, 1, 2)
    planted = planted_edits()
    support = tuple(product((0, 1), repeat=8))
    truth = pred(support, base, planted)
    vocab = weighted_vocab(
        build_frozen_vocabulary(
            support, base, context_indices=(4, 5, 6, 7), max_context_width=2
        )
    )

    rows = []
    for n in (64, 128, 256, 640):
        for noise in (0.0, 0.05):
            for seed in range(5):
                rng = Random(20260905 + 10000 * n + 100 * int(noise * 100) + seed)
                states = tuple(rng.choice(support) for _ in range(n))
                clean = list(pred(states, base, planted))
                noisy = tuple((1 - y) if rng.random() < noise else y for y in clean)
                samples = tuple(RepairSample(s, y) for s, y in zip(states, noisy))
                fit = solve_contextual_repair_soft_milp(
                    base,
                    vocab,
                    samples,
                    false_allow_weight=1.0,
                    false_block_weight=1.0,
                )
                contextual = pred(support, base, fit.selected_edits)
                global_subset = best_global_subset(base, samples)
                global_pred = pred(support, global_subset)
                rows.append(
                    {
                        "n": n,
                        "label_noise": noise,
                        "seed": seed,
                        "selected_edit_count": len(fit.selected_edits),
                        "exact_recovery": set(fit.selected_edits) == set(weighted_vocab(planted)),
                        "contextual_risk": sum(a != b for a, b in zip(contextual, truth)) / len(support),
                        "global_risk": sum(a != b for a, b in zip(global_pred, truth)) / len(support),
                        "base_risk": sum(a != b for a, b in zip(pred(support, base), truth)) / len(support),
                        "training_errors": len(fit.error_indices),
                    }
                )

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["n"], row["label_noise"])].append(row)
    summary = []
    for (n, noise), group in sorted(grouped.items()):
        summary.append(
            {
                "n": n,
                "label_noise": noise,
                "seeds": len(group),
                "exact_recovery_rate": mean([float(x["exact_recovery"]) for x in group]),
                "contextual_risk_mean": mean([x["contextual_risk"] for x in group]),
                "contextual_risk_max": max(x["contextual_risk"] for x in group),
                "global_risk_mean": mean([x["global_risk"] for x in group]),
                "base_risk_mean": mean([x["base_risk"] for x in group]),
                "training_errors_mean": mean([x["training_errors"] for x in group]),
            }
        )

    report = {
        "schema": "dovod-paper-a-stress-v2",
        "rows": rows,
        "summary": summary,
        "elapsed_seconds": time.perf_counter() - start,
        "claim_boundary": (
            "Synthetic robustness diagnostic over sample size and 5% label corruption. "
            "It is not external benchmark evidence."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
