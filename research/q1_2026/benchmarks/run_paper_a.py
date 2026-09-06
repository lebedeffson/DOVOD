from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from random import Random

from paper_a.contextual_repair import (
    RepairEdit,
    RepairSample,
    build_frozen_vocabulary,
    repaired_allows,
    solve_contextual_repair_milp,
)
from paper_a.pac_bayes import sparse_repair_risk_upper

OUT = ROOT / "results" / "paper_a_contextual_benchmark.json"


def best_global_subset(base: tuple[int, ...], samples: tuple[RepairSample, ...]) -> tuple[tuple[int, ...], float]:
    best = None
    for mask in product((0, 1), repeat=len(base)):
        keep = tuple(p for p, bit in zip(base, mask) if bit)
        err = sum(int(repaired_allows(s.state, keep, tuple())) != s.allow for s in samples) / len(samples)
        candidate = (err, len(keep), keep)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best[2], float(best[0])


def main() -> None:
    rng = Random(20260905)
    n_bits = 8
    base = (0, 1, 2)
    context_indices = (4, 5, 6, 7)
    true_edits = (
        RepairEdit("exception", ((4, 1), (5, 0)), prerequisite=0),
        RepairEdit("exception", ((6, 1),), prerequisite=1),
        RepairEdit("guard", ((4, 0), (7, 1))),
    )
    support = tuple(product((0, 1), repeat=n_bits))
    vocabulary = build_frozen_vocabulary(
        support,
        base,
        context_indices=context_indices,
        max_context_width=2,
    )

    def labelled_sample() -> RepairSample:
        state = tuple(rng.choice((0, 1)) for _ in range(n_bits))
        return RepairSample(state, int(repaired_allows(state, base, true_edits)))

    train = tuple(labelled_sample() for _ in range(640))
    certification = tuple(labelled_sample() for _ in range(4096))
    learned = solve_contextual_repair_milp(base, vocabulary, train)
    cert_errors = sum(
        int(repaired_allows(s.state, base, learned.selected_edits)) != s.allow
        for s in certification
    )
    cert_risk = cert_errors / len(certification)
    global_model, train_global_risk = best_global_subset(base, train)
    global_cert_risk = sum(
        int(repaired_allows(s.state, global_model, tuple())) != s.allow
        for s in certification
    ) / len(certification)

    delta = 0.05
    rho = 0.02
    upper = sparse_repair_risk_upper(
        cert_risk,
        len(certification),
        delta,
        vocabulary_size=len(vocabulary),
        selected=len(learned.selected_edits),
        rho=rho,
    )
    support_disagreement = sum(
        repaired_allows(s, base, learned.selected_edits)
        != repaired_allows(s, base, true_edits)
        for s in support
    ) / len(support)

    report = {
        "schema": "dovod-paper-a-contextual-repair-v2",
        "seed": 20260905,
        "n_bits": n_bits,
        "support_states": len(support),
        "base_prerequisites": list(base),
        "context_indices": list(context_indices),
        "max_context_width": 2,
        "frozen_vocabulary_size": len(vocabulary),
        "train_samples": len(train),
        "certification_samples": len(certification),
        "selected_edits": [
            {
                "kind": e.kind,
                "context": [list(x) for x in e.context],
                "prerequisite": e.prerequisite,
                "weight": e.weight,
            }
            for e in learned.selected_edits
        ],
        "selected_edit_count": len(learned.selected_edits),
        "train_fit_exact": all(p == s.allow for p, s in zip(learned.predictions, train)),
        "certification_errors": cert_errors,
        "certification_risk": cert_risk,
        "pac_bayes": {
            "delta": delta,
            "rho": rho,
            "risk_upper": upper,
            "sampling_statement": "Certification states are IID draws from the prespecified uniform distribution on {0,1}^8 and are independent of repair fitting. The edit vocabulary is frozen from the unlabeled finite support before fitting labels.",
        },
        "support_disagreement_vs_true_contextual_model": support_disagreement,
        "global_deletion_baseline": {
            "selected_base_prerequisites": list(global_model),
            "train_risk": train_global_risk,
            "certification_risk": global_cert_risk,
        },
        "claim_boundary": "Controlled finite contextual benchmark validating exact bidirectional repair and its certificate mechanics. It is not AMLGym/IPC or real-procedure external evidence.",
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
