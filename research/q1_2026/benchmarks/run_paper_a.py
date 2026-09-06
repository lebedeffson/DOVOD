from __future__ import annotations

import json
import sys
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
    solve_contextual_repair_milp,
)
from paper_a.pac_bayes import sparse_repair_risk_upper
from paper_a.statistics import (
    certify_binary_errors,
    counterexample_samples_for_detection,
    finite_mask_count,
    paired_error_comparison,
    zero_error_uniform_upper,
)

OUT = ROOT / "results" / "paper_a_contextual_benchmark.json"
SEED = 20260905


def true_edits() -> tuple[RepairEdit, ...]:
    return (
        RepairEdit("exception", ((4, 1), (5, 0)), prerequisite=0),
        RepairEdit("exception", ((6, 1),), prerequisite=1),
        RepairEdit("guard", ((4, 0), (7, 1))),
    )


def labels(states, base, edits):
    return tuple(int(repaired_allows(s, base, edits)) for s in states)


def predictions(states, base, edits=()):
    return tuple(int(repaired_allows(s, base, edits)) for s in states)


def best_global_subset(base, samples):
    best = None
    for mask in product((0, 1), repeat=len(base)):
        keep = tuple(p for p, bit in zip(base, mask) if bit)
        pred = predictions((s.state for s in samples), keep)
        errors = sum(p != s.allow for p, s in zip(pred, samples))
        candidate = (errors, len(keep), keep)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best[2], best[0] / len(samples)


def serial_edit(edit: RepairEdit) -> dict:
    return {
        "kind": edit.kind,
        "context": [list(x) for x in edit.context],
        "prerequisite": edit.prerequisite,
        "weight": edit.weight,
    }


def main() -> None:
    rng = Random(SEED)
    n_bits = 8
    base = (0, 1, 2)
    planted = true_edits()
    support = tuple(product((0, 1), repeat=n_bits))
    support_labels = labels(support, base, planted)

    vocabulary = build_frozen_vocabulary(
        support,
        base,
        context_indices=(4, 5, 6, 7),
        max_context_width=2,
    )
    if len(vocabulary) != 128:
        raise AssertionError(f"unexpected vocabulary size: {len(vocabulary)}")

    train_states = tuple(rng.choice(support) for _ in range(640))
    train_labels = labels(train_states, base, planted)
    train = tuple(RepairSample(s, y) for s, y in zip(train_states, train_labels))
    fit = solve_contextual_repair_milp(base, vocabulary, train)

    cert_states = tuple(rng.choice(support) for _ in range(4096))
    cert_labels = labels(cert_states, base, planted)
    contextual_pred = predictions(cert_states, base, fit.selected_edits)
    base_pred = predictions(cert_states, base)

    global_subset, train_global_risk = best_global_subset(base, train)
    global_pred = predictions(cert_states, global_subset)

    contextual_cert = certify_binary_errors(cert_labels, contextual_pred)
    base_cert = certify_binary_errors(cert_labels, base_pred)
    global_cert = certify_binary_errors(cert_labels, global_pred)

    negative_idx = [i for i, y in enumerate(cert_labels) if y == 0]
    positive_idx = [i for i, y in enumerate(cert_labels) if y == 1]
    false_allow_cert = certify_binary_errors(
        [0] * len(negative_idx), [contextual_pred[i] for i in negative_idx]
    )
    false_block_cert = certify_binary_errors(
        [1] * len(positive_idx), [contextual_pred[i] for i in positive_idx]
    )

    support_contextual = predictions(support, base, fit.selected_edits)
    support_base = predictions(support, base)
    support_global = predictions(support, global_subset)

    h = finite_mask_count(len(vocabulary), len(fit.selected_edits))
    uniform_upper = zero_error_uniform_upper(
        len(train), len(vocabulary), len(fit.selected_edits), delta=0.05
    )
    pac_upper = sparse_repair_risk_upper(
        contextual_cert.empirical_risk,
        contextual_cert.n,
        0.05,
        vocabulary_size=len(vocabulary),
        selected=len(fit.selected_edits),
        rho=0.02,
    )

    report = {
        "schema": "dovod-paper-a-contextual-repair-v4",
        "seed": SEED,
        "problem": {
            "bits": n_bits,
            "base_prerequisites": list(base),
            "support_states": len(support),
            "frozen_vocabulary_size": len(vocabulary),
            "train_samples": len(train),
            "independent_certification_samples": len(cert_states),
            "planted_edits": [serial_edit(e) for e in planted],
        },
        "fit": {
            "selected_edits": [serial_edit(e) for e in fit.selected_edits],
            "selected_edit_count": len(fit.selected_edits),
            "objective": fit.objective,
            "train_fit_exact": fit.predictions == train_labels,
            "exact_planted_edit_recovery": set(fit.selected_edits) == set(planted),
        },
        "complete_support": {
            "contextual_risk": sum(p != y for p, y in zip(support_contextual, support_labels)) / len(support),
            "base_risk": sum(p != y for p, y in zip(support_base, support_labels)) / len(support),
            "global_deletion_risk": sum(p != y for p, y in zip(support_global, support_labels)) / len(support),
            "global_subset": list(global_subset),
            "global_subset_train_risk": train_global_risk,
        },
        "independent_holdout": {
            "contextual": contextual_cert.__dict__,
            "base": base_cert.__dict__,
            "global_deletion": global_cert.__dict__,
            "contextual_false_allow": false_allow_cert.__dict__,
            "contextual_false_block": false_block_cert.__dict__,
            "base_vs_contextual": paired_error_comparison(cert_labels, base_pred, contextual_pred).__dict__,
            "global_vs_contextual": paired_error_comparison(cert_labels, global_pred, contextual_pred).__dict__,
        },
        "finite_class_training_certificate": {
            "delta": 0.05,
            "hypothesis_count_size_at_most_selected": h,
            "zero_training_error_uniform_risk_upper": uniform_upper,
            "assumption": "The edit vocabulary is fixed before training labels are inspected.",
        },
        "counterexample_discovery": {
            "counterexample_mass_lower": 0.05,
            "delta": 0.05,
            "minimum_positive_observations": counterexample_samples_for_detection(0.05, delta=0.05),
        },
        "pac_bayes_secondary": {
            "risk_upper": pac_upper,
            "rho": 0.02,
            "delta": 0.05,
            "note": "Secondary synthetic fixed-vocabulary certificate; independent exact holdout is primary.",
        },
        "claim_boundary": (
            "Controlled finite benchmark validating optimization, identifiability-side certificates, "
            "and baseline behavior. It is not AMLGym/IPC or real-procedure external evidence."
        ),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
