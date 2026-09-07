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

OUT = ROOT / "results" / "paper_a_multiaction16.json"
STATE_BITS = 16
EDIT_WEIGHT = 0.25


def _edit(kind, context, prerequisite=-1):
    return RepairEdit(kind, tuple(sorted(context)), prerequisite=prerequisite, weight=EDIT_WEIGHT)


def action_specs() -> dict[str, dict]:
    """Two action-local repair problems embedded in one 16-bit state space.

    The actions share prerequisites 2/3 and context variables 10/11, but have
    distinct additional prerequisites and distinct planted contextual edits.
    This makes the benchmark strictly larger than the original 8-bit single-
    action diagnostic without changing the repair semantics.
    """
    return {
        "action_A": {
            "base": (0, 1, 2, 3),
            "context_indices": (8, 9, 10, 11),
            "planted": (
                _edit("exception", ((8, 1), (9, 0)), 0),
                _edit("exception", ((10, 1),), 1),
                _edit("exception", ((11, 0),), 2),
                _edit("guard", ((8, 0), (10, 1))),
            ),
        },
        "action_B": {
            "base": (2, 3, 4, 5),
            "context_indices": (10, 11, 12, 13),
            "planted": (
                _edit("exception", ((10, 0), (12, 1)), 2),
                _edit("exception", ((11, 1),), 3),
                _edit("exception", ((12, 0), (13, 1)), 4),
                _edit("guard", ((10, 1), (13, 0))),
            ),
        },
    }


def _context_support(indices: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    rows = []
    for values in product((0, 1), repeat=len(indices)):
        state = [0] * STATE_BITS
        for idx, value in zip(indices, values):
            state[idx] = value
        rows.append(tuple(state))
    return tuple(rows)


def _weighted_vocab(states, base, context_indices):
    raw = build_frozen_vocabulary(
        states,
        base,
        context_indices=context_indices,
        max_context_width=2,
    )
    return tuple(RepairEdit(e.kind, e.context, e.prerequisite, EDIT_WEIGHT) for e in raw)


def _truth(state, spec) -> int:
    return int(repaired_allows(state, spec["base"], spec["planted"]))


def _sample_state(rng: Random) -> tuple[int, ...]:
    bits = rng.getrandbits(STATE_BITS)
    return tuple((bits >> i) & 1 for i in range(STATE_BITS))


def _risk_over_full_support(base, edits, spec) -> float:
    errors = 0
    total = 1 << STATE_BITS
    for bits in range(total):
        state = tuple((bits >> i) & 1 for i in range(STATE_BITS))
        pred = int(repaired_allows(state, base, edits))
        errors += pred != _truth(state, spec)
    return errors / total


def _best_global_subset(base, samples):
    best = None
    for mask in product((0, 1), repeat=len(base)):
        keep = tuple(p for p, bit in zip(base, mask) if bit)
        errors = sum(int(repaired_allows(s.state, keep, ())) != s.allow for s in samples)
        candidate = (errors, len(base) - len(keep), keep)
        if best is None or candidate < best:
            best = candidate
    return best[2]


def _mean(values):
    return sum(values) / len(values) if values else None


def main() -> None:
    started = time.perf_counter()
    rows = []
    for action_name, spec in action_specs().items():
        vocab = _weighted_vocab(
            _context_support(spec["context_indices"]),
            spec["base"],
            spec["context_indices"],
        )
        planted_set = set(spec["planted"])
        missing = planted_set.difference(vocab)
        if missing:
            raise RuntimeError(f"planted edits missing from vocabulary for {action_name}: {sorted(missing)}")

        base_risk = _risk_over_full_support(spec["base"], (), spec)
        for n in (256, 512, 1024):
            for seed in range(3):
                rng = Random(20260907 + 100_000 * (action_name == "action_B") + 100 * n + seed)
                states = tuple(_sample_state(rng) for _ in range(n))
                samples = tuple(RepairSample(s, _truth(s, spec)) for s in states)
                t0 = time.perf_counter()
                fit = solve_contextual_repair_soft_milp(
                    spec["base"],
                    vocab,
                    samples,
                    false_allow_weight=1.0,
                    false_block_weight=1.0,
                )
                fit_seconds = time.perf_counter() - t0
                global_subset = _best_global_subset(spec["base"], samples)
                contextual_risk = _risk_over_full_support(spec["base"], fit.selected_edits, spec)
                global_risk = _risk_over_full_support(global_subset, (), spec)
                rows.append(
                    {
                        "action": action_name,
                        "state_bits": STATE_BITS,
                        "n": n,
                        "seed": seed,
                        "vocabulary_size": len(vocab),
                        "planted_edit_count": len(spec["planted"]),
                        "selected_edit_count": len(fit.selected_edits),
                        "exact_structural_recovery": set(fit.selected_edits) == planted_set,
                        "training_errors": len(fit.error_indices),
                        "contextual_full_support_risk": contextual_risk,
                        "global_deletion_full_support_risk": global_risk,
                        "base_full_support_risk": base_risk,
                        "fit_seconds": fit_seconds,
                    }
                )

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["action"], row["n"])].append(row)
    summary = []
    for (action, n), group in sorted(grouped.items()):
        summary.append(
            {
                "action": action,
                "n": n,
                "seeds": len(group),
                "exact_structural_recovery_rate": _mean([float(r["exact_structural_recovery"]) for r in group]),
                "contextual_full_support_risk_mean": _mean([r["contextual_full_support_risk"] for r in group]),
                "contextual_full_support_risk_max": max(r["contextual_full_support_risk"] for r in group),
                "global_deletion_full_support_risk_mean": _mean([r["global_deletion_full_support_risk"] for r in group]),
                "base_full_support_risk_mean": _mean([r["base_full_support_risk"] for r in group]),
                "fit_seconds_mean": _mean([r["fit_seconds"] for r in group]),
            }
        )

    report = {
        "schema": "dovod-paper-a-multiaction16-v1",
        "state_bits": STATE_BITS,
        "actions": {
            name: {
                "base_prerequisites": list(spec["base"]),
                "context_indices": list(spec["context_indices"]),
                "planted_edit_count": len(spec["planted"]),
            }
            for name, spec in action_specs().items()
        },
        "shared_structure": {
            "shared_base_prerequisites": [2, 3],
            "shared_context_variables": [10, 11],
        },
        "rows": rows,
        "summary": summary,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Controlled 16-bit two-action scalability/recovery benchmark with planted contextual edits. "
            "It tests whether the existing MILP repair mechanism remains effective when the state dimension doubles and action structures overlap. "
            "It is synthetic controlled evidence, not an external-dataset result."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
