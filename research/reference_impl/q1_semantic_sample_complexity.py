from __future__ import annotations

import csv
import itertools
import json
import statistics
from collections import defaultdict
from pathlib import Path

from hardening_loro_graph import action_set, edges, evaluate, next_events, read_records
from q1_decision_equivalent_graph import N, load_full_dependencies, split_states

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = RESULTS / "q1_extension"
OUT.mkdir(parents=True, exist_ok=True)


def infer_from_selected_train_recordings(recs, selected: tuple[str, ...]) -> dict[int, list[int]]:
    selected_set = set(selected)
    pre: dict[int, list[list[int]]] = defaultdict(list)
    for split, rec, rows in recs:
        if split != "train" or rec not in selected_set:
            continue
        for k in range(len(rows) - 1):
            state, nxt = rows[k][1], rows[k + 1][1]
            for action, (before, after) in enumerate(zip(state, nxt)):
                if before != 1 and after == 1:
                    pre[action].append(state)
    deps = {j: [] for j in range(N)}
    for action in range(N):
        states = pre[action]
        if not states:
            continue
        for pred in range(N):
            if pred == action:
                continue
            if all(int(state[pred]) == 1 for state in states):
                deps[action].append(pred)
    return deps


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    recs = read_records()
    train_names = sorted(rec for sp, rec, _ in recs if sp == "train")
    full = load_full_dependencies()
    val_states = split_states(recs, "val")
    test_states = split_states(recs, "test")
    val_events = next_events(recs, split="val")
    test_events = next_events(recs, split="test")

    rows: list[dict] = []
    for k in range(1, len(train_names) + 1):
        for subset in itertools.combinations(train_names, k):
            graph = infer_from_selected_train_recordings(recs, subset)
            val_eval = evaluate(val_events, graph)
            test_eval = evaluate(test_events, graph)
            val_diff = sum(action_set(s, graph) != action_set(s, full) for s in val_states)
            test_diff = sum(action_set(s, graph) != action_set(s, full) for s in test_states)

            held_events = [
                event
                for rec in train_names
                if rec not in subset
                for event in next_events(recs, split="train", recording=rec)
            ]
            held_eval = evaluate(held_events, graph) if held_events else {
                "n": 0,
                "recall": None,
                "mean_candidates": None,
            }
            rows.append({
                "train_recordings": k,
                "subset": ";".join(subset),
                "edges": len(edges(graph)),
                "heldout_train_events": held_eval["n"],
                "heldout_train_next_recall": held_eval["recall"],
                "heldout_train_mean_candidates": held_eval["mean_candidates"],
                "validation_next_recall": val_eval["recall"],
                "validation_mean_candidates": val_eval["mean_candidates"],
                "validation_states_different_from_full": val_diff,
                "test_next_recall": test_eval["recall"],
                "test_mean_candidates": test_eval["mean_candidates"],
                "test_states_different_from_full": test_diff,
            })

    aggregated: list[dict] = []
    for k in range(1, len(train_names) + 1):
        group = [r for r in rows if r["train_recordings"] == k]
        held = [r["heldout_train_next_recall"] for r in group if r["heldout_train_next_recall"] is not None]
        aggregated.append({
            "train_recordings": k,
            "subsets": len(group),
            "edges_mean": statistics.mean(r["edges"] for r in group),
            "edges_median": statistics.median(r["edges"] for r in group),
            "edges_min": min(r["edges"] for r in group),
            "edges_max": max(r["edges"] for r in group),
            "heldout_train_recall_mean": statistics.mean(held) if held else None,
            "test_recall_mean": statistics.mean(r["test_next_recall"] for r in group),
            "test_recall_median": statistics.median(r["test_next_recall"] for r in group),
            "test_recall_min": min(r["test_next_recall"] for r in group),
            "test_recall_max": max(r["test_next_recall"] for r in group),
            "test_exact_policy_fraction": statistics.mean(r["test_states_different_from_full"] == 0 for r in group),
            "validation_exact_policy_fraction": statistics.mean(r["validation_states_different_from_full"] == 0 for r in group),
            "test_diff_states_median": statistics.median(r["test_states_different_from_full"] for r in group),
        })

    summary = {
        "schema": "tinyapv-semantic-sample-complexity-v1",
        "protocol": (
            "Exact enumeration of all non-empty subsets of the 11 MECCANO train recordings. "
            "For each subset, prerequisites are inferred with the frozen support=1.0 rule; no "
            "validation/test labels are used for graph construction. Remaining train recordings, "
            "validation and test are evaluation only."
        ),
        "subsets_total": len(rows),
        "full_train_recordings": len(train_names),
        "full_graph_edges": len(edges(full)),
        "aggregated": aggregated,
        "primary_supported_observation": (
            "With a strict support=1 prerequisite rule, small training subsets contain many "
            "accidental prerequisites and overconstrain the action set. As independent train "
            "recordings accumulate, inferred graphs become smaller and held-out/test next-action "
            "recall rises. Exact held-out action-policy recovery is substantially more data-hungry "
            "than obtaining high next-action recall."
        ),
        "claim_boundary": (
            "This is an exhaustive subsampling analysis of one dataset, not a universal sample-"
            "complexity theorem. Fixed validation/test reuse is descriptive robustness analysis, "
            "not hyperparameter selection."
        ),
    }
    (OUT / "semantic_sample_complexity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(OUT / "semantic_sample_complexity_subsets.csv", rows)
    write_csv(OUT / "semantic_sample_complexity_aggregated.csv", aggregated)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
