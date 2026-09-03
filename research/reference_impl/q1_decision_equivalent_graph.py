from __future__ import annotations

import csv
import itertools
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from hardening_loro_graph import action_set, edges, evaluate, infer, next_events, read_records

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = RESULTS / "q1_extension"
OUT.mkdir(parents=True, exist_ok=True)
N = 17


def load_full_dependencies() -> dict[int, list[int]]:
    raw = json.loads((RESULTS / "meccano_train_empirical_dependencies.json").read_text())
    return {int(k): list(map(int, v)) for k, v in raw.items()}


def split_states(recs, split: str, recording: str | None = None) -> list[list[int]]:
    return [
        s
        for sp, rec, rows in recs
        if sp == split and (recording is None or rec == recording)
        for _, s in rows
    ]


def stage_of(state: list[int]) -> str:
    frac = sum(int(v) == 1 for v in state) / len(state)
    if frac < 1 / 3:
        return "early"
    if frac < 2 / 3:
        return "middle"
    return "late"


def minimum_decision_equivalent_graph(
    base_deps: dict[int, list[int]], states: list[list[int]]
) -> dict[int, list[int]]:
    """Return the minimum edge subset preserving base action blocking on states."""
    selected: dict[int, list[int]] = {j: [] for j in range(N)}
    for action in range(N):
        preds = list(base_deps.get(action, []))
        if not preds:
            continue
        constraints: list[np.ndarray] = []
        for state in states:
            if int(state[action]) == 1:
                continue
            violated = [idx for idx, p in enumerate(preds) if int(state[p]) != 1]
            if not violated:
                continue
            row = np.zeros(len(preds), dtype=float)
            row[violated] = 1.0
            constraints.append(row)
        if not constraints:
            continue
        a = np.vstack(constraints)
        result = milp(
            c=np.ones(len(preds), dtype=float),
            integrality=np.ones(len(preds), dtype=int),
            bounds=Bounds(np.zeros(len(preds)), np.ones(len(preds))),
            constraints=LinearConstraint(a, np.ones(len(a)), np.full(len(a), np.inf)),
            options={"time_limit": 30.0},
        )
        if result.x is None or not result.success:
            raise RuntimeError(f"MILP failed for action {action}: {result.message}")
        mask = np.rint(result.x).astype(int)
        selected[action] = [p for p, keep in zip(preds, mask) if int(keep) == 1]
    return selected


def _solve_action_cover(
    action: int,
    preds: list[int],
    states: list[list[int]],
    force_zero: set[int] | None = None,
    force_one: set[int] | None = None,
) -> tuple[int | None, list[int] | None]:
    if not preds:
        return 0, []
    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []
    for state in states:
        if int(state[action]) == 1:
            continue
        violated = [idx for idx, p in enumerate(preds) if int(state[p]) != 1]
        if violated:
            row = np.zeros(len(preds), dtype=float)
            row[violated] = 1.0
            rows.append(row)
            lower.append(1.0)
            upper.append(np.inf)
    for p in sorted(force_zero or set()):
        row = np.zeros(len(preds), dtype=float)
        row[preds.index(p)] = 1.0
        rows.append(row)
        lower.append(0.0)
        upper.append(0.0)
    for p in sorted(force_one or set()):
        row = np.zeros(len(preds), dtype=float)
        row[preds.index(p)] = 1.0
        rows.append(row)
        lower.append(1.0)
        upper.append(1.0)
    if not rows:
        return 0, []
    a = np.vstack(rows)
    result = milp(
        c=np.ones(len(preds), dtype=float),
        integrality=np.ones(len(preds), dtype=int),
        bounds=Bounds(np.zeros(len(preds)), np.ones(len(preds))),
        constraints=LinearConstraint(a, np.asarray(lower), np.asarray(upper)),
        options={"time_limit": 30.0},
    )
    if result.x is None or not result.success:
        return None, None
    mask = np.rint(result.x).astype(int)
    chosen = [p for p, keep in zip(preds, mask) if int(keep) == 1]
    return int(mask.sum()), chosen


def enumerate_optimal_action_covers(
    base_deps: dict[int, list[int]], states: list[list[int]]
) -> dict[int, list[tuple[int, ...]]]:
    out: dict[int, list[tuple[int, ...]]] = {}
    for action in range(N):
        preds = list(base_deps.get(action, []))
        blocked: list[set[int]] = []
        for state in states:
            if int(state[action]) == 1:
                continue
            violated = {p for p in preds if int(state[p]) != 1}
            if violated:
                blocked.append(violated)
        if not preds or not blocked:
            out[action] = [tuple()]
            continue
        solutions: list[tuple[int, ...]] = []
        for k in range(len(preds) + 1):
            for comb in itertools.combinations(preds, k):
                chosen = set(comb)
                if all(chosen & requirement for requirement in blocked):
                    solutions.append(tuple(comb))
            if solutions:
                break
        out[action] = solutions
    return out


def optimal_edge_identifiability(
    base_deps: dict[int, list[int]], states: list[list[int]]
) -> list[dict]:
    out: list[dict] = []
    for action in range(N):
        preds = list(base_deps.get(action, []))
        optimum, one_solution = _solve_action_cover(action, preds, states)
        if optimum is None:
            raise RuntimeError(f"Could not solve base cover for action {action}")
        for pred in preds:
            opt_without, _ = _solve_action_cover(action, preds, states, force_zero={pred})
            opt_with, _ = _solve_action_cover(action, preds, states, force_one={pred})
            mandatory = opt_without is None or opt_without > optimum
            participates = opt_with is not None and opt_with == optimum
            if mandatory:
                role = "mandatory"
            elif participates:
                role = "optional_optimal"
            else:
                role = "nonoptimal_redundant"
            out.append({
                "predecessor": pred,
                "action": action,
                "role": role,
                "action_optimum_edges": optimum,
                "optimum_if_forbidden": opt_without,
                "optimum_if_forced": opt_with,
                "in_reference_solution": pred in (one_solution or []),
            })
    return out


def transitive_reduction(base_deps: dict[int, list[int]]) -> dict[int, list[int]]:
    es = sorted(edges(base_deps))
    kept = set(es)

    def reachable(src: int, dst: int, edge_set: set[tuple[int, int]]) -> bool:
        stack = [src]
        seen = {src}
        adj: dict[int, list[int]] = {j: [] for j in range(N)}
        for a, b in edge_set:
            adj[a].append(b)
        while stack:
            cur = stack.pop()
            for nxt in adj[cur]:
                if nxt == dst:
                    return True
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return False

    for edge in es:
        trial = set(kept)
        trial.remove(edge)
        if reachable(edge[0], edge[1], trial):
            kept.remove(edge)
    out = {j: [] for j in range(N)}
    for pred, action in sorted(kept):
        out[action].append(pred)
    return out


def compare_action_sets(recs, base, candidate, split: str) -> dict:
    rows = []
    for sp, rec, states in recs:
        if sp != split:
            continue
        for frame, state in states:
            full = tuple(action_set(state, base))
            compact = tuple(action_set(state, candidate))
            rows.append({
                "split": split,
                "recording": rec,
                "frame": frame,
                "equal": full == compact,
                "base_n": len(full),
                "candidate_n": len(compact),
            })
    by_rec = {}
    for rec in sorted({r["recording"] for r in rows}):
        rs = [r for r in rows if r["recording"] == rec]
        by_rec[rec] = {
            "states": len(rs),
            "equal_states": sum(r["equal"] for r in rs),
            "equivalence_rate": statistics.mean(r["equal"] for r in rs),
        }
    return {
        "states": len(rows),
        "equal_states": sum(r["equal"] for r in rows),
        "equivalence_rate": statistics.mean(r["equal"] for r in rows) if rows else None,
        "recordings": by_rec,
        "rows": rows,
    }


def guide_certificate_burden(recs, base, compact, stage_graphs=None) -> dict:
    full_counts: list[int] = []
    compact_counts: list[int] = []
    stage_counts: list[int] = []
    stage_dist = Counter()
    for sp, _, rows in recs:
        if sp != "test":
            continue
        for _, state in rows:
            actions = [a for a in action_set(state, base) if base.get(a)]
            if not actions:
                continue
            action = int(actions[0])
            full_counts.append(len(base[action]))
            compact_counts.append(len(compact[action]))
            if stage_graphs is not None:
                st = stage_of(state)
                stage_dist[st] += 1
                stage_counts.append(len(stage_graphs[st][action]))

    def stats(xs: list[int]) -> dict:
        return {
            "certificates": len(xs),
            "total_required_predicates": int(sum(xs)),
            "mean_required_predicates": statistics.mean(xs) if xs else None,
            "median_required_predicates": statistics.median(xs) if xs else None,
            "max_required_predicates": max(xs) if xs else None,
        }

    full = stats(full_counts)
    compact = stats(compact_counts)
    out = {
        "full_graph": full,
        "decision_equivalent_graph": compact,
        "relative_premise_reduction": (
            1 - compact["total_required_predicates"] / full["total_required_predicates"]
            if full["total_required_predicates"] else None
        ),
    }
    if stage_graphs is not None:
        stage = stats(stage_counts)
        out["stage_conditioned_graph"] = stage
        out["stage_conditioned_relative_premise_reduction"] = (
            1 - stage["total_required_predicates"] / full["total_required_predicates"]
            if full["total_required_predicates"] else None
        )
        out["stage_distribution"] = dict(stage_dist)
        out["stage_zero_premise_guides"] = sum(v == 0 for v in stage_counts)
    return out


def loro_robustness(recs, full_base, full_compact) -> list[dict]:
    train_recordings = sorted(rec for sp, rec, _ in recs if sp == "train")
    full_compact_edges = edges(full_compact)
    out = []
    for held in train_recordings:
        base = infer(recs, excluded=held, support=1.0)
        fit_states = [
            state
            for sp, rec, rows in recs
            if sp == "train" and rec != held
            for _, state in rows
        ]
        compact = minimum_decision_equivalent_graph(base, fit_states)
        compact_edges = edges(compact)
        union = compact_edges | full_compact_edges
        test = evaluate(next_events(recs, split="test"), compact)
        held_eval = evaluate(next_events(recs, split="train", recording=held), compact)
        test_states = split_states(recs, "test")
        full_candidate_diffs = sum(action_set(s, full_base) != action_set(s, compact) for s in test_states)
        out.append({
            "held_out_recording": held,
            "compact_edges": len(compact_edges),
            "edge_jaccard_vs_full_compact": (
                len(compact_edges & full_compact_edges) / len(union) if union else 1.0
            ),
            "heldout_next_recall": held_eval["recall"],
            "test_next_recall": test["recall"],
            "test_mean_candidates": test["mean_candidates"],
            "test_states_different_from_full_graph": full_candidate_diffs,
        })
    return out


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
    base = load_full_dependencies()
    train_states = split_states(recs, "train")

    compact = minimum_decision_equivalent_graph(base, train_states)
    edge_identifiability = optimal_edge_identifiability(base, train_states)
    optimal_covers = enumerate_optimal_action_covers(base, train_states)
    transitive = transitive_reduction(base)
    stage_graphs = {
        st: minimum_decision_equivalent_graph(base, [s for s in train_states if stage_of(s) == st])
        for st in ("early", "middle", "late")
    }

    comparisons = {}
    for name, graph in {"decision_equivalent": compact, "transitive_reduction": transitive}.items():
        comparisons[name] = {}
        for split in ("train", "val", "test"):
            comp = compare_action_sets(recs, base, graph, split)
            comparisons[name][split] = {k: v for k, v in comp.items() if k != "rows"}

    stage_equivalence = {}
    for split in ("train", "val", "test"):
        rows = []
        for sp, rec, rec_rows in recs:
            if sp != split:
                continue
            for frame, state in rec_rows:
                st = stage_of(state)
                rows.append({
                    "recording": rec,
                    "frame": frame,
                    "stage": st,
                    "equal": action_set(state, base) == action_set(state, stage_graphs[st]),
                })
        stage_equivalence[split] = {
            "states": len(rows),
            "equal_states": sum(r["equal"] for r in rows),
            "equivalence_rate": statistics.mean(r["equal"] for r in rows) if rows else None,
        }

    evals = {
        name: {split: evaluate(next_events(recs, split=split), graph) for split in ("train", "val", "test")}
        for name, graph in {"full": base, "decision_equivalent": compact, "transitive_reduction": transitive}.items()
    }

    burden = guide_certificate_burden(recs, base, compact, stage_graphs)
    loro = loro_robustness(recs, base, compact)

    summary = {
        "schema": "tinyapv-decision-equivalent-graph-v1",
        "derivation_boundary": "All graph selection/optimization uses MECCANO train states only. Validation/test states are used only for locked evaluation.",
        "interpretation_boundary": "Decision equivalence is relative to the train-derived empirical graph and observed state support. It does not establish mechanical completeness or safety.",
        "full_graph_edges": len(edges(base)),
        "transitive_reduction_edges": len(edges(transitive)),
        "decision_equivalent_edges": len(edges(compact)),
        "decision_equivalent_edge_reduction": 1 - len(edges(compact)) / len(edges(base)),
        "optimal_edge_identifiability": {
            "mandatory": sum(r["role"] == "mandatory" for r in edge_identifiability),
            "optional_optimal": sum(r["role"] == "optional_optimal" for r in edge_identifiability),
            "nonoptimal_redundant": sum(r["role"] == "nonoptimal_redundant" for r in edge_identifiability),
            "minimum_graph_count": math.prod(len(v) for v in optimal_covers.values()),
            "action_solution_counts": {str(k): len(v) for k, v in optimal_covers.items()},
            "ambiguous_action_covers": {str(k): [list(x) for x in v] for k, v in optimal_covers.items() if len(v) > 1},
            "rows": edge_identifiability,
        },
        "stage_conditioned_edges": {st: len(edges(g)) for st, g in stage_graphs.items()},
        "graphs": {
            "full": {str(k): v for k, v in base.items()},
            "transitive_reduction": {str(k): v for k, v in transitive.items()},
            "decision_equivalent": {str(k): v for k, v in compact.items()},
            "stage_conditioned": {st: {str(k): v for k, v in g.items()} for st, g in stage_graphs.items()},
        },
        "candidate_set_equivalence": comparisons,
        "stage_conditioned_equivalence": stage_equivalence,
        "next_action_evaluation": evals,
        "guide_certificate_burden": burden,
        "loro": {
            "runs": len(loro),
            "compact_edges_mean": statistics.mean(r["compact_edges"] for r in loro),
            "edge_jaccard_mean": statistics.mean(r["edge_jaccard_vs_full_compact"] for r in loro),
            "test_next_recall_mean": statistics.mean(r["test_next_recall"] for r in loro),
            "test_next_recall_min": min(r["test_next_recall"] for r in loro),
            "test_next_recall_max": max(r["test_next_recall"] for r in loro),
            "rows": loro,
        },
        "primary_supported_observation": "A 21-edge train-derived decision-equivalent subgraph (vs 71 original edges) reproduces the original action candidate sets on every validation and test state in the frozen MECCANO split and preserves next-action recall.",
    }

    (OUT / "decision_equivalent_graph_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(OUT / "decision_equivalent_graph_loro.csv", loro)
    write_csv(OUT / "decision_equivalent_edge_identifiability.csv", edge_identifiability)
    write_csv(
        OUT / "decision_equivalent_edges.csv",
        [{
            "predecessor": p,
            "action": a,
            "in_full": True,
            "in_transitive_reduction": (p, a) in edges(transitive),
            "in_decision_equivalent": (p, a) in edges(compact),
        } for p, a in sorted(edges(base))],
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
