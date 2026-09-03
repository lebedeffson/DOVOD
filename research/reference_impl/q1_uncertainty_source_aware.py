from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from e4_real_state_calibrated_observation import train_state_priors
from hardening_loro_graph import action_set, read_records
from joint_semantic_physical_uncertainty import (
    action_uncertainty,
    best_physical_intervention,
    best_semantic_intervention,
)
from q1_decision_equivalent_graph import (
    enumerate_optimal_action_covers,
    load_full_dependencies,
    split_states,
)
from q1_semantic_version_space import assemble_graphs
from recording_bootstrap import paired_recording_bootstrap

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "source_aware"
OUT.mkdir(parents=True, exist_ok=True)
EPS = 1e-12


def _hash_rank(recording: str, frame: str, action: int, component: int) -> int:
    raw = f"{recording}|{frame}|{action}|{component}|source-aware-v1".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def completion_vector(state: list[int], masked: list[int], priors: np.ndarray) -> np.ndarray:
    q = np.asarray([1.0 if int(x) == 1 else 0.0 for x in state], dtype=float)
    for j in masked:
        q[int(j)] = float(priors[int(j), 2])
    return q


def action_union_prerequisites(graphs: list[dict[int, list[int]]], action: int) -> list[int]:
    return sorted({int(p) for g in graphs for p in g.get(int(action), [])})


def action_truth(state: list[int], graph: dict[int, list[int]], action: int) -> int:
    return int(int(state[action]) != 1 and all(int(state[p]) == 1 for p in graph.get(action, [])))


def collapse_graphs_for_semantic_review(
    active_graphs: list[dict[int, list[int]]], latent_graph: dict[int, list[int]], action: int
) -> list[dict[int, list[int]]]:
    target = tuple(sorted(latent_graph.get(int(action), [])))
    out = [g for g in active_graphs if tuple(sorted(g.get(int(action), []))) == target]
    if not out:
        raise AssertionError("Semantic review eliminated the latent semantics")
    return out


def run_policy(
    *,
    policy: str,
    q0: np.ndarray,
    masked: list[int],
    state: list[int],
    action: int,
    graphs: list[dict[int, list[int]]],
    latent_graph_index: int,
    semantic_cost: float = 1.0,
    physical_cost: float = 1.0,
) -> dict:
    q = q0.copy()
    active = list(graphs)
    latent = graphs[int(latent_graph_index)]
    unresolved = set(int(x) for x in masked)
    physical_queries = 0
    semantic_queries = 0
    trace: list[str] = []
    initial = action_uncertainty(q, action, active)

    for _ in range(len(masked) + 3):
        current = action_uncertainty(q, action, active).total_variance
        if current <= EPS:
            break
        pv = (
            best_physical_intervention(q, sorted(unresolved), active, actions=[action], normalize=False)
            if unresolved
            else None
        )
        sv = best_semantic_intervention(q, [action], active, actions=[action], normalize=False)

        choose = None
        if policy == "physical_first":
            if pv is not None and pv.expected_reduction > EPS:
                choose = ("physical", pv)
            elif sv is not None and sv.expected_reduction > EPS:
                choose = ("semantic", sv)
        elif policy == "semantic_first":
            if sv is not None and sv.expected_reduction > EPS:
                choose = ("semantic", sv)
            elif pv is not None and pv.expected_reduction > EPS:
                choose = ("physical", pv)
        elif policy.startswith("source_aware"):
            options = []
            if pv is not None and pv.expected_reduction > EPS:
                options.append((pv.expected_reduction / float(physical_cost), "physical", pv))
            if sv is not None and sv.expected_reduction > EPS:
                options.append((sv.expected_reduction / float(semantic_cost), "semantic", sv))
            if options:
                _, kind, val = max(options, key=lambda x: (x[0], x[1] == "semantic", -x[2].target))
                choose = (kind, val)
        else:
            raise ValueError(policy)

        if choose is None:
            break
        kind, val = choose
        if kind == "physical":
            j = int(val.target)
            q[j] = 1.0 if int(state[j]) == 1 else 0.0
            unresolved.discard(j)
            physical_queries += 1
            trace.append(f"PHYSICAL:{j}")
        else:
            active = collapse_graphs_for_semantic_review(active, latent, action)
            semantic_queries += 1
            trace.append(f"SEMANTIC:{action}")

    final_u = action_uncertainty(q, action, active)
    latent_truth = action_truth(state, latent, action)
    p = float(final_u.admissibility_probability)
    if final_u.total_variance <= EPS:
        predicted = int(p >= 0.5)
        resolved = 1
        correct = int(predicted == latent_truth)
    else:
        predicted = -1
        resolved = 0
        correct = 0

    return {
        "policy": policy,
        "physical_queries": physical_queries,
        "semantic_queries": semantic_queries,
        "total_interventions": physical_queries + semantic_queries,
        "weighted_intervention_cost": physical_queries * physical_cost + semantic_queries * semantic_cost,
        "resolved": resolved,
        "correct_when_resolved": correct,
        "final_total_variance": final_u.total_variance,
        "initial_total_variance": initial.total_variance,
        "initial_semantic_variance": initial.semantic_variance,
        "initial_physical_variance": initial.physical_variance,
        "initial_semantic_fraction": initial.semantic_variance / initial.total_variance if initial.total_variance > EPS else 0.0,
        "trace": ">".join(trace),
    }


def build_episodes():
    recs = read_records()
    base = load_full_dependencies()
    covers = enumerate_optimal_action_covers(base, split_states(recs, "train"))
    graphs = assemble_graphs(base, covers)
    priors = train_state_priors()
    rows = []

    for split, rec, rec_rows in recs:
        if split != "test":
            continue
        for frame, state in rec_rows:
            possible = sorted(set().union(*(set(action_set(state, g)) for g in graphs)))
            fully_known_q = np.asarray([1.0 if int(x) == 1 else 0.0 for x in state], dtype=float)
            for action in possible:
                relevant = sorted(set(action_union_prerequisites(graphs, action)) | {int(action)})
                ranked = sorted(relevant, key=lambda j: (_hash_rank(rec, frame, action, j), j))
                max_k = min(3, len(ranked))
                full_semantic = action_uncertainty(fully_known_q, action, graphs)
                for k in range(1, max_k + 1):
                    masked = ranked[:k]
                    q = completion_vector(state, masked, priors)
                    u = action_uncertainty(q, action, graphs)
                    if u.total_variance <= EPS:
                        continue
                    pv = best_physical_intervention(q, masked, graphs, actions=[action], normalize=False)
                    sv = best_semantic_intervention(q, [action], graphs, actions=[action], normalize=False)
                    rows.append(
                        {
                            "recording": rec,
                            "frame": frame,
                            "action": int(action),
                            "mask_k": int(k),
                            "masked": ";".join(map(str, masked)),
                            "masked_list": masked,
                            "state": state,
                            "q": q,
                            "initial_total_variance": u.total_variance,
                            "initial_semantic_variance": u.semantic_variance,
                            "initial_physical_variance": u.physical_variance,
                            "initial_semantic_fraction": u.semantic_variance / u.total_variance,
                            "best_physical_reduction": pv.expected_reduction if pv else 0.0,
                            "best_physical_target": pv.target if pv else -1,
                            "semantic_review_reduction": sv.expected_reduction if sv else 0.0,
                            "oracle_state_semantic_variance": full_semantic.semantic_variance,
                            "oracle_state_semantic_ambiguity": int(full_semantic.semantic_variance > EPS),
                        }
                    )
    return rows, graphs


def initial_summary_df(episodes: list[dict]) -> pd.DataFrame:
    fields = [
        "recording", "frame", "action", "mask_k", "masked",
        "initial_total_variance", "initial_semantic_variance", "initial_physical_variance",
        "initial_semantic_fraction", "best_physical_reduction", "best_physical_target",
        "semantic_review_reduction", "oracle_state_semantic_variance", "oracle_state_semantic_ambiguity",
    ]
    return pd.DataFrame([{k: e[k] for k in fields} for e in episodes])


def summarize_initial(df: pd.DataFrame) -> dict:
    return {
        "episodes": int(len(df)),
        "recordings": int(df["recording"].nunique()),
        "mean_total_variance": float(df["initial_total_variance"].mean()),
        "mean_semantic_variance": float(df["initial_semantic_variance"].mean()),
        "mean_physical_variance": float(df["initial_physical_variance"].mean()),
        "mean_semantic_fraction": float(df["initial_semantic_fraction"].mean()),
        "episodes_with_semantic_component": int((df["initial_semantic_variance"] > EPS).sum()),
        "episodes_with_physical_component": int((df["initial_physical_variance"] > EPS).sum()),
        "semantic_review_dominates_physical_equal_cost": int((df["semantic_review_reduction"] > df["best_physical_reduction"] + EPS).sum()),
        "physical_query_dominates_semantic_equal_cost": int((df["best_physical_reduction"] > df["semantic_review_reduction"] + EPS).sum()),
        "ties": int((np.abs(df["best_physical_reduction"] - df["semantic_review_reduction"]) <= EPS).sum()),
        "semantic_challenge_episodes": int(df["oracle_state_semantic_ambiguity"].sum()),
        "semantic_challenge_action_instances": int(df[df["oracle_state_semantic_ambiguity"] == 1][["recording", "frame", "action"]].drop_duplicates().shape[0]),
    }


def main() -> None:
    episodes, graphs = build_episodes()
    init_df = initial_summary_df(episodes)
    init_df.to_csv(OUT / "uncertainty_source_decomposition.csv", index=False)

    challenge_eps = [e for e in episodes if e["oracle_state_semantic_ambiguity"]]
    policies = [
        ("physical_first", 1.0),
        ("semantic_first", 1.0),
        ("source_aware_c1", 1.0),
        ("source_aware_c2", 2.0),
        ("source_aware_c5", 5.0),
    ]
    rows = []
    for eid, ep in enumerate(challenge_eps):
        for latent_idx in range(len(graphs)):
            for policy, sem_cost in policies:
                r = run_policy(
                    policy=policy,
                    q0=ep["q"],
                    masked=ep["masked_list"],
                    state=ep["state"],
                    action=ep["action"],
                    graphs=graphs,
                    latent_graph_index=latent_idx,
                    semantic_cost=sem_cost,
                    physical_cost=1.0,
                )
                rows.append(
                    {
                        "episode": eid,
                        "latent_graph": latent_idx,
                        "recording": ep["recording"],
                        "frame": ep["frame"],
                        "action": ep["action"],
                        "mask_k": ep["mask_k"],
                        "masked": ep["masked"],
                        **r,
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "semantic_challenge_intervention_episodes.csv", index=False)
    summary = (
        df.groupby("policy", as_index=False)
        .agg(
            episodes=("episode", "size"),
            mean_physical_queries=("physical_queries", "mean"),
            mean_semantic_queries=("semantic_queries", "mean"),
            mean_total_interventions=("total_interventions", "mean"),
            mean_weighted_cost=("weighted_intervention_cost", "mean"),
            resolution_rate=("resolved", "mean"),
            correctness_rate=("correct_when_resolved", "mean"),
            final_variance=("final_total_variance", "mean"),
        )
    )
    summary.to_csv(OUT / "semantic_challenge_intervention_summary.csv", index=False)

    boot = {}
    for policy in ("source_aware_c1", "source_aware_c2", "source_aware_c5", "semantic_first"):
        for col in ("physical_queries", "total_interventions"):
            boot[f"{policy}_vs_physical_first_{col}"] = paired_recording_bootstrap(
                df,
                policy,
                "physical_first",
                col,
                index_cols=("latent_graph", "recording", "frame", "action", "mask_k"),
                n_boot=4000,
                seed=20260901 + len(boot),
            )

    fixed_errors = 0
    fixed_total = 0
    seen_instances = set()
    for ep in episodes:
        key = (ep["recording"], ep["frame"], ep["action"])
        if key in seen_instances:
            continue
        seen_instances.add(key)
        fixed = action_truth(ep["state"], graphs[0], ep["action"])
        for latent in graphs:
            fixed_total += 1
            fixed_errors += int(fixed != action_truth(ep["state"], latent, ep["action"]))

    traces = Counter(df[df["policy"] == "source_aware_c1"]["trace"])
    report = {
        "schema": "tinyapv-uncertainty-source-aware-v1",
        "derivation_boundary": (
            "Semantic version space is derived from TRAIN states only. Physical uncertainty is a controlled missing-evidence stress on frozen MECCANO test states using TRAIN completion priors. Sequential physical interventions are perfect completion reveals to isolate intervention-source logic; they are not real RGB evidence."
        ),
        "semantic_version_space_graphs": len(graphs),
        "initial_uncertainty": summarize_initial(init_df),
        "fixed_arbitrary_minimum_graph_best_perception_semantic_error_rate": fixed_errors / fixed_total,
        "fixed_arbitrary_minimum_graph_best_perception_semantic_errors": fixed_errors,
        "fixed_arbitrary_minimum_graph_best_perception_comparisons": fixed_total,
        "semantic_challenge_summary": summary.to_dict(orient="records"),
        "semantic_challenge_recording_cluster_bootstrap": boot,
        "source_aware_c1_common_traces": traces.most_common(20),
        "claim_boundary": (
            "The new result is a decision-layer uncertainty decomposition and targeted-intervention mechanism. It separates physical-state uncertainty from uncertainty over train-equivalent procedure semantics and shows when physical sensing cannot substitute for semantic validation. No real sensor/RGB/headset/energy/human or mechanically authoritative semantics claim follows from this controlled missing-evidence block."
        ),
    }
    (OUT / "source_aware_intervention_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
