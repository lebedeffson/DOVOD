from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import time
from importlib.metadata import version as package_version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from paper_a.amlgym_bridge import (  # noqa: E402
    DecisionObservation,
    decision_metrics,
    fit_operator_repair,
    parse_action_label,
    predict_operator_repair,
    stable_bucket,
)


def _action_label(operator: str, args) -> str:
    args = tuple(map(str, args))
    return f"({operator}{(' ' + ' '.join(args)) if args else ''})"


def _sample_states(states, *, domain: str, problem_name: str, limit: int):
    ranked = sorted(
        enumerate(states),
        key=lambda item: stable_bucket(f"state|{domain}|{problem_name}|{item[0]}", 2**31 - 1),
    )
    return ranked[: min(limit, len(ranked))]


def _sample_ground_actions(env, *, domain: str, problem_name: str, state_index: int, per_operator: int):
    labels = []
    for operator, combos in sorted(env.ground_actions.items()):
        op_labels = [_action_label(operator, args) for args in combos]
        op_labels.sort(
            key=lambda a: stable_bucket(
                f"action|{domain}|{problem_name}|{state_index}|{a}", 2**31 - 1
            )
        )
        labels.extend(op_labels[: min(per_operator, len(op_labels))])
    return labels


def collect_decision_observations(
    *,
    domain: str,
    reference_domain_path: str,
    learned_domain_path: str,
    max_problems: int,
    max_states_per_problem: int,
    max_actions_per_operator: int,
):
    from amlgym.benchmarks import get_problems_path, get_test_states
    from amlgym.modeling.UPEnv import UPEnv

    all_states = get_test_states(domain, kind="predictive_power")
    problem_paths = get_problems_path(domain, kind="predictive_power")[:max_problems]
    rows = []
    for problem_path in problem_paths:
        problem_name = Path(problem_path).name
        states = all_states[problem_name]
        ref_env = UPEnv(reference_domain_path, problem_path)
        learned_env = UPEnv(learned_domain_path, problem_path)
        for state_index, state in _sample_states(
            states, domain=domain, problem_name=problem_name, limit=max_states_per_problem
        ):
            state = tuple(map(str, state))
            truth = ref_env.applicable_actions(set(state))
            base = learned_env.applicable_actions(set(state))
            actions = _sample_ground_actions(
                ref_env,
                domain=domain,
                problem_name=problem_name,
                state_index=state_index,
                per_operator=max_actions_per_operator,
            )
            for action in actions:
                operator, _ = parse_action_label(action)
                truth_allow = int(action in truth.get(operator, set()))
                base_allow = int(action in base.get(operator, set()))
                key = f"pair|{domain}|{problem_name}|{state_index}|{action}"
                bucket = stable_bucket(key, 1000)
                split = "repair" if bucket < 500 else ("calibration" if bucket < 750 else "test")
                rows.append(
                    (
                        split,
                        DecisionObservation(
                            state_literals=state,
                            action_label=action,
                            base_allow=base_allow,
                            truth_allow=truth_allow,
                        ),
                    )
                )
    return rows


def _gate_repairs_by_calibration(rows, repaired_models):
    """Deploy only repairs that strictly improve independent calibration risk.

    The repair split proposes sparse edits. Calibration is then used only as a
    deployment gate, never to construct features or edit candidates. Non-empty edits
    that merely tie the upstream model are rejected to avoid gratuitous modification.
    """
    deployed = {}
    gate = {}
    for op, model in sorted(repaired_models.items()):
        obs = [
            o for split, o in rows
            if split == "calibration" and parse_action_label(o.action_label)[0] == op
        ]
        if not obs:
            gate[op] = {"deployed": False, "reason": "no_calibration_samples"}
            continue
        base_metrics = decision_metrics(obs, [o.base_allow for o in obs])
        preds = [predict_operator_repair(model, o) for o in obs]
        repair_metrics = decision_metrics(obs, preds)
        improve = repair_metrics["risk"] < base_metrics["risk"] - 1e-15
        unchanged_empty = (
            not model.fit.selected_edits and repair_metrics["risk"] == base_metrics["risk"]
        )
        use = bool(improve or unchanged_empty)
        if use:
            deployed[op] = model
        gate[op] = {
            "deployed": use,
            "base_risk": base_metrics["risk"],
            "repair_risk": repair_metrics["risk"],
            "base_class_balanced_risk": base_metrics["class_balanced_risk"],
            "repair_class_balanced_risk": repair_metrics["class_balanced_risk"],
            "selected_edit_count": len(model.fit.selected_edits),
            "reason": (
                "strict_calibration_improvement" if improve
                else ("empty_noop" if unchanged_empty else "no_strict_calibration_gain")
            ),
        }
    return deployed, gate


def _split_metrics(rows, repaired_models):
    out = {}
    for split in ("repair", "calibration", "test"):
        obs = [o for s, o in rows if s == split]
        base_preds = [o.base_allow for o in obs]
        repaired_preds = []
        for o in obs:
            op, _ = parse_action_label(o.action_label)
            model = repaired_models.get(op)
            repaired_preds.append(o.base_allow if model is None else predict_operator_repair(model, o))
        out[split] = {
            "base": decision_metrics(obs, base_preds),
            "dovod": decision_metrics(obs, repaired_preds),
        }
    return out


def run_case(args) -> dict:
    from amlgym.algorithms import get_algorithm
    from amlgym.benchmarks import get_domain_path, get_trajectories_path
    from amlgym.metrics import syntactic_precision, syntactic_recall

    t0 = time.perf_counter()
    ref_domain = get_domain_path(args.domain)
    trajectories = get_trajectories_path(args.domain, kind="learning")[: args.trace_budget]
    if len(trajectories) < args.trace_budget:
        raise RuntimeError(
            f"{args.domain}: requested {args.trace_budget} traces but only {len(trajectories)} available"
        )

    kwargs = {"noise": 0.0} if args.algorithm.lower() == "nolam" else {}
    learner = get_algorithm(args.algorithm, **kwargs)
    learned_text = learner.learn(ref_domain, trajectories)
    learn_seconds = time.perf_counter() - t0

    with tempfile.TemporaryDirectory(prefix="dovod-amlgym-") as td:
        learned_path = Path(td) / f"{args.domain}_{args.algorithm}.pddl"
        learned_path.write_text(str(learned_text), encoding="utf-8")
        syntactic = {
            "precision": syntactic_precision(str(learned_path), ref_domain),
            "recall": syntactic_recall(str(learned_path), ref_domain),
        }

        pred_error = None
        try:
            rows = collect_decision_observations(
                domain=args.domain,
                reference_domain_path=ref_domain,
                learned_domain_path=str(learned_path),
                max_problems=args.max_problems,
                max_states_per_problem=args.max_states,
                max_actions_per_operator=args.max_actions_per_operator,
            )
            repair_obs = [o for split, o in rows if split == "repair"]
            by_operator = {}
            for o in repair_obs:
                op, _ = parse_action_label(o.action_label)
                by_operator.setdefault(op, []).append(o)
            repairs = {}
            for op, obs in sorted(by_operator.items()):
                if len(obs) < args.min_repair_samples:
                    continue
                repairs[op] = fit_operator_repair(
                    op,
                    obs,
                    max_features=args.max_features,
                    max_context_width=args.context_width,
                    edit_penalty=args.edit_penalty,
                )
            deployed_repairs, calibration_gate = _gate_repairs_by_calibration(rows, repairs)
            decision = _split_metrics(rows, deployed_repairs)
            ungated_decision = _split_metrics(rows, repairs)
            for split in decision:
                decision[split]["dovod_ungated"] = ungated_decision[split]["dovod"]
            repair_summary = {
                op: {
                    "repair_samples": len(by_operator[op]),
                    "features": list(model.feature_names),
                    "vocabulary_size": len(model.vocabulary),
                    "selected_edits": [
                        {
                            "kind": e.kind,
                            "context": [list(x) for x in e.context],
                            "prerequisite": e.prerequisite,
                            "weight": e.weight,
                        }
                        for e in model.fit.selected_edits
                    ],
                    "fit_errors": len(model.fit.error_indices),
                    "calibration_gate": calibration_gate.get(op),
                }
                for op, model in repairs.items()
            }
        except Exception as exc:
            decision = None
            repair_summary = None
            pred_error = f"{type(exc).__name__}: {exc}"

    return {
        "schema": "dovod-q1-amlgym-case-v3",
        "status": "ok",
        "domain": args.domain,
        "algorithm": args.algorithm,
        "trace_budget": args.trace_budget,
        "learning_trajectories": len(trajectories),
        "amlgym_version": package_version("amlgym"),
        "python": platform.python_version(),
        "learn_seconds": learn_seconds,
        "syntactic": syntactic,
        "decision": decision,
        "repairs": repair_summary,
        "predictive_bridge_error": pred_error,
        "claim_boundary": (
            "The learned PDDL is produced by the named AMLGym learner. DOVOD is evaluated only as a "
            "post-hoc applicability decision repair on a hash-frozen state/action split. Non-empty repairs "
            "are deployed only after strict independent calibration-risk improvement. It does not alter "
            "effects or claim downstream planning improvement in this case result."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True)
    p.add_argument("--algorithm", required=True)
    p.add_argument("--trace-budget", type=int, required=True)
    p.add_argument("--max-problems", type=int, default=2)
    p.add_argument("--max-states", type=int, default=12)
    p.add_argument("--max-actions-per-operator", type=int, default=4)
    p.add_argument("--min-repair-samples", type=int, default=4)
    p.add_argument("--max-features", type=int, default=8)
    p.add_argument("--context-width", type=int, default=1)
    p.add_argument("--edit-penalty", type=float, default=0.25)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    try:
        report = run_case(args)
    except Exception as exc:
        report = {
            "schema": "dovod-q1-amlgym-case-v3",
            "status": "failed",
            "domain": args.domain,
            "algorithm": args.algorithm,
            "trace_budget": args.trace_budget,
            "error": f"{type(exc).__name__}: {exc}",
        }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
