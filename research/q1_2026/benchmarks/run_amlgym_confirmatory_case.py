from __future__ import annotations

import argparse
import json
import platform
import tempfile
import time
from collections import defaultdict
from importlib.metadata import version as package_version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from paper_a.amlgym_bridge import (  # noqa: E402
    DecisionObservation,
    decision_metrics,
    fit_global_decision_baseline,
    fit_operator_repair,
    parse_action_label,
    predict_global_decision_baseline,
    predict_operator_repair,
    stable_bucket,
)
from paper_a.deployment import (  # noqa: E402
    calibration_gate_from_predictions,
    choose_operator_calibration_gates,
    gated_prediction,
)
from paper_a.protocol import (  # noqa: E402
    assert_stage_disjoint,
    confirmatory_states_excluding_pilot,
    pilot_index_ranked_states,
    state_split,
)


def _action_label(operator: str, args) -> str:
    args = tuple(map(str, args))
    return f"({operator}{(' ' + ' '.join(args)) if args else ''})"


def _sample_ground_actions(
    env,
    *,
    domain: str,
    problem_name: str,
    state_fingerprint: str,
    per_operator: int,
):
    labels = []
    for operator, combos in sorted(env.ground_actions.items()):
        op_labels = [_action_label(operator, args) for args in combos]
        op_labels.sort(
            key=lambda action: stable_bucket(
                f"action|{domain}|{problem_name}|{state_fingerprint}|{action}",
                2**31 - 1,
            )
        )
        labels.extend(op_labels[: min(per_operator, len(op_labels))])
    return labels


def collect_confirmatory_observations(
    *,
    domain: str,
    reference_domain_path: str,
    learned_domain_path: str,
    max_problems: int,
    states_per_problem: int,
    pilot_states_per_problem: int,
    max_actions_per_operator: int,
):
    from amlgym.benchmarks import get_problems_path, get_test_states
    from amlgym.modeling.UPEnv import UPEnv

    all_states = get_test_states(domain, kind="predictive_power")
    problem_paths = get_problems_path(domain, kind="predictive_power")[:max_problems]

    rows: list[tuple[str, DecisionObservation]] = []
    state_split_counts = defaultdict(int)
    observation_split_counts = defaultdict(int)
    selected_state_fingerprints: set[str] = set()
    pilot_state_fingerprints: set[str] = set()
    problem_rows = []

    pilot_by_problem = {}
    for problem_path in problem_paths:
        problem_name = Path(problem_path).name
        pilot = pilot_index_ranked_states(
            all_states[problem_name],
            domain=domain,
            problem_name=problem_name,
            limit=pilot_states_per_problem,
        )
        pilot_by_problem[problem_name] = pilot
        pilot_state_fingerprints.update(x.fingerprint for x in pilot)

    for problem_path in problem_paths:
        problem_name = Path(problem_path).name
        states = all_states[problem_name]
        pilot = pilot_by_problem[problem_name]
        confirmatory = confirmatory_states_excluding_pilot(
            states,
            domain=domain,
            problem_name=problem_name,
            states_per_problem=states_per_problem,
            pilot_states_per_problem=pilot_states_per_problem,
            excluded_fingerprints=pilot_state_fingerprints,
        )
        assert_stage_disjoint(pilot, confirmatory)

        ref_env = UPEnv(reference_domain_path, problem_path)
        learned_env = UPEnv(learned_domain_path, problem_path)

        per_problem_counts = defaultdict(int)
        for item in confirmatory:
            state = item.state_literals
            split = state_split(
                domain=domain, problem_name=problem_name, fingerprint=item.fingerprint
            )
            if item.fingerprint in selected_state_fingerprints:
                raise AssertionError(
                    f"duplicate confirmatory semantic state across problems: {item.fingerprint}"
                )
            selected_state_fingerprints.add(item.fingerprint)
            state_split_counts[split] += 1
            per_problem_counts[split] += 1

            truth = ref_env.applicable_actions(set(state))
            base = learned_env.applicable_actions(set(state))
            actions = _sample_ground_actions(
                ref_env,
                domain=domain,
                problem_name=problem_name,
                state_fingerprint=item.fingerprint,
                per_operator=max_actions_per_operator,
            )
            for action in actions:
                operator, _ = parse_action_label(action)
                rows.append(
                    (
                        split,
                        DecisionObservation(
                            state_literals=state,
                            action_label=action,
                            base_allow=int(action in base.get(operator, set())),
                            truth_allow=int(action in truth.get(operator, set())),
                        ),
                    )
                )
                observation_split_counts[split] += 1

        problem_rows.append(
            {
                "problem": problem_name,
                "available_states": len(states),
                "pilot_unique_selected": len(pilot),
                "confirmatory_unique_selected": len(confirmatory),
                "confirmatory_split_state_counts": dict(sorted(per_problem_counts.items())),
            }
        )

    overlap = selected_state_fingerprints.intersection(pilot_state_fingerprints)
    if overlap:
        raise AssertionError(f"confirmatory/pilot overlap across problems: {len(overlap)}")

    protocol = {
        "stage": "confirmatory",
        "pilot_selector_replay": "historical sha256 index ranking: state|domain|problem|index",
        "state_ranking": "sha256 over domain/problem/semantic-state-fingerprint after excluding actual pilot fingerprints",
        "split_unit": "semantic_state_fingerprint",
        "split_hash": "sha256",
        "split_buckets": {"repair": [0, 500], "calibration": [500, 750], "test": [750, 1000]},
        "pilot_states_per_problem_replayed": pilot_states_per_problem,
        "confirmatory_states_per_problem_requested": states_per_problem,
        "selected_unique_state_count": len(selected_state_fingerprints),
        "pilot_overlap_count": 0,
        "state_split_counts": dict(sorted(state_split_counts.items())),
        "observation_split_counts": dict(sorted(observation_split_counts.items())),
        "problems": problem_rows,
    }
    return rows, protocol


def _split_metrics(rows, deployed_repairs):
    out = {}
    for split in ("repair", "calibration", "test"):
        obs = [o for s, o in rows if s == split]
        base_predictions = [o.base_allow for o in obs]
        dovod_predictions = [gated_prediction(o, deployed_repairs) for o in obs]
        out[split] = {
            "base": decision_metrics(obs, base_predictions),
            "dovod": decision_metrics(obs, dovod_predictions),
        }
    return out


def _global_predictions(rows, global_models, global_gates):
    predictions = {}
    for split in ("repair", "calibration", "test"):
        obs = [o for s, o in rows if s == split]
        values = []
        for o in obs:
            op, _ = parse_action_label(o.action_label)
            model = global_models.get(op)
            gate = global_gates.get(op)
            if model is None or gate is None or not gate.deployed:
                values.append(int(o.base_allow))
            else:
                values.append(predict_global_decision_baseline(model, o))
        predictions[split] = decision_metrics(obs, values)
    return predictions


def run_case(args) -> dict:
    from amlgym.algorithms import get_algorithm
    from amlgym.benchmarks import get_domain_path, get_trajectories_path
    from amlgym.metrics import syntactic_precision, syntactic_recall

    started = time.perf_counter()
    ref_domain = get_domain_path(args.domain)
    trajectories = get_trajectories_path(args.domain, kind="learning")[: args.trace_budget]
    if len(trajectories) < args.trace_budget:
        raise RuntimeError("Insufficient learning trajectories")

    kwargs = {"noise": 0.0} if args.algorithm.lower() == "nolam" else {}
    learner = get_algorithm(args.algorithm, **kwargs)
    learned_text = learner.learn(ref_domain, trajectories)
    learn_seconds = time.perf_counter() - started

    with tempfile.TemporaryDirectory(prefix="dovod-amlgym-confirmatory-") as td:
        learned_path = Path(td) / "learned.pddl"
        learned_path.write_text(str(learned_text), encoding="utf-8")
        syntactic = {
            "precision": syntactic_precision(str(learned_path), ref_domain),
            "recall": syntactic_recall(str(learned_path), ref_domain),
        }

        rows, protocol = collect_confirmatory_observations(
            domain=args.domain,
            reference_domain_path=ref_domain,
            learned_domain_path=str(learned_path),
            max_problems=args.max_problems,
            states_per_problem=args.max_states,
            pilot_states_per_problem=args.pilot_states_per_problem,
            max_actions_per_operator=args.max_actions_per_operator,
        )

        repair_observations = [o for split, o in rows if split == "repair"]
        by_operator = defaultdict(list)
        for observation in repair_observations:
            operator, _ = parse_action_label(observation.action_label)
            by_operator[operator].append(observation)

        global_candidates = {}
        for operator, observations in sorted(by_operator.items()):
            if len(observations) < args.min_repair_samples:
                continue
            global_candidates[operator] = fit_global_decision_baseline(
                operator,
                observations,
                override_penalty=args.edit_penalty,
                false_allow_weight=args.false_allow_weight,
                false_block_weight=args.false_block_weight,
            )

        global_gates = {}
        for operator, model in sorted(global_candidates.items()):
            calibration = tuple(
                o for split, o in rows
                if split == "calibration" and parse_action_label(o.action_label)[0] == operator
            )
            predictions = tuple(predict_global_decision_baseline(model, o) for o in calibration)
            global_gates[operator] = calibration_gate_from_predictions(
                operator,
                calibration,
                predictions,
                selected_edit_count=int(model.policy != "identity"),
                require_false_allow_nonincrease=True,
            )

        candidates = {}
        for operator, observations in sorted(by_operator.items()):
            if len(observations) < args.min_repair_samples:
                continue
            candidates[operator] = fit_operator_repair(
                operator,
                observations,
                max_features=args.max_features,
                max_context_width=args.context_width,
                edit_penalty=args.edit_penalty,
                false_allow_weight=args.false_allow_weight,
                false_block_weight=args.false_block_weight,
            )

        deployed, gates = choose_operator_calibration_gates(
            rows,
            candidates,
            require_false_allow_nonincrease=True,
        )
        decision = _split_metrics(rows, deployed)

        candidate_metrics = _split_metrics(rows, candidates)
        global_metrics = _global_predictions(rows, global_candidates, global_gates)
        for split in decision:
            decision[split]["candidate_ungated"] = candidate_metrics[split]["dovod"]
            decision[split]["global_override_gated"] = global_metrics[split]

        gate_json = {
            operator: {
                "deployed": gate.deployed,
                "reason": gate.reason,
                "base": gate.base_metrics,
                "candidate": gate.candidate_metrics,
                "selected_edit_count": gate.selected_edit_count,
            }
            for operator, gate in sorted(gates.items())
        }
        global_baseline = {
            operator: {
                "policy": model.policy,
                "repair_objective": model.objective,
                "calibration_gate": {
                    "deployed": global_gates[operator].deployed,
                    "reason": global_gates[operator].reason,
                    "base": global_gates[operator].base_metrics,
                    "candidate": global_gates[operator].candidate_metrics,
                },
            }
            for operator, model in sorted(global_candidates.items())
        }
        repairs = {
            operator: {
                "repair_samples": len(by_operator[operator]),
                "features": list(model.feature_names),
                "vocabulary_size": len(model.vocabulary),
                "selected_edit_count": len(model.fit.selected_edits),
                "fit_errors": len(model.fit.error_indices),
                "calibration_gate": gate_json.get(operator),
            }
            for operator, model in sorted(candidates.items())
        }

    return {
        "schema": "dovod-q1-amlgym-confirmatory-case-v1",
        "status": "ok",
        "domain": args.domain,
        "algorithm": args.algorithm,
        "trace_budget": args.trace_budget,
        "learning_trajectories": len(trajectories),
        "amlgym_version": package_version("amlgym"),
        "python": platform.python_version(),
        "learn_seconds": learn_seconds,
        "wall_seconds": time.perf_counter() - started,
        "syntactic": syntactic,
        "protocol": protocol,
        "decision": decision,
        "global_baseline": global_baseline,
        "repairs": repairs,
        "claim_boundary": (
            "Confirmatory evaluation explicitly replays the historical pilot index-hash selector and excludes every selected pilot state fingerprint (including duplicates) before choosing the new semantic-state window. "
            "Candidate repairs are fitted only on the repair split; operator deployment is decided only on calibration data; "
            "test labels are used only after deployment is frozen. The gate requires strict calibration-risk improvement and "
            "forbids an increase in calibration false-allow count. DOVOD repairs applicability decisions only."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--trace-budget", type=int, required=True)
    parser.add_argument("--max-problems", type=int, default=2)
    parser.add_argument("--max-states", type=int, default=12)
    parser.add_argument("--pilot-states-per-problem", type=int, default=12)
    parser.add_argument("--max-actions-per-operator", type=int, default=4)
    parser.add_argument("--min-repair-samples", type=int, default=4)
    parser.add_argument("--max-features", type=int, default=8)
    parser.add_argument("--context-width", type=int, default=1)
    parser.add_argument("--edit-penalty", type=float, default=0.25)
    parser.add_argument("--false-allow-weight", type=float, default=1.0)
    parser.add_argument("--false-block-weight", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        report = run_case(args)
    except Exception as exc:
        report = {
            "schema": "dovod-q1-amlgym-confirmatory-case-v1",
            "status": "failed",
            "domain": args.domain,
            "algorithm": args.algorithm,
            "trace_budget": args.trace_budget,
            "failure_stage": "confirmatory_case",
            "error": f"{type(exc).__name__}: {exc}",
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
