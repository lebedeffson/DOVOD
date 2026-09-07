from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_amlgym_confirmatory_case import collect_confirmatory_observations
from paper_a.amlgym_bridge import decision_metrics, fit_operator_repair, parse_action_label
from paper_a.deployment import calibration_gate_from_predictions, choose_operator_calibration_gates, gated_prediction
from paper_a.simple_baselines import (
    fit_counterexample_frequency_single_edit,
    fit_random_single_edit,
    predict_single_edit,
    serialize_single_edit,
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2**32 - 1))
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


def _fit_simple(rows, args):
    repair = defaultdict(list)
    for split, observation in rows:
        if split == "repair":
            operator, _ = parse_action_label(observation.action_label)
            repair[operator].append(observation)
    random_models = {}
    frequency_models = {}
    for operator, observations in sorted(repair.items()):
        if len(observations) < args.min_repair_samples:
            continue
        random_models[operator] = fit_random_single_edit(
            operator,
            observations,
            seed=args.seed,
            max_features=args.max_features,
            max_context_width=args.context_width,
        )
        frequency_models[operator] = fit_counterexample_frequency_single_edit(
            operator,
            observations,
            max_features=args.max_features,
            max_context_width=args.context_width,
        )
    return repair, random_models, frequency_models


def _gate_simple(rows, models):
    gates = {}
    for operator, model in sorted(models.items()):
        if model.edit is None:
            continue
        calibration = tuple(
            observation
            for split, observation in rows
            if split == "calibration" and parse_action_label(observation.action_label)[0] == operator
        )
        predictions = tuple(predict_single_edit(model, observation) for observation in calibration)
        gates[operator] = calibration_gate_from_predictions(
            operator,
            calibration,
            predictions,
            selected_edit_count=1,
            require_false_allow_nonincrease=True,
        )
    return gates


def _predict_gated_simple(observation, models, gates) -> int:
    operator, _ = parse_action_label(observation.action_label)
    model = models.get(operator)
    gate = gates.get(operator)
    if model is None or model.edit is None or gate is None or not gate.deployed:
        return int(observation.base_allow)
    return predict_single_edit(model, observation)


def _metrics(rows, dovod_deployed, random_models, random_gates, frequency_models, frequency_gates):
    report = {}
    for split in ("repair", "calibration", "test"):
        observations = [observation for name, observation in rows if name == split]
        report[split] = {
            "base": decision_metrics(observations, [o.base_allow for o in observations]),
            "dovod": decision_metrics(observations, [gated_prediction(o, dovod_deployed) for o in observations]),
            "random_single_edit_gated": decision_metrics(
                observations,
                [_predict_gated_simple(o, random_models, random_gates) for o in observations],
            ),
            "frequency_single_edit_gated": decision_metrics(
                observations,
                [_predict_gated_simple(o, frequency_models, frequency_gates) for o in observations],
            ),
        }
    return report


def run(args) -> dict:
    from amlgym.algorithms import get_algorithm
    from amlgym.benchmarks import get_domain_path, get_trajectories_path

    os.environ.setdefault("PYTHONHASHSEED", "0")
    _seed_everything(args.seed)
    started = time.perf_counter()
    reference_domain = get_domain_path(args.domain)
    trajectories = get_trajectories_path(args.domain, kind="learning")[:args.trace_budget]
    if len(trajectories) < args.trace_budget:
        raise RuntimeError("Insufficient learning trajectories")
    learner = get_algorithm(args.algorithm, **({"noise": 0.0} if args.algorithm.lower() == "nolam" else {}))
    learned_text = learner.learn(reference_domain, trajectories)
    learn_seconds = time.perf_counter() - started

    with tempfile.TemporaryDirectory(prefix="dovod-simple-baselines-") as td:
        learned_path = Path(td) / "learned.pddl"
        learned_path.write_text(str(learned_text), encoding="utf-8")
        rows, protocol = collect_confirmatory_observations(
            domain=args.domain,
            reference_domain_path=reference_domain,
            learned_domain_path=str(learned_path),
            max_problems=args.max_problems,
            states_per_problem=args.max_states,
            pilot_states_per_problem=args.pilot_states_per_problem,
            max_actions_per_operator=args.max_actions_per_operator,
        )

        repair, random_models, frequency_models = _fit_simple(rows, args)
        dovod_candidates = {}
        for operator, observations in sorted(repair.items()):
            if len(observations) < args.min_repair_samples:
                continue
            dovod_candidates[operator] = fit_operator_repair(
                operator,
                observations,
                max_features=args.max_features,
                max_context_width=args.context_width,
                edit_penalty=args.edit_penalty,
                false_allow_weight=1.0,
                false_block_weight=1.0,
            )
        dovod_deployed, dovod_gates = choose_operator_calibration_gates(
            rows, dovod_candidates, require_false_allow_nonincrease=True
        )
        random_gates = _gate_simple(rows, random_models)
        frequency_gates = _gate_simple(rows, frequency_models)
        metrics = _metrics(
            rows,
            dovod_deployed,
            random_models,
            random_gates,
            frequency_models,
            frequency_gates,
        )

    def model_rows(models, gates):
        out = {}
        for operator, model in sorted(models.items()):
            gate = gates.get(operator)
            row = serialize_single_edit(model)
            row["calibration_gate"] = None if gate is None else {
                "deployed": gate.deployed,
                "reason": gate.reason,
                "base": gate.base_metrics,
                "candidate": gate.candidate_metrics,
            }
            out[operator] = row
        return out

    return {
        "schema": "dovod-q1-amlgym-simple-baseline-case-v1",
        "status": "ok",
        "domain": args.domain,
        "algorithm": args.algorithm,
        "trace_budget": args.trace_budget,
        "seed": args.seed,
        "learning_trajectories": len(trajectories),
        "learn_seconds": learn_seconds,
        "wall_seconds": time.perf_counter() - started,
        "protocol": protocol,
        "metrics": metrics,
        "dovod_deployed_operator_count": len(dovod_deployed),
        "dovod_gate_count": len(dovod_gates),
        "random_single_edit": model_rows(random_models, random_gates),
        "frequency_single_edit": model_rows(frequency_models, frequency_gates),
        "claim_boundary": (
            "Post-confirmatory comparator study. Random and counterexample-frequency baselines use the same label-free local feature/vocabulary construction as DOVOD, at most one edit per operator, repair data only for fitting, and the same calibration-only deployment gate. Test labels do not select any method or parameter. Results do not replace the frozen AMLGym primary analysis."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--algorithm", required=True)
    ap.add_argument("--trace-budget", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--max-problems", type=int, default=2)
    ap.add_argument("--max-states", type=int, default=12)
    ap.add_argument("--pilot-states-per-problem", type=int, default=12)
    ap.add_argument("--max-actions-per-operator", type=int, default=4)
    ap.add_argument("--min-repair-samples", type=int, default=4)
    ap.add_argument("--max-features", type=int, default=8)
    ap.add_argument("--context-width", type=int, default=1)
    ap.add_argument("--edit-penalty", type=float, default=0.25)
    args = ap.parse_args()
    try:
        report = run(args)
    except Exception as exc:
        report = {
            "schema": "dovod-q1-amlgym-simple-baseline-case-v1",
            "status": "failed",
            "domain": args.domain,
            "algorithm": args.algorithm,
            "trace_budget": args.trace_budget,
            "failure_stage": "post_confirmatory_simple_baseline_case",
            "error": f"{type(exc).__name__}: {exc}",
            "claim_boundary": "Failure retained as an outcome; no scientific metric is imputed.",
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report.get("status") != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
