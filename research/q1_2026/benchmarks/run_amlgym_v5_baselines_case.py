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

from benchmarks.run_amlgym_confirmatory_case import collect_confirmatory_observations  # noqa: E402
from paper_a.amlgym_bridge import (  # noqa: E402
    decision_metrics,
    fit_operator_repair,
    parse_action_label,
    predict_operator_repair,
)
from paper_a.deployment import (  # noqa: E402
    calibration_gate_from_predictions,
    choose_operator_calibration_gates,
)
from paper_a.simple_baselines import (  # noqa: E402
    fit_frequency_one_edit,
    predict_one_edit,
    random_one_edit,
)


def _calibration_for(rows, operator):
    return tuple(
        obs for split, obs in rows
        if split == "calibration" and parse_action_label(obs.action_label)[0] == operator
    )


def _one_edit_gate(operator, model, rows):
    calibration = _calibration_for(rows, operator)
    predictions = tuple(predict_one_edit(model, obs) for obs in calibration)
    return calibration_gate_from_predictions(
        operator,
        calibration,
        predictions,
        selected_edit_count=int(model.mode != "identity"),
        require_false_allow_nonincrease=True,
    )


def _predict_one_edit_gated(observation, models, gates):
    operator, _ = parse_action_label(observation.action_label)
    model = models.get(operator)
    gate = gates.get(operator)
    if model is None or gate is None or not gate.deployed:
        return int(observation.base_allow)
    return predict_one_edit(model, observation)


def _metrics(rows, *, dovod=None, heuristic_models=None, heuristic_gates=None):
    out = {}
    for split in ("repair", "calibration", "test"):
        obs = [o for s, o in rows if s == split]
        if dovod is not None:
            preds = []
            for o in obs:
                operator, _ = parse_action_label(o.action_label)
                model = dovod.get(operator)
                preds.append(int(o.base_allow) if model is None else predict_operator_repair(model, o))
        else:
            preds = [
                _predict_one_edit_gated(o, heuristic_models or {}, heuristic_gates or {})
                for o in obs
            ]
        out[split] = decision_metrics(obs, preds)
    return out


def _gate_json(gates):
    return {
        operator: {
            "deployed": gate.deployed,
            "reason": gate.reason,
            "selected_edit_count": gate.selected_edit_count,
            "base": gate.base_metrics,
            "candidate": gate.candidate_metrics,
        }
        for operator, gate in sorted(gates.items())
    }


def run_case(args) -> dict:
    from amlgym.algorithms import get_algorithm
    from amlgym.benchmarks import get_domain_path, get_trajectories_path

    started = time.perf_counter()
    ref_domain = get_domain_path(args.domain)
    trajectories = get_trajectories_path(args.domain, kind="learning")[: args.trace_budget]
    if len(trajectories) < args.trace_budget:
        raise RuntimeError("Insufficient learning trajectories")

    kwargs = {"noise": 0.0} if args.algorithm.lower() == "nolam" else {}
    learner = get_algorithm(args.algorithm, **kwargs)
    learned_text = learner.learn(ref_domain, trajectories)
    learn_seconds = time.perf_counter() - started

    with tempfile.TemporaryDirectory(prefix="dovod-amlgym-v5-baselines-") as td:
        learned_path = Path(td) / "learned.pddl"
        learned_path.write_text(str(learned_text), encoding="utf-8")
        rows, protocol = collect_confirmatory_observations(
            domain=args.domain,
            reference_domain_path=ref_domain,
            learned_domain_path=str(learned_path),
            max_problems=args.max_problems,
            states_per_problem=args.max_states,
            pilot_states_per_problem=args.pilot_states_per_problem,
            max_actions_per_operator=args.max_actions_per_operator,
        )

        repair_by_operator = defaultdict(list)
        for split, observation in rows:
            if split != "repair":
                continue
            operator, _ = parse_action_label(observation.action_label)
            repair_by_operator[operator].append(observation)

        dovod_candidates = {}
        frequency_models = {}
        for operator, observations in sorted(repair_by_operator.items()):
            if len(observations) < args.min_repair_samples:
                continue
            dovod_candidates[operator] = fit_operator_repair(
                operator,
                observations,
                max_features=args.max_features,
                max_context_width=args.context_width,
                edit_penalty=args.edit_penalty,
                false_allow_weight=args.false_allow_weight,
                false_block_weight=args.false_block_weight,
            )
            frequency_models[operator] = fit_frequency_one_edit(
                operator,
                observations,
                max_features=args.max_features,
                edit_penalty=args.edit_penalty,
                false_allow_weight=args.false_allow_weight,
                false_block_weight=args.false_block_weight,
            )

        dovod_deployed, dovod_gates = choose_operator_calibration_gates(
            rows,
            dovod_candidates,
            require_false_allow_nonincrease=True,
        )
        frequency_gates = {
            operator: _one_edit_gate(operator, model, rows)
            for operator, model in sorted(frequency_models.items())
        }

        base_metrics = {}
        dovod_metrics = _metrics(rows, dovod=dovod_deployed)
        frequency_metrics = _metrics(
            rows,
            heuristic_models=frequency_models,
            heuristic_gates=frequency_gates,
        )
        for split in ("repair", "calibration", "test"):
            obs = [o for s, o in rows if s == split]
            base_metrics[split] = decision_metrics(obs, [o.base_allow for o in obs])

        random_replicates = []
        random_gate_summaries = []
        for replicate in range(args.random_replicates):
            models = {}
            gates = {}
            for operator, observations in sorted(repair_by_operator.items()):
                if len(observations) < args.min_repair_samples:
                    continue
                model = random_one_edit(
                    operator,
                    observations,
                    seed_key=(
                        f"v5-random|{args.domain}|{args.algorithm}|{args.trace_budget}|"
                        f"{operator}|replicate={replicate}"
                    ),
                    max_features=args.max_features,
                )
                models[operator] = model
                gates[operator] = _one_edit_gate(operator, model, rows)
            metrics = _metrics(rows, heuristic_models=models, heuristic_gates=gates)
            random_replicates.append({
                "replicate": replicate,
                "test": metrics["test"],
            })
            random_gate_summaries.append({
                "replicate": replicate,
                "deployed_operator_count": sum(g.deployed for g in gates.values()),
            })

    random_test_risks = [float(r["test"]["risk"]) for r in random_replicates]
    random_mean = sum(random_test_risks) / len(random_test_risks) if random_test_risks else None
    test_n = int(base_metrics["test"]["n"])
    return {
        "schema": "dovod-q1-amlgym-v5-baseline-case-v1",
        "status": "ok",
        "domain": args.domain,
        "algorithm": args.algorithm,
        "trace_budget": args.trace_budget,
        "amlgym_version": package_version("amlgym"),
        "python": platform.python_version(),
        "learning_trajectories": len(trajectories),
        "learn_seconds": learn_seconds,
        "wall_seconds": time.perf_counter() - started,
        "protocol": protocol,
        "test": {
            "base": base_metrics["test"],
            "dovod": dovod_metrics["test"],
            "frequency_one_edit": frequency_metrics["test"],
            "random_one_edit_mean_risk": random_mean,
            "random_one_edit_risks": random_test_risks,
            "dovod_minus_frequency_risk": (
                None if test_n == 0 else float(dovod_metrics["test"]["risk"]) - float(frequency_metrics["test"]["risk"])
            ),
            "dovod_minus_random_mean_risk": (
                None if test_n == 0 or random_mean is None else float(dovod_metrics["test"]["risk"]) - random_mean
            ),
        },
        "frequency_models": {
            operator: {
                "mode": model.mode,
                "feature": model.feature,
                "repair_objective": model.objective,
                "calibration_gate": _gate_json({operator: frequency_gates[operator]})[operator],
            }
            for operator, model in sorted(frequency_models.items())
        },
        "dovod_gates": _gate_json(dovod_gates),
        "random_gate_summaries": random_gate_summaries,
        "random_replicates": args.random_replicates,
        "claim_boundary": (
            "V5 diagnostic on the confirmatory semantic-state protocol. DOVOD, a best one-feature one-edit contingency heuristic, "
            "and frozen random one-edit policies are fitted/constructed without test labels and use the same operator-local calibration gate. "
            "Random results are averaged over prespecified deterministic replicates; no best-of-random test selection is used. "
            "This is a post-freeze comparator study and does not replace or retroactively alter the frozen v4 confirmatory result."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True)
    p.add_argument("--algorithm", required=True)
    p.add_argument("--trace-budget", required=True, type=int)
    p.add_argument("--max-problems", type=int, default=2)
    p.add_argument("--max-states", type=int, default=12)
    p.add_argument("--pilot-states-per-problem", type=int, default=12)
    p.add_argument("--max-actions-per-operator", type=int, default=4)
    p.add_argument("--min-repair-samples", type=int, default=4)
    p.add_argument("--max-features", type=int, default=8)
    p.add_argument("--context-width", type=int, default=1)
    p.add_argument("--edit-penalty", type=float, default=0.25)
    p.add_argument("--false-allow-weight", type=float, default=1.0)
    p.add_argument("--false-block-weight", type=float, default=1.0)
    p.add_argument("--random-replicates", type=int, default=16)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    try:
        report = run_case(args)
    except Exception as exc:
        report = {
            "schema": "dovod-q1-amlgym-v5-baseline-case-v1",
            "status": "failed",
            "domain": args.domain,
            "algorithm": args.algorithm,
            "trace_budget": args.trace_budget,
            "failure_stage": "v5_baseline_case",
            "error": f"{type(exc).__name__}: {exc}",
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"frequency_models", "dovod_gates", "random_gate_summaries"}}, indent=2, sort_keys=True))
    if report["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
