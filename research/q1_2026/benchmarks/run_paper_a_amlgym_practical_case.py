from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_amlgym_confirmatory_case import collect_confirmatory_observations
from paper_a.amlgym_bridge import (
    decision_metrics,
    fit_operator_repair,
    parse_action_label,
    predict_operator_repair,
)
from paper_a.deployment import (
    calibration_gate_from_predictions,
    choose_operator_calibration_gates,
    gated_prediction,
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


def _flatten_preconditions(action) -> list[str]:
    out: list[str] = []
    for precondition in getattr(action, "preconditions", ()):
        stack = [precondition]
        while stack:
            expr = stack.pop()
            try:
                is_and = expr.is_and()
            except Exception:
                is_and = False
            if is_and:
                stack.extend(reversed(tuple(expr.args)))
            else:
                out.append(str(expr))
    return out


def _schema_summary(problem) -> dict:
    per_operator = {}
    for action in problem.actions:
        atoms = _flatten_preconditions(action)
        per_operator[action.name] = {
            "atomic_precondition_count": len(atoms),
            "preconditions": atoms,
        }
    return {
        "operators": len(per_operator),
        "atomic_preconditions": sum(row["atomic_precondition_count"] for row in per_operator.values()),
        "per_operator": per_operator,
    }


def _serialize_edit(edit) -> dict:
    return {
        "kind": edit.kind,
        "context": [list(x) if isinstance(x, tuple) else x for x in edit.context],
        "prerequisite": edit.prerequisite,
        "weight": float(edit.weight),
    }


def _fit_threshold_policy(rows, threshold: float, min_samples: int) -> dict[str, dict]:
    """Fit a deliberately coarse non-contextual comparator on repair data only."""
    by_operator: dict[str, list] = defaultdict(list)
    for split, obs in rows:
        if split != "repair" or obs.base_allow != 0:
            continue
        operator, _ = parse_action_label(obs.action_label)
        by_operator[operator].append(obs)
    policies = {}
    for operator, observations in sorted(by_operator.items()):
        support = sum(o.truth_allow for o in observations) / len(observations)
        policies[operator] = {
            "override_blocks": len(observations) >= min_samples and support >= threshold,
            "blocked_samples": len(observations),
            "reference_allow_fraction": support,
        }
    return policies


def _threshold_predict(obs, policies: dict[str, dict]) -> int:
    operator, _ = parse_action_label(obs.action_label)
    policy = policies.get(operator)
    if obs.base_allow == 0 and policy and policy["override_blocks"]:
        return 1
    return int(obs.base_allow)


def _threshold_metrics(rows, policies, gates=None):
    out = {}
    for split in ("repair", "calibration", "test"):
        observations = [o for s, o in rows if s == split]
        predictions = []
        for obs in observations:
            operator, _ = parse_action_label(obs.action_label)
            use = policies.get(operator, {}).get("override_blocks", False)
            if gates is not None:
                gate = gates.get(operator)
                use = use and gate is not None and gate.deployed
            predictions.append(1 if obs.base_allow == 0 and use else int(obs.base_allow))
        out[split] = decision_metrics(observations, predictions)
    return out


def _gate_threshold_policies(rows, policies):
    gates = {}
    for operator, policy in sorted(policies.items()):
        if not policy["override_blocks"]:
            continue
        calibration = tuple(
            obs for split, obs in rows
            if split == "calibration" and parse_action_label(obs.action_label)[0] == operator
        )
        predictions = tuple(_threshold_predict(obs, policies) for obs in calibration)
        gates[operator] = calibration_gate_from_predictions(
            operator,
            calibration,
            predictions,
            selected_edit_count=1,
            require_false_allow_nonincrease=True,
        )
    return gates


def run(args) -> dict:
    from amlgym.algorithms import get_algorithm
    from amlgym.benchmarks import get_domain_path, get_trajectories_path, get_problems_path
    from amlgym.modeling.UPEnv import UPEnv

    _seed_everything(args.seed)
    os.environ.setdefault("PYTHONHASHSEED", "0")
    reference_domain = get_domain_path(args.domain)
    trajectories = get_trajectories_path(args.domain, kind="learning")[:args.trace_budget]
    if len(trajectories) < args.trace_budget:
        raise RuntimeError("Insufficient learning trajectories")
    learner = get_algorithm(args.algorithm, **({"noise": 0.0} if args.algorithm.lower() == "nolam" else {}))
    learned_text = learner.learn(reference_domain, trajectories)

    with tempfile.TemporaryDirectory(prefix="dovod-a-practical-") as td:
        learned_path = Path(td) / "learned.pddl"
        learned_path.write_text(str(learned_text), encoding="utf-8")
        problem_path = get_problems_path(args.domain, kind="predictive_power")[0]
        learned_env = UPEnv(str(learned_path), problem_path)
        reference_env = UPEnv(reference_domain, problem_path)
        learned_schema = _schema_summary(learned_env.problem)
        reference_schema = _schema_summary(reference_env.problem)
        rows, protocol = collect_confirmatory_observations(
            domain=args.domain,
            reference_domain_path=reference_domain,
            learned_domain_path=str(learned_path),
            max_problems=args.max_problems,
            states_per_problem=args.max_states,
            pilot_states_per_problem=args.pilot_states_per_problem,
            max_actions_per_operator=args.max_actions_per_operator,
        )

        repair_by_operator = defaultdict(list)
        for split, obs in rows:
            if split == "repair":
                operator, _ = parse_action_label(obs.action_label)
                repair_by_operator[operator].append(obs)
        candidates = {}
        for operator, observations in sorted(repair_by_operator.items()):
            if len(observations) < args.min_repair_samples:
                continue
            candidates[operator] = fit_operator_repair(
                operator,
                observations,
                max_features=args.max_features,
                max_context_width=args.context_width,
                edit_penalty=args.edit_penalty,
                false_allow_weight=1.0,
                false_block_weight=1.0,
            )
        deployed, gates = choose_operator_calibration_gates(rows, candidates, require_false_allow_nonincrease=True)

        threshold = _fit_threshold_policy(rows, args.threshold, args.threshold_min_samples)
        threshold_raw = _threshold_metrics(rows, threshold)
        threshold_gates = _gate_threshold_policies(rows, threshold)
        threshold_gated = _threshold_metrics(rows, threshold, threshold_gates)

        split_metrics = {}
        for split in ("repair", "calibration", "test"):
            observations = [o for s, o in rows if s == split]
            split_metrics[split] = {
                "base": decision_metrics(observations, [o.base_allow for o in observations]),
                "dovod": decision_metrics(observations, [gated_prediction(o, deployed) for o in observations]),
                "threshold_raw": threshold_raw[split],
                "threshold_gated": threshold_gated[split],
            }

        operator_rows = {}
        for operator in sorted(set(repair_by_operator) | set(candidates)):
            observations = repair_by_operator.get(operator, [])
            base_fa = sum(o.base_allow == 1 and o.truth_allow == 0 for o in observations)
            base_fb = sum(o.base_allow == 0 and o.truth_allow == 1 for o in observations)
            model = candidates.get(operator)
            gate = gates.get(operator)
            operator_rows[operator] = {
                "repair_samples": len(observations),
                "base_false_allows": base_fa,
                "base_false_blocks": base_fb,
                "flagged_by_counterexample": (base_fa + base_fb) > 0,
                "learned_atomic_preconditions": learned_schema["per_operator"].get(operator, {}).get("atomic_precondition_count", 0),
                "reference_atomic_preconditions": reference_schema["per_operator"].get(operator, {}).get("atomic_precondition_count", 0),
                "candidate_edit_count": 0 if model is None else len(model.fit.selected_edits),
                "candidate_edits": [] if model is None else [_serialize_edit(e) for e in model.fit.selected_edits],
                "deployed": bool(gate and gate.deployed),
                "gate_reason": None if gate is None else gate.reason,
                "threshold_policy": threshold.get(operator),
                "threshold_gate": None if operator not in threshold_gates else {
                    "deployed": threshold_gates[operator].deployed,
                    "reason": threshold_gates[operator].reason,
                },
            }

        exemplar = None
        for split in ("test", "calibration", "repair"):
            for obs in sorted((o for s, o in rows if s == split), key=lambda x: x.action_label):
                dovod = gated_prediction(obs, deployed)
                if obs.base_allow == 0 and obs.truth_allow == 1 and dovod == 1:
                    operator, _ = parse_action_label(obs.action_label)
                    model = deployed.get(operator)
                    exemplar = {
                        "split": split,
                        "operator": operator,
                        "action": obs.action_label,
                        "upstream_decision": "BLOCK",
                        "reference_decision": "ALLOW",
                        "dovod_decision": "ALLOW",
                        "state_literals": list(obs.state_literals),
                        "local_features": [] if model is None else list(model.feature_names),
                        "selected_edits": [] if model is None else [_serialize_edit(e) for e in model.fit.selected_edits],
                    }
                    break
            if exemplar is not None:
                break

    test_base = split_metrics["test"]["base"]
    test_dovod = split_metrics["test"]["dovod"]
    flagged_ops = sum(row["flagged_by_counterexample"] for row in operator_rows.values())
    deployed_ops = sum(row["deployed"] for row in operator_rows.values())
    unresolved_ops = sum(row["flagged_by_counterexample"] and not row["deployed"] for row in operator_rows.values())
    return {
        "schema": "dovod-paper-a-amlgym-practical-case-v1",
        "status": "ok",
        "domain": args.domain,
        "algorithm": args.algorithm,
        "trace_budget": args.trace_budget,
        "seed": args.seed,
        "protocol": protocol,
        "learned_schema": learned_schema,
        "reference_schema": reference_schema,
        "audit": {
            "operator_decision_surfaces": len(operator_rows),
            "operators_flagged_by_observed_applicability_counterexample": flagged_ops,
            "operators_with_deployed_contextual_repair": deployed_ops,
            "flagged_operators_left_unresolved_after_calibration_gate": unresolved_ops,
            "learned_atomic_precondition_context_size": learned_schema["atomic_preconditions"],
            "boundary": "Flagged/deployed/unresolved counts are operator-level applicability audit units. The lifted precondition count is schema context only.",
        },
        "metrics": split_metrics,
        "operator_details": operator_rows,
        "representative_corrected_false_block": exemplar,
        "test_risk_reduction": float(test_base["risk"]) - float(test_dovod["risk"]),
        "claim_boundary": (
            "This is a post-confirmatory explanatory case study on prespecified AMLGym domains. "
            "It is not used to re-select the frozen v4 method or to change the primary statistical result. "
            "The 0.8 comparator is a deliberately coarse repair-split-only global override, not a model of human behavior."
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
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--threshold-min-samples", type=int, default=4)
    args = ap.parse_args()
    try:
        report = run(args)
    except Exception as exc:
        report = {
            "schema": "dovod-paper-a-amlgym-practical-case-v1",
            "status": "failed",
            "domain": args.domain,
            "algorithm": args.algorithm,
            "trace_budget": args.trace_budget,
            "seed": args.seed,
            "error": f"{type(exc).__name__}: {exc}",
            "claim_boundary": "Upstream/tool failure retained as an outcome; no scientific metric is imputed.",
        }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
