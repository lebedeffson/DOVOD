from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
from importlib.metadata import version as package_version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_a.amlgym_bridge import (
    DecisionObservation,
    decision_metrics,
    fit_operator_repair,
    parse_action_label,
    predict_operator_repair,
    stable_bucket,
)
from benchmarks.run_amlgym_case import _sample_ground_actions, _sample_states


def collect_raw_observations(
    *, domain: str, reference_domain_path: str, learned_domain_path: str,
    max_problems: int, max_states_per_problem: int, max_actions_per_operator: int,
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
                ref_env, domain=domain, problem_name=problem_name, state_index=state_index,
                per_operator=max_actions_per_operator,
            )
            for action in actions:
                operator, _ = parse_action_label(action)
                obs = DecisionObservation(
                    state_literals=state,
                    action_label=action,
                    base_allow=int(action in base.get(operator, set())),
                    truth_allow=int(action in truth.get(operator, set())),
                )
                key = f"pair|{domain}|{problem_name}|{state_index}|{action}"
                rows.append((key, obs))
    return rows


def assign_split(raw_rows, salt: int):
    rows = []
    for key, obs in raw_rows:
        bucket = stable_bucket(f"split-stability|{salt}|{key}", 1000)
        split = "repair" if bucket < 500 else ("calibration" if bucket < 750 else "test")
        rows.append((split, obs))
    return rows


def split_metrics(rows, models):
    out = {}
    for split in ("repair", "calibration", "test"):
        obs = [o for s, o in rows if s == split]
        base = [o.base_allow for o in obs]
        repaired = []
        for o in obs:
            op, _ = parse_action_label(o.action_label)
            model = models.get(op)
            repaired.append(o.base_allow if model is None else predict_operator_repair(model, o))
        out[split] = {"base": decision_metrics(obs, base), "dovod": decision_metrics(obs, repaired)}
    return out


def fit_and_gate(rows, args):
    repair_obs = [o for split, o in rows if split == "repair"]
    by_op = {}
    for o in repair_obs:
        op, _ = parse_action_label(o.action_label)
        by_op.setdefault(op, []).append(o)
    candidates = {}
    for op, obs in sorted(by_op.items()):
        if len(obs) < args.min_repair_samples:
            continue
        candidates[op] = fit_operator_repair(
            op, obs, max_features=args.max_features,
            max_context_width=args.context_width, edit_penalty=args.edit_penalty,
        )

    deployed = {}
    gate = {}
    for op, model in sorted(candidates.items()):
        cal = [
            o for split, o in rows
            if split == "calibration" and parse_action_label(o.action_label)[0] == op
        ]
        if not cal:
            gate[op] = {"deployed": False, "reason": "no_calibration_samples"}
            continue
        base_m = decision_metrics(cal, [o.base_allow for o in cal])
        pred = [predict_operator_repair(model, o) for o in cal]
        rep_m = decision_metrics(cal, pred)
        strict = rep_m["risk"] < base_m["risk"] - 1e-15
        empty = (not model.fit.selected_edits and rep_m["risk"] == base_m["risk"])
        use = bool(strict or empty)
        if use:
            deployed[op] = model
        gate[op] = {
            "deployed": use,
            "base_risk": base_m["risk"],
            "candidate_risk": rep_m["risk"],
            "selected_edit_count": len(model.fit.selected_edits),
            "reason": "strict_calibration_improvement" if strict else ("empty_noop" if empty else "no_strict_calibration_gain"),
        }
    return candidates, deployed, gate


def run(args):
    from amlgym.algorithms import get_algorithm
    from amlgym.benchmarks import get_domain_path, get_trajectories_path

    started = time.perf_counter()
    ref_domain = get_domain_path(args.domain)
    trajectories = get_trajectories_path(args.domain, kind="learning")[: args.trace_budget]
    if len(trajectories) < args.trace_budget:
        raise RuntimeError("insufficient learning trajectories")
    kwargs = {"noise": 0.0} if args.algorithm.lower() == "nolam" else {}
    learner = get_algorithm(args.algorithm, **kwargs)
    learned_text = learner.learn(ref_domain, trajectories)

    with tempfile.TemporaryDirectory(prefix="dovod-amlgym-stability-") as td:
        learned_path = Path(td) / "learned.pddl"
        learned_path.write_text(str(learned_text), encoding="utf-8")
        raw = collect_raw_observations(
            domain=args.domain, reference_domain_path=ref_domain, learned_domain_path=str(learned_path),
            max_problems=args.max_problems, max_states_per_problem=args.max_states,
            max_actions_per_operator=args.max_actions_per_operator,
        )
        rows = []
        for salt in range(args.splits):
            assigned = assign_split(raw, salt)
            candidates, deployed, gate = fit_and_gate(assigned, args)
            metrics = split_metrics(assigned, deployed)
            test = metrics["test"]
            base_bal = test["base"]["class_balanced_risk"]
            rep_bal = test["dovod"]["class_balanced_risk"]
            rows.append({
                "split_salt": salt,
                "test": test,
                "raw_risk_reduction": test["base"]["risk"] - test["dovod"]["risk"],
                "class_balanced_risk_reduction": None if base_bal is None or rep_bal is None else base_bal - rep_bal,
                "candidate_operator_count": len(candidates),
                "deployed_nonempty_operator_count": sum(bool(m.fit.selected_edits) for m in deployed.values()),
                "deployed_edit_count": sum(len(m.fit.selected_edits) for m in deployed.values()),
                "gate": gate,
            })

    deltas = [r["raw_risk_reduction"] for r in rows]
    balanced = [r["class_balanced_risk_reduction"] for r in rows if r["class_balanced_risk_reduction"] is not None]
    return {
        "schema": "dovod-q1-amlgym-split-stability-v1",
        "domain": args.domain,
        "algorithm": args.algorithm,
        "trace_budget": args.trace_budget,
        "amlgym_version": package_version("amlgym"),
        "python": platform.python_version(),
        "observation_pool_size": len(raw),
        "split_count": len(rows),
        "rows": rows,
        "summary": {
            "mean_raw_risk_reduction": float(sum(deltas) / len(deltas)),
            "median_raw_risk_reduction": float(sorted(deltas)[len(deltas)//2]),
            "positive_raw_reduction_splits": int(sum(x > 0 for x in deltas)),
            "negative_raw_reduction_splits": int(sum(x < 0 for x in deltas)),
            "tied_raw_reduction_splits": int(sum(x == 0 for x in deltas)),
            "mean_class_balanced_risk_reduction": None if not balanced else float(sum(balanced) / len(balanced)),
            "positive_class_balanced_splits": int(sum(x > 0 for x in balanced)),
            "negative_class_balanced_splits": int(sum(x < 0 for x in balanced)),
        },
        "wall_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "This is a deterministic split-sensitivity diagnostic on one learned model and one finite AMLGym state/action pool. "
            "Repeated hash splits are not independent domains or repeated learner training seeds and are not used as pseudo-replicates for population significance."
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True)
    p.add_argument("--algorithm", required=True)
    p.add_argument("--trace-budget", required=True, type=int)
    p.add_argument("--splits", type=int, default=20)
    p.add_argument("--max-problems", type=int, default=2)
    p.add_argument("--max-states", type=int, default=12)
    p.add_argument("--max-actions-per-operator", type=int, default=4)
    p.add_argument("--min-repair-samples", type=int, default=4)
    p.add_argument("--max-features", type=int, default=8)
    p.add_argument("--context-width", type=int, default=1)
    p.add_argument("--edit-penalty", type=float, default=0.25)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    report = run(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
