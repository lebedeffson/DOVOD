from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "amlgym_q1_contract.json"
CASE_RUNNER = ROOT / "benchmarks" / "run_amlgym_confirmatory_case.py"


def _failed_case(domain, algorithm, budget, error, *, status="failed", tail=""):
    return {
        "schema": "dovod-q1-amlgym-confirmatory-case-v1",
        "status": status,
        "domain": domain,
        "algorithm": algorithm,
        "trace_budget": int(budget),
        "failure_stage": "matrix_runner",
        "error": str(error),
        "process_tail": tail[-4000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default=str(CONTRACT))
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "amlgym-confirmatory"))
    parser.add_argument("--summary", default=str(ROOT / "results" / "paper_a_amlgym_confirmatory_shard.json"))
    parser.add_argument("--timeout-per-case", type=int, default=900)
    parser.add_argument("--domains", nargs="*")
    parser.add_argument("--algorithms", nargs="*")
    parser.add_argument("--trace-budgets", nargs="*", type=int)
    parser.add_argument("--fail-on-case-failure", action="store_true")
    args = parser.parse_args()

    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    if contract.get("schema") != "dovod-q1-amlgym-contract-v4":
        raise SystemExit(f"unexpected contract schema: {contract.get('schema')}")

    domains = args.domains or contract["domains"]
    algorithms = args.algorithms or contract["learner_families"]
    budgets = args.trace_budgets or contract["trace_budgets"]
    sampling = contract["predictive_sampling"]
    repair = contract["dovod_repair"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    started = time.perf_counter()

    for domain in domains:
        for algorithm in algorithms:
            for budget in budgets:
                case_path = output_dir / f"{domain}__{algorithm.lower()}__n{budget}.json"
                cmd = [
                    sys.executable,
                    str(CASE_RUNNER),
                    "--domain", domain,
                    "--algorithm", algorithm,
                    "--trace-budget", str(budget),
                    "--max-problems", str(sampling["problems_per_domain"]),
                    "--max-states", str(sampling["confirmatory_states_per_problem"]),
                    "--pilot-states-per-problem", str(sampling["pilot_states_per_problem_reserved"]),
                    "--max-actions-per-operator", str(sampling["ground_actions_per_operator_per_state"]),
                    "--min-repair-samples", str(repair["minimum_repair_samples_per_operator"]),
                    "--max-features", str(repair["max_action_local_features"]),
                    "--context-width", str(repair["max_context_width"]),
                    "--edit-penalty", str(repair["edit_penalty"]),
                    "--false-allow-weight", str(repair["false_allow_weight"]),
                    "--false-block-weight", str(repair["false_block_weight"]),
                    "--output", str(case_path),
                ]
                t0 = time.perf_counter()
                try:
                    proc = subprocess.run(
                        cmd,
                        cwd=str(ROOT),
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        timeout=args.timeout_per_case,
                        check=False,
                    )
                    if case_path.exists():
                        try:
                            case = json.loads(case_path.read_text(encoding="utf-8"))
                        except Exception as exc:
                            case = _failed_case(
                                domain, algorithm, budget,
                                f"invalid result JSON: {type(exc).__name__}: {exc}",
                                tail=proc.stdout,
                            )
                    else:
                        case = _failed_case(
                            domain, algorithm, budget,
                            "case runner produced no result file",
                            tail=proc.stdout,
                        )
                    case["process_returncode"] = proc.returncode
                    case["process_seconds"] = time.perf_counter() - t0
                    case["process_tail"] = proc.stdout[-4000:]
                    if proc.returncode != 0 and case.get("status") == "ok":
                        case = _failed_case(
                            domain, algorithm, budget,
                            "nonzero case-runner exit despite status=ok",
                            tail=proc.stdout,
                        )
                        case["process_returncode"] = proc.returncode
                        case["process_seconds"] = time.perf_counter() - t0
                except subprocess.TimeoutExpired as exc:
                    tail = exc.stdout if isinstance(exc.stdout, str) else ""
                    case = _failed_case(
                        domain, algorithm, budget,
                        f"timeout after {args.timeout_per_case}s",
                        status="timeout",
                        tail=tail,
                    )
                    case["process_seconds"] = time.perf_counter() - t0
                    case_path.write_text(json.dumps(case, indent=2, sort_keys=True) + "\n", encoding="utf-8")

                cases.append(case)
                print(f"[{case['status']}] {domain} / {algorithm} / n={budget}", flush=True)

    ok = [case for case in cases if case.get("status") == "ok"]
    protocol_violations = []
    for case in ok:
        protocol = case.get("protocol") or {}
        if protocol.get("stage") != "confirmatory":
            protocol_violations.append([case["domain"], case["algorithm"], case["trace_budget"], "wrong_stage"])
        if protocol.get("pilot_overlap_count") != 0:
            protocol_violations.append([case["domain"], case["algorithm"], case["trace_budget"], "pilot_overlap"])
        if protocol.get("split_unit") != "semantic_state_fingerprint":
            protocol_violations.append([case["domain"], case["algorithm"], case["trace_budget"], "wrong_split_unit"])

    summary = {
        "schema": "dovod-q1-amlgym-confirmatory-shard-v1",
        "contract_schema": contract["schema"],
        "cases": cases,
        "case_count": len(cases),
        "ok_count": len(ok),
        "failed_or_timeout_count": len(cases) - len(ok),
        "protocol_violations": protocol_violations,
        "wall_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "This shard is an execution artifact. Scientific performance is intentionally not a pass/fail condition. "
            "Failures, timeouts, and protocol violations are retained for the final confirmatory merge."
        ),
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2, sort_keys=True))

    if protocol_violations:
        raise SystemExit(4)
    if args.fail_on_case_failure and len(ok) != len(cases):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
