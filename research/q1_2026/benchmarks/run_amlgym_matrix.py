from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "amlgym_q1_contract.json"
CASE_RUNNER = ROOT / "benchmarks" / "run_amlgym_case.py"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--contract", default=str(CONTRACT))
    p.add_argument("--output-dir", default=str(ROOT / "artifacts" / "amlgym"))
    p.add_argument("--summary", default=str(ROOT / "results" / "paper_a_amlgym_matrix.json"))
    p.add_argument("--timeout-per-case", type=int, default=900)
    p.add_argument("--domains", nargs="*")
    p.add_argument("--algorithms", nargs="*")
    p.add_argument("--trace-budgets", nargs="*", type=int)
    p.add_argument("--fail-on-case-failure", action="store_true")
    args = p.parse_args()

    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    domains = args.domains or contract["domains"]
    algorithms = args.algorithms or contract["learner_families"]
    budgets = args.trace_budgets or contract["trace_budgets"]
    sampling = contract["predictive_sampling"]
    repair = contract["dovod_repair"]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    started = time.perf_counter()

    for domain in domains:
        for algorithm in algorithms:
            for budget in budgets:
                case_path = out_dir / f"{domain}__{algorithm.lower()}__n{budget}.json"
                cmd = [
                    sys.executable,
                    str(CASE_RUNNER),
                    "--domain", domain,
                    "--algorithm", algorithm,
                    "--trace-budget", str(budget),
                    "--max-problems", str(sampling["problems_per_domain"]),
                    "--max-states", str(sampling["states_per_problem"]),
                    "--max-actions-per-operator", str(sampling["ground_actions_per_operator_per_state"]),
                    "--max-features", str(repair["max_action_local_features"]),
                    "--context-width", str(repair["max_context_width"]),
                    "--edit-penalty", str(repair["edit_penalty"]),
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
                        case = json.loads(case_path.read_text(encoding="utf-8"))
                    else:
                        case = {
                            "schema": "dovod-q1-amlgym-case-v2",
                            "status": "failed",
                            "domain": domain,
                            "algorithm": algorithm,
                            "trace_budget": budget,
                            "error": "case runner produced no result file",
                        }
                    case["process_returncode"] = proc.returncode
                    case["process_seconds"] = time.perf_counter() - t0
                    case["process_tail"] = proc.stdout[-4000:]
                except subprocess.TimeoutExpired as exc:
                    case = {
                        "schema": "dovod-q1-amlgym-case-v2",
                        "status": "timeout",
                        "domain": domain,
                        "algorithm": algorithm,
                        "trace_budget": budget,
                        "process_seconds": time.perf_counter() - t0,
                        "error": f"timeout after {args.timeout_per_case}s",
                        "process_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
                    }
                    case_path.write_text(json.dumps(case, indent=2, sort_keys=True), encoding="utf-8")
                cases.append(case)
                print(f"[{case['status']}] {domain} / {algorithm} / n={budget}", flush=True)

    ok = [c for c in cases if c["status"] == "ok"]
    with_decision = [c for c in ok if c.get("decision") is not None]
    paired_test_deltas = []
    for c in with_decision:
        test = c["decision"]["test"]
        if test["base"]["n"]:
            paired_test_deltas.append(test["base"]["risk"] - test["dovod"]["risk"])

    summary = {
        "schema": "dovod-q1-amlgym-matrix-v2",
        "contract": contract,
        "cases": cases,
        "case_count": len(cases),
        "ok_count": len(ok),
        "decision_bridge_count": len(with_decision),
        "failed_or_timeout_count": len(cases) - len(ok),
        "mean_test_risk_reduction": (
            sum(paired_test_deltas) / len(paired_test_deltas) if paired_test_deltas else None
        ),
        "improved_test_cases": sum(d > 0 for d in paired_test_deltas),
        "worsened_test_cases": sum(d < 0 for d in paired_test_deltas),
        "tied_test_cases": sum(d == 0 for d in paired_test_deltas),
        "wall_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "This matrix retains failures/timeouts and is descriptive until all prespecified cells are reviewed. "
            "Domain, not individual state/action pair, is the intended broad-claim statistical unit."
        ),
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2, sort_keys=True))

    if args.fail_on_case_failure and len(ok) != len(cases):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
