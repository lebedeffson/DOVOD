from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DOMAINS = (
    "barman", "blocksworld", "childsnack", "depots", "elevators", "ferry",
    "floortile", "goldminer", "grippers", "matchingbw", "miconic", "nomystery",
    "npuzzle", "parking", "rovers", "satellite", "sokoban", "spanner", "tpp", "transport",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algorithm", required=True)
    ap.add_argument("--trace-budget", type=int, required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--case-timeout", type=int, default=900)
    ap.add_argument("--domains", nargs="*", default=list(DOMAINS))
    args = ap.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    statuses = {}
    for domain in args.domains:
        path = outdir / f"{domain}__{args.algorithm}__n{args.trace_budget}.json"
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("run_amlgym_simple_baseline_case.py")),
            "--domain", domain,
            "--algorithm", args.algorithm,
            "--trace-budget", str(args.trace_budget),
            "--output", str(path),
        ]
        try:
            completed = subprocess.run(cmd, timeout=args.case_timeout, check=False)
            status = "ok" if completed.returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            status = "timeout"
            payload = {
                "schema": "dovod-q1-amlgym-simple-baseline-case-v1",
                "status": "timeout",
                "domain": domain,
                "algorithm": args.algorithm,
                "trace_budget": args.trace_budget,
                "failure_stage": "post_confirmatory_case_timeout",
                "error": f"case exceeded {args.case_timeout} seconds",
            }
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not path.exists():
            status = "missing_output"
            payload = {
                "schema": "dovod-q1-amlgym-simple-baseline-case-v1",
                "status": "failed",
                "domain": domain,
                "algorithm": args.algorithm,
                "trace_budget": args.trace_budget,
                "failure_stage": "missing_case_output",
                "error": "runner returned without writing a case artifact",
            }
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        statuses[domain] = status

    summary = {
        "schema": "dovod-q1-amlgym-simple-baseline-shard-v1",
        "algorithm": args.algorithm,
        "trace_budget": args.trace_budget,
        "case_timeout_seconds": args.case_timeout,
        "case_count": len(args.domains),
        "statuses": statuses,
        "case_files": [f"{d}__{args.algorithm}__n{args.trace_budget}.json" for d in args.domains],
    }
    summary_path = outdir / f"shard__{args.algorithm}__n{args.trace_budget}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
