from __future__ import annotations

import json
import platform
import tempfile
import time
from importlib.metadata import version as package_version
from pathlib import Path

from amlgym.algorithms import get_algorithm
from amlgym.benchmarks import get_domain_path, get_trajectories_path
from amlgym.metrics import syntactic_precision, syntactic_recall

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "amlgym_upstream_smoke.json"


def run_one(domain: str, algorithm: str, trace_budget: int) -> dict:
    ref_domain = get_domain_path(domain)
    trajectories = get_trajectories_path(domain, kind="learning")[:trace_budget]
    if len(trajectories) < trace_budget:
        raise RuntimeError(f"{domain}: requested {trace_budget}, got {len(trajectories)}")
    kwargs = {"noise": 0.0} if algorithm.lower() == "nolam" else {}
    learner = get_algorithm(algorithm, **kwargs)
    t0 = time.perf_counter()
    learned_text = learner.learn(ref_domain, trajectories)
    elapsed = time.perf_counter() - t0
    with tempfile.TemporaryDirectory(prefix="amlgym-smoke-") as td:
        learned_path = Path(td) / f"{domain}_{algorithm}.pddl"
        learned_path.write_text(str(learned_text), encoding="utf-8")
        precision = syntactic_precision(str(learned_path), ref_domain)
        recall = syntactic_recall(str(learned_path), ref_domain)
    return {
        "domain": domain,
        "algorithm": algorithm,
        "trace_budget": trace_budget,
        "learn_seconds": elapsed,
        "syntactic_precision": precision,
        "syntactic_recall": recall,
    }


def main() -> None:
    cases = [
        ("blocksworld", "SAM", 3),
        ("blocksworld", "OffLAM", 3),
        ("blocksworld", "NOLAM", 3),
        ("blocksworld", "ROSAME", 3),
    ]
    results = []
    for domain, algorithm, budget in cases:
        try:
            row = run_one(domain, algorithm, budget)
            row["status"] = "ok"
        except Exception as exc:
            row = {
                "domain": domain,
                "algorithm": algorithm,
                "trace_budget": budget,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(row)
    report = {
        "schema": "dovod-q1-amlgym-upstream-smoke-v1",
        "amlgym_version": package_version("amlgym"),
        "python": platform.python_version(),
        "cases": results,
        "ok_count": sum(r["status"] == "ok" for r in results),
        "claim_boundary": "Upstream compatibility/syntactic smoke only; no DOVOD repair result is claimed here.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["ok_count"] == 0:
        raise SystemExit("all AMLGym upstream smoke cases failed")


if __name__ == "__main__":
    main()
