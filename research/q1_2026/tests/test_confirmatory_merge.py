from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "merge_amlgym_confirmatory_results.py"
    spec = importlib.util.spec_from_file_location("confirmatory_merge", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_cases_ignores_embedded_shard_summary_duplicates(tmp_path: Path) -> None:
    case = {
        "schema": "dovod-q1-amlgym-confirmatory-case-v1",
        "domain": "blocksworld",
        "algorithm": "SAM",
        "trace_budget": 3,
        "status": "timeout",
    }
    (tmp_path / "case.json").write_text(json.dumps(case), encoding="utf-8")
    (tmp_path / "summary.json").write_text(
        json.dumps({"schema": "dovod-q1-amlgym-confirmatory-shard-v1", "cases": [case]}),
        encoding="utf-8",
    )

    rows = _module().load_cases(tmp_path)
    assert len(rows) == 1
    assert rows[0]["domain"] == "blocksworld"
    assert rows[0]["status"] == "timeout"
