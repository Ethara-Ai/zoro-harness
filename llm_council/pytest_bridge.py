"""pytest_bridge.py — pytest wiring for the RetailBench evaluation layer.

Loads a trajectory via llm_council.snapshot.load_snapshot() and exposes it as
a session-scoped `snapshot` fixture. Registers the @pytest.mark.check(...)
marker and writes test_results.json (integer verdicts + name/verdict/weight/
knockout/category/kind/reason per test) after the session.

Per-task conftest.py should contain a single line:
    from llm_council.pytest_bridge import *  # noqa: F401,F403
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from .snapshot import Snapshot, load_snapshot

__all__ = [
    "snapshot",
    "pytest_addoption",
    "pytest_configure",
    "pytest_runtest_makereport",
    "pytest_sessionfinish",
]

_TEST_RECORDS: list[dict] = []


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--task-id",
        action="store",
        default="unknown",
        help="Task ID (echoed into test_results.json).",
    )
    parser.addoption(
        "--traj-path",
        action="store",
        default=None,
        help="Path to trajectory directory or tool_calls.jsonl (required for snapshot fixture).",
    )
    parser.addoption(
        "--dataset-path",
        action="store",
        default=None,
        help="Optional path to dataset.json (reserved; currently unused by snapshot loader).",
    )
    parser.addoption(
        "--days-requested",
        action="store",
        default=None,
        help="Horizon in days (int). Reserved for scoring metadata.",
    )
    parser.addoption(
        "--json-out",
        action="store",
        default=None,
        help="Path to write test_results.json. If omitted, no results file is written.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "check(id, weight, knockout, category, kind): RetailBench evaluation marker.",
    )


@pytest.fixture(scope="session")
def snapshot(request: pytest.FixtureRequest) -> Snapshot:
    traj = request.config.getoption("--traj-path")
    if not traj:
        pytest.skip("no --traj-path supplied; snapshot-dependent tests skipped")
    task_id = request.config.getoption("--task-id") or "unknown"
    return load_snapshot(traj, task_id=task_id)


def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> None:
    if call.when != "call":
        return
    marker = item.get_closest_marker("check")
    if marker is None:
        return
    kwargs = marker.kwargs
    check_id = kwargs.get("id") or item.name
    weight = float(kwargs.get("weight", 0.0))
    knockout = bool(kwargs.get("knockout", False))
    category = str(kwargs.get("category", ""))
    kind = str(kwargs.get("kind", ""))

    if call.excinfo is None:
        verdict: int | None = 1
        reason = ""
    elif call.excinfo.type.__name__ == "Skipped":
        verdict = None
        reason = str(call.excinfo.value)[:200]
    else:
        verdict = 0
        reason = repr(call.excinfo.value)[:200]

    _TEST_RECORDS.append(
        {
            "name": check_id,
            "verdict": verdict,
            "weight": weight,
            "knockout": knockout,
            "category": category,
            "kind": kind,
            "reason": reason,
        }
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    out_path = session.config.getoption("--json-out")
    if not out_path:
        return
    task_id = session.config.getoption("--task-id") or "unknown"
    traj = session.config.getoption("--traj-path") or ""

    n_passed = sum(1 for r in _TEST_RECORDS if r["verdict"] == 1)
    n_failed = sum(1 for r in _TEST_RECORDS if r["verdict"] == 0)
    n_skipped = sum(1 for r in _TEST_RECORDS if r["verdict"] is None)
    knockouts_failed = [r["name"] for r in _TEST_RECORDS if r["knockout"] and r["verdict"] == 0]

    payload = {
        "task_id": task_id,
        "traj_path": str(traj),
        "n_tests": len(_TEST_RECORDS),
        "n_passed": n_passed,
        "n_failed": n_failed,
        "n_skipped": n_skipped,
        "knockouts_failed": knockouts_failed,
        "tests": _TEST_RECORDS,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
