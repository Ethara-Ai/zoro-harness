#!/usr/bin/env python3
"""
verify_oracle_reproducibility.py — run the Oracle twice on the same task and
assert that the golden answers are byte-identical (excluding wall-clock timing).

Optionally diff against a pinned baseline golden.json (--baseline) to catch
silent drift from earlier oracle_version generations.

Exit codes:
    0 — the two runs produced identical goldens (and matched --baseline if given)
    1 — mismatch (drift, non-determinism, or baseline divergence)
    2 — usage / input error

Usage:
    python tools/verify_oracle_reproducibility.py \
        --task dataset/still_middle/<task_id>.json \
        --days 30 \
        [--baseline runs/oracle/<task_id>/golden.json]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_ORACLE = REPO_ROOT / "tools" / "run_oracle.py"

# runtime_seconds is intentionally excluded — it's wall clock, not a policy output.
_NON_DETERMINISTIC_KEYS = {"runtime_seconds"}

# Simulation-outcome keys — what a historical baseline must still agree on
# even after the metadata schema (hashes, extra policy knobs) has expanded.
_OUTCOME_KEYS = (
    "task_id",
    "final_net_worth",
    "days_completed",
    "days_requested",
    "survival_fraction",
    "initial_funds",
)


def _strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in _NON_DETERMINISTIC_KEYS}


def _outcome(d: dict) -> dict:
    return {k: d.get(k) for k in _OUTCOME_KEYS}


def _run_oracle(task: Path, out_dir: Path, days: int) -> dict:
    result = subprocess.run(
        [
            sys.executable,
            str(RUN_ORACLE),
            "--task", str(task),
            "--days", str(days),
            "--out", str(out_dir),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(
            f"run_oracle.py failed (exit {result.returncode}) for out_dir={out_dir}"
        )
    return json.loads((out_dir / "golden.json").read_text())


def _diff_report(label_a: str, a: dict, label_b: str, b: dict) -> str:
    keys = sorted(set(a) | set(b))
    lines = []
    for k in keys:
        va, vb = a.get(k, "<missing>"), b.get(k, "<missing>")
        if va != vb:
            lines.append(f"  {k}:\n    {label_a}: {va!r}\n    {label_b}: {vb!r}")
    return "\n".join(lines) if lines else "  (identical)"


def main() -> None:
    ap = argparse.ArgumentParser(description="Determinism / regression check for the Oracle")
    ap.add_argument("--task", required=True, help="Path to a generated task json")
    ap.add_argument("--days", type=int, required=True, help="Simulation horizon")
    ap.add_argument("--baseline", default=None,
                    help="Optional pinned golden.json to diff against")
    args = ap.parse_args()

    task_path = Path(args.task)
    if not task_path.is_file():
        print(f"ERROR: task file not found: {task_path}", file=sys.stderr)
        sys.exit(2)

    tmp = Path(tempfile.mkdtemp(prefix="oracle_verify_"))
    try:
        out_a = tmp / "run_a"
        out_b = tmp / "run_b"
        golden_a = _strip(_run_oracle(task_path, out_a, args.days))
        golden_b = _strip(_run_oracle(task_path, out_b, args.days))

        ok = True
        if golden_a != golden_b:
            print("FAIL: two consecutive runs produced different goldens:")
            print(_diff_report("run_a", golden_a, "run_b", golden_b))
            ok = False
        else:
            print(f"OK: determinism confirmed. final_net_worth={golden_a['final_net_worth']}")

        if args.baseline:
            baseline_path = Path(args.baseline)
            if not baseline_path.is_file():
                print(f"ERROR: baseline not found: {baseline_path}", file=sys.stderr)
                sys.exit(2)
            baseline_outcome = _outcome(json.loads(baseline_path.read_text()))
            run_outcome = _outcome(golden_a)
            if baseline_outcome != run_outcome:
                print("FAIL: simulation outcome diverged from baseline:")
                print(_diff_report("baseline", baseline_outcome, "run_a", run_outcome))
                ok = False
            else:
                print(f"OK: outcome matches baseline {baseline_path}")

        sys.exit(0 if ok else 1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
