#!/usr/bin/env python3
"""
generate_all.py — Post-oracle chained generator for a harbor-format
RetailBench task.

Runs, in order:
    1. script/generate_truth.py   <task_dir>   (writes TRUTH.md)
    2. script/generate_rubrics.py <task_dir>   (writes tests/rubrics.json)
    3. script/generate_tests.py   <task_dir>   (writes tests/test_outputs.py
                                                + tests/conftest.py)

Stages 2 and 3 hard-require TRUTH.md on disk, so the chain aborts the moment
any stage fails.

Prerequisite: claude_bridge running on 127.0.0.1:8738. See
`claude_bridge/bridge-setup.md` and `script/start_claude_bridge.sh`.

Usage:

    python script/generate_all.py <task_dir>
    python script/generate_all.py <task_dir> --overwrite
    python script/generate_all.py <task_dir> --overwrite --model sonnet
    python script/generate_all.py <task_dir> --dry-run     # forward to each stage
    python script/generate_all.py <task_dir> --skip-truth  # rerun only rubrics+tests
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
STAGES: list[tuple[str, str]] = [
    ("truth",   "generate_truth.py"),
    ("rubrics", "generate_rubrics.py"),
    ("tests",   "generate_tests.py"),
]


def _run_stage(stage: str, script_name: str, task_dir: Path, passthrough: list[str]) -> int:
    script_path = SCRIPT_DIR / script_name
    if not script_path.exists():
        print(f"[generate_all] {stage}: FAIL — script not found at {script_path}", file=sys.stderr)
        return 2
    cmd = [sys.executable, str(script_path), str(task_dir), *passthrough]
    print(f"[generate_all] ---- {stage} ----", file=sys.stderr)
    print(f"[generate_all] {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[generate_all] {stage}: FAIL (exit {result.returncode})", file=sys.stderr)
    else:
        print(f"[generate_all] {stage}: ok", file=sys.stderr)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run generate_truth.py, generate_rubrics.py, generate_tests.py "
                    "in order for a harbor-format task directory. Aborts on first failure.",
    )
    parser.add_argument("task_dir", type=Path, help="Harbor-format task directory")
    parser.add_argument("--overwrite", action="store_true",
                        help="Forward --overwrite to each stage")
    parser.add_argument("--model", type=str, default=None,
                        help="Forward --model <name> to each stage")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Forward --max-tokens <n> to each stage")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Forward --temperature <f> to each stage")
    parser.add_argument("--dry-run", action="store_true",
                        help="Forward --dry-run to each stage (assemble prompts, "
                             "no LLM calls, no writes)")
    parser.add_argument("--skip-truth", action="store_true",
                        help="Skip stage 1 (assumes TRUTH.md already on disk)")
    parser.add_argument("--skip-rubrics", action="store_true",
                        help="Skip stage 2")
    parser.add_argument("--skip-tests", action="store_true",
                        help="Skip stage 3")
    args = parser.parse_args()

    task_dir = args.task_dir.resolve()
    if not task_dir.is_dir():
        print(f"error: task_dir does not exist or is not a directory: {task_dir}", file=sys.stderr)
        return 1

    passthrough: list[str] = []
    if args.overwrite:
        passthrough.append("--overwrite")
    if args.model is not None:
        passthrough.extend(["--model", args.model])
    if args.max_tokens is not None:
        passthrough.extend(["--max-tokens", str(args.max_tokens)])
    if args.temperature is not None:
        passthrough.extend(["--temperature", str(args.temperature)])
    if args.dry_run:
        passthrough.append("--dry-run")

    skip_map = {
        "truth":   args.skip_truth,
        "rubrics": args.skip_rubrics,
        "tests":   args.skip_tests,
    }

    for stage, script_name in STAGES:
        if skip_map[stage]:
            print(f"[generate_all] ---- {stage}: skipped ----", file=sys.stderr)
            continue
        rc = _run_stage(stage, script_name, task_dir, passthrough)
        if rc != 0:
            print(f"[generate_all] aborted at stage '{stage}' (exit {rc}). "
                  f"Fix and rerun; --skip-truth / --skip-rubrics can resume mid-chain.",
                  file=sys.stderr)
            return rc

    print(f"[generate_all] all stages complete for {task_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
