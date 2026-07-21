#!/usr/bin/env python3
"""Tier-1 TOCTOU stress test for _pick_next_run_dir.

Spawns N concurrent workers via multiprocessing (spawn context, matching
production subprocess topology) that all call the REAL _pick_next_run_dir
source extracted verbatim from run_env.py against the same model_dir.

The function source is pulled out with `ast` so we test the exact shipped
bytes without triggering run_env.py's heavy top-level imports (openai,
retail_environment, etc.).

Asserts:
  * Every worker returns a distinct run_N path (no collisions).
  * Every returned path is an existing directory.
  * The returned set is exactly {run_1, run_2, ..., run_N} (gapless).

Exits 0 on pass, non-zero on any assertion failure.
"""

from __future__ import annotations

import ast

import multiprocessing as mp
import os
import sys
import tempfile
import textwrap
import time
import traceback
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
RUN_ENV_PATH = HARNESS_DIR / "run_env.py"
TARGET_FUNC = "_pick_next_run_dir"


def _extract_function_source(path: Path, name: str) -> str:
    """Extract a single top-level function's source verbatim from a .py file."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return textwrap.dedent(ast.get_source_segment(src, node) or "")
    raise RuntimeError(f"function {name!r} not found in {path}")


PICK_NEXT_RUN_DIR_SRC = _extract_function_source(RUN_ENV_PATH, TARGET_FUNC)


def _worker(payload: tuple[str, str]) -> str:
    """Compile the extracted function source in-worker and call it."""
    model_dir_str, func_src = payload
    from pathlib import Path as _Path  # noqa: PLC0415

    ns: dict = {"Path": _Path}
    exec(compile(func_src, "<pick_next_run_dir>", "exec"), ns)
    pick = ns[TARGET_FUNC]
    return str(pick(_Path(model_dir_str)))


def _run_stress(num_workers: int) -> int:
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="toctou_stress_") as tmp:
        model_dir = Path(tmp) / "task_id_stress" / "gpt-5.6"
        model_dir_str = str(model_dir)

        t0 = time.monotonic()
        with ctx.Pool(processes=num_workers) as pool:
            try:
                results = pool.map(_worker, [(model_dir_str, PICK_NEXT_RUN_DIR_SRC)] * num_workers)
            except Exception:
                traceback.print_exc()
                return 2
        elapsed = time.monotonic() - t0

        paths = [Path(r) for r in results]
        unique = set(results)

        print(f"[stress] workers={num_workers} elapsed={elapsed:.2f}s")
        print(f"[stress] returned={len(results)} unique={len(unique)}")

        failures: list[str] = []

        if len(unique) != num_workers:
            dupes = [r for r in results if results.count(r) > 1]
            failures.append(
                f"expected {num_workers} unique paths, got {len(unique)}; dupes={sorted(set(dupes))}"
            )

        missing_dirs = [p for p in paths if not p.is_dir()]
        if missing_dirs:
            failures.append(f"{len(missing_dirs)} returned paths are not directories: {missing_dirs[:5]}")

        bad_names = [p for p in paths if not p.name.startswith("run_")]
        if bad_names:
            failures.append(f"{len(bad_names)} returned paths have wrong prefix: {bad_names[:5]}")

        try:
            nums = sorted(int(p.name[len("run_"):]) for p in paths)
            expected = list(range(1, num_workers + 1))
            if nums != expected:
                gaps = [n for n in expected if n not in nums]
                extras = [n for n in nums if n not in expected]
                failures.append(
                    f"run_N numbering not gapless 1..{num_workers}; missing={gaps[:5]} extras={extras[:5]}"
                )
        except ValueError as e:
            failures.append(f"non-integer suffix in run_N names: {e}")

        on_disk = sorted(p.name for p in model_dir.iterdir() if p.is_dir())
        expected_on_disk = sorted(f"run_{i}" for i in range(1, num_workers + 1))
        if on_disk != expected_on_disk:
            failures.append(
                f"on-disk dirs mismatch: got {on_disk[:5]}... ({len(on_disk)} total) expected {expected_on_disk[:5]}... ({num_workers})"
            )

        if failures:
            print("[stress] FAIL:")
            for f in failures:
                print(f"  - {f}")
            return 1

        print(f"[stress] PASS — all {num_workers} workers claimed distinct run_N directories.")
        return 0


def main() -> int:
    num_workers = int(os.environ.get("STRESS_WORKERS", "50"))
    return _run_stress(num_workers)


if __name__ == "__main__":
    sys.exit(main())
