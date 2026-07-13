#!/usr/bin/env python3
"""Batch grader: iterates all trajectory dirs, grades each, aggregates results."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # zoro/harness/
from tools.grade_task import grade_task


def grade_batch(trajectories_dir: Path, out_path: Path) -> dict:
    trajectories_dir = Path(trajectories_dir)
    task_dirs = sorted(d for d in trajectories_dir.iterdir() if d.is_dir())
    if not task_dirs:
        print(f"[WARN] No trajectory directories found in {trajectories_dir}", file=sys.stderr)

    results = []
    per_archetype: dict = defaultdict(lambda: {"n_total": 0, "n_passed": 0})

    for task_dir in task_dirs:
        if not (task_dir / "tool_calls.jsonl").exists():
            print(f"[SKIP] {task_dir.name}: missing tool_calls.jsonl")
            continue
        if not (task_dir / "dataset.json").exists():
            print(f"[SKIP] {task_dir.name}: missing dataset.json")
            continue

        try:
            grade = grade_task(task_dir)
        except Exception as e:
            print(f"[ERROR] {task_dir.name}: {e}", file=sys.stderr)
            continue

        # Infer archetype for per-archetype aggregation
        try:
            ec = json.loads((task_dir / "dataset.json").read_text()).get("env_config", {})
            mode = "dynamic" if "dynamic" in ec.get("data_dir", "") else "still"
            diff = "hard" if len(ec.get("selected_categories", [])) > 10 else "middle"
            archetype = f"{mode}_{diff}"
        except Exception:
            archetype = "unknown"

        results.append(grade)
        per_archetype[archetype]["n_total"] += 1
        per_archetype[archetype]["n_passed"] += int(grade["passed"])

    n_total = len(results)
    n_passed = sum(1 for g in results if g["passed"])
    pass_rate = n_passed / n_total if n_total else 0.0

    output = {
        "n_total":             n_total,
        "n_passed":            n_passed,
        "pass_rate":           pass_rate,
        "per_archetype_stats": {k: dict(v) for k, v in per_archetype.items()},
        "tasks":               results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"[BATCH] {n_passed}/{n_total} passed ({pass_rate * 100:.1f}%)  →  {out_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade all task trajectories in a directory")
    parser.add_argument("--trajectories_dir", required=True,
                        help="Directory containing {task_id}/ subdirectories")
    parser.add_argument("--out", required=True,
                        help="Output path for grades.json")
    args = parser.parse_args()
    grade_batch(Path(args.trajectories_dir), Path(args.out))


if __name__ == "__main__":
    main()
