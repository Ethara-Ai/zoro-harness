#!/usr/bin/env python3
"""Per-task grader: reads trajectory dir, computes metrics, emits grade.json."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # zoro/harness/
from analysis.analyze_experiment_data.analyze_paper_data_final import analyze_tool_calls


def grade_task(task_dir: Path) -> dict:
    task_dir = Path(task_dir)
    ds = json.load((task_dir / "dataset.json").open())
    stats = analyze_tool_calls(task_dir / "tool_calls.jsonl")
    if stats is None:
        raise RuntimeError(
            f"analyze_tool_calls returned None — no end_today records in {task_dir}"
        )

    history = stats["networth_history"]
    if not history:
        raise RuntimeError(f"networth_history is empty in {task_dir}")
    final_net_worth = history[-1]["net_worth"]

    # Flat schema: ds IS the env_config; task_id derived from directory name
    task_id = task_dir.name
    n_categories = len(ds["selected_categories"])
    corrected_avg_per_category = (
        stats["avg_daily_sales"] / n_categories if n_categories else 0.0
    )

    grade = {
        "task_id":                      task_id,
        "final_net_worth":              final_net_worth,
        "run_days":                     stats["run_days"],
        "avg_daily_sales_per_category": corrected_avg_per_category,
        "metrics":                      stats,
    }
    (task_dir / "grade.json").write_text(json.dumps(grade, indent=2))
    return grade


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade a single task trajectory")
    parser.add_argument("--task_dir", required=True,
                        help="Path to trajectories/{task_id}/ directory")
    args = parser.parse_args()
    task_dir = Path(args.task_dir)
    grade = grade_task(task_dir)
    print(
        f"task_id={grade['task_id']}  "
        f"net_worth={grade['final_net_worth']:.2f}  "
        f"run_days={grade['run_days']}"
    )


if __name__ == "__main__":
    main()
