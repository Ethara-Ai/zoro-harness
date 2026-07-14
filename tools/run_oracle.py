#!/usr/bin/env python3
"""
run_oracle.py — Run the Oracle (rule-based, non-LLM shopkeeper) on a generated task.

The Oracle is `simulate_quality_based_environment()`. It plays the SAME task the LLM agent
plays, using a fixed hand-written policy ("buy from the highest-quality supplier, restock when
low, focus on the most historically profitable SKUs"). Its final net worth is the reference
ceiling the agent is later graded against.

Two kinds of input, deliberately kept separate:

  * THE WORLD  — comes from the task file (dataset/<archetype>/<task_id>.json). All 27 keys
                 pass straight through to RetailEnvironment: starting cash, rent, capacity,
                 selected categories, elasticity, seeds, news/review settings.
  * THE POLICY — comes from the CLI (--days, --sample_size, --bulk_qty_multiplier). These
                 are NOT part of the task. They define how the reference shopkeeper behaves,
                 and must be held constant across tasks or golden answers aren't comparable.
                 The policy values are recorded into metadata.json AND golden.json next to
                 every trajectory, alongside a SHA-256 of the oracle's Python source and a
                 SHA-256 of every external data file the task references, so a reviewer can
                 detect silent drift.

Usage:
    python tools/run_oracle.py \
        --task dataset/still_middle/<task_id>.json \
        --days 30 \
        --out runs/oracle/<task_id>

Outputs (into --out):
    tool_calls.jsonl   full Oracle trajectory (same schema the agent produces)
    order_records/     this run's isolated sqlite DB
    metadata.json      exactly what was run (task id, horizon, policy knobs, hashes)
    golden.json        the golden answer: final net worth, days survived, survival fraction,
                       policy snapshot, and reproducibility hashes
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from retail_environment import simulate_quality_based_environment

ORACLE_VERSION = "v1"

_EXTERNAL_DATA_KEYS = (
    "init_sql_path",
    "customer_data_path",
    "review_model_path",
    "review_source_path",
    "news_source_path",
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_tree(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix().encode("utf-8")
        h.update(len(rel).to_bytes(4, "big"))
        h.update(rel)
        h.update(_sha256_file(p).encode("ascii"))
    return h.hexdigest()


def _hash_external_inputs(task: dict) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for key in _EXTERNAL_DATA_KEYS:
        raw = task.get(key)
        if not raw:
            continue
        p = Path(raw)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if p.is_file():
            hashes[key] = _sha256_file(p)
        elif p.is_dir():
            hashes[key] = f"dir:{_sha256_tree(p)}"
        else:
            hashes[key] = "MISSING"
    return hashes


def _hash_policy_source() -> str:
    return _sha256_text(inspect.getsource(simulate_quality_based_environment))


def run_oracle(
    task_path: Path,
    out_dir: Path,
    days: int,
    sample_size: int,
    bulk_qty_multiplier: int,
) -> dict:
    task = json.loads(task_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    task_id = task_path.stem
    external_hashes = _hash_external_inputs(task)
    policy_source_sha256 = _hash_policy_source()

    started = time.time()
    result = simulate_quality_based_environment(
        env_config=task,
        days=days,
        sample_size=sample_size,
        bulk_qty_multiplier=bulk_qty_multiplier,
        log_dir=str(out_dir),
    )
    elapsed = time.time() - started

    trajectory = out_dir / "tool_calls.jsonl"
    if not trajectory.exists():
        raise RuntimeError(
            f"Oracle produced no trajectory at {trajectory}. "
            "The run did not log any tool calls."
        )

    (out_dir / "metadata.json").write_text(json.dumps({
        "task_id":              task_id,
        "task_path":            str(task_path),
        "oracle_version":       ORACLE_VERSION,
        "policy_source_sha256": policy_source_sha256,
        "external_data_sha256": external_hashes,
        "days":                 days,
        "sample_size":          sample_size,
        "bulk_qty_multiplier":  bulk_qty_multiplier,
        "runtime_seconds":      round(elapsed, 2),
    }, indent=2))

    (out_dir / "dataset.json").write_text(json.dumps(task, indent=2))

    days_completed = result["days_completed"]
    golden = {
        "task_id":              task_id,
        "oracle_version":       ORACLE_VERSION,
        "policy_source_sha256": policy_source_sha256,
        "external_data_sha256": external_hashes,
        "final_net_worth":      result["final_net_worth"],
        "days_completed":       days_completed,
        "days_requested":       days,
        "survival_fraction":    (days_completed / days) if days else 0.0,
        "initial_funds":        task.get("initial_funds"),
        "policy": {
            "sample_size":         sample_size,
            "bulk_qty_multiplier": bulk_qty_multiplier,
        },
        "trajectory_path": "tool_calls.jsonl",
    }
    (out_dir / "golden.json").write_text(json.dumps(golden, indent=2))
    return golden


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Oracle on a generated task")
    ap.add_argument("--task", required=True, help="Path to a generated task json")
    ap.add_argument("--out", required=True, help="Output directory for this Oracle run")
    ap.add_argument("--days", type=int, required=True,
                    help="Simulation horizon. Must match the agent run you'll compare against.")
    ap.add_argument("--sample_size", type=int, default=2,
                    help="SKUs per category the Oracle trades (oracle policy; default 2)")
    ap.add_argument("--bulk_qty_multiplier", type=int, default=4,
                    help="Restock knob: reorder threshold and order quantity multiplier (default 4)")
    args = ap.parse_args()

    task_path = Path(args.task)
    if not task_path.is_file():
        print(f"ERROR: task file not found: {task_path}", file=sys.stderr)
        sys.exit(2)

    golden = run_oracle(
        task_path=task_path,
        out_dir=Path(args.out),
        days=args.days,
        sample_size=args.sample_size,
        bulk_qty_multiplier=args.bulk_qty_multiplier,
    )

    print(
        f"\n[ORACLE {golden['oracle_version']}] task={golden['task_id'][:8]}  "
        f"net_worth={golden['final_net_worth']:.2f}  "
        f"days={golden['days_completed']}/{golden['days_requested']}  "
        f"(start funds={golden['initial_funds']})"
    )
    if golden["survival_fraction"] < 1.0:
        print("  WARNING: Oracle did not complete the full horizon — task may be too harsh.")
    print(f"  golden answer -> {Path(args.out) / 'golden.json'}")


if __name__ == "__main__":
    main()
