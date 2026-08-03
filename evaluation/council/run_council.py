#!/usr/bin/env python3
"""evaluation.council.run_council — the LLM-council entry point a task's test.sh invokes.

test.sh calls this as::

    python -m evaluation.council.run_council \
        --rubrics tests/rubrics.json --truth TRUTH.md \
        --traj-path <agent run> --dataset-path environment/dataset.json \
        --out verifier/council_results.json

The council implementation is shared in the llm_council package; this module is the thin
entry point that adapts the test.sh CLI to it: it builds the per-rubric judge inputs from the
task's rubrics.json + the agent trajectory, runs the judge panel, and writes council_results.json.

It never hard-fails the pipeline — if the judge bridges are unreachable it emits a
council_status="unavailable" result (allow_degraded), which the aggregator scores pytest-only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The council logic lives in the shared llm_council package; this package is its entry point.
from llm_council.prepare_inputs import build as build_inputs
from llm_council.run_council import DEFAULT_PANEL, load_panel, run_council


def _derive_task_id(*paths: str | None) -> str:
    """Best-effort: the task UUID is usually a path component of the trajectory or dataset."""
    for p in paths:
        if not p:
            continue
        for part in Path(p).parts:
            if len(part) == 36 and part.count("-") == 4:
                return part
    return "unknown"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Run the LLM council over a task's rubrics and an agent trajectory.")
    ap.add_argument("--rubrics", required=True, help="tests/rubrics.json (JSON array of criteria)")
    ap.add_argument("--traj-path", required=True, help="agent run dir or tool_calls.jsonl")
    ap.add_argument("--out", required=True, help="council_results.json output path")
    # accepted for the test.sh contract:
    ap.add_argument("--truth", default=None, help="TRUTH.md (context only; the rubrics carry the criteria)")
    ap.add_argument("--dataset-path", default=None, help="environment/dataset.json (context only)")
    ap.add_argument("--task-id", default=None, help="task UUID (derived from the paths if omitted)")
    ap.add_argument("--trajectory-model", default=None,
                    help="model id that produced this trajectory (e.g. claude-opus-4-8, gpt-5.6); "
                         "selects the judge panel via judges.yaml 'panels'. Auto-detected from "
                         "sibling metadata.json/manifest.json if omitted; unmapped -> 'default' panel.")
    ap.add_argument("--panel", default=str(DEFAULT_PANEL), help="judge-panel yaml (default: the shared panel)")
    ap.add_argument("--skip-preflight", action="store_true", help="offline/testing only; skip the bridge ping")
    ap.add_argument("--cache", default=None, help="optional verdict cache json path")
    args = ap.parse_args(argv)

    rubrics = json.loads(Path(args.rubrics).read_text())
    task_id = args.task_id or _derive_task_id(args.traj_path, args.dataset_path)

    inputs = build_inputs(rubrics, args.traj_path, task_id,
                          trajectory_model=args.trajectory_model)
    cfg = load_panel(Path(args.panel))
    # allow_degraded=True: a bridge outage yields a council_unavailable result, never a crash,
    # so the aggregator falls back to a pytest-only score rather than the pipeline stopping.
    results = run_council(
        inputs,
        cfg,
        allow_degraded=True,
        skip_preflight=args.skip_preflight,
        cache_path=Path(args.cache) if args.cache else None,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(
        f"council_status={results.get('council_status')} "
        f"met={results.get('n_met')} unmet={results.get('n_unmet')} "
        f"unassessed={results.get('n_unassessed')} -> {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
