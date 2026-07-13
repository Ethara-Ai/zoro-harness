#!/usr/bin/env python3
"""A3 ablation: sparse reflection (once per day, at end-of-day).

Uses the same reflection quality as run_step_reflection but only fires when the day
completes (end_today or forced end after max_steps). Isolates dose/frequency effects
from reflection *quality* effects.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PARENT))
sys.path.insert(0, str(_PARENT / "module"))

from openai import OpenAI

from retail_environment import RetailEnvironment
from util.default_config import (
    create_dynamic_hard_config,
    create_dynamic_middle_config,
    create_still_hard_config,
    create_still_middle_config,
)

from agents.run_step_reflection import (
    DEFAULT_GOAL,
    DEFAULT_MODEL,
    build_goal,
    create_openai_client,
    recover_from_checkpoint,
    recover_from_day_checkpoint,
    run_step_reflection_loop,
)


def _end_of_day_gate(day: int, step: int, execution_phase_complete: bool) -> bool:
    return bool(execution_phase_complete)


def run_sparse_reflection_loop(
    goal: str,
    env: RetailEnvironment,
    model: str = DEFAULT_MODEL,
    log_path: Path = Path("logs/run_env_history.json"),
    max_input_tokens: int = 60000,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_interval: int = 10,
    initial_messages: Optional[List[Dict[str, Any]]] = None,
    start_turn: int = 0,
    start_day: int = 1,
    initial_memory: Optional[List[str]] = None,
    log_dir: Optional[Path] = None,
    max_days: int = 30,
    max_steps_per_day: int = 20,
    client: Optional[OpenAI] = None,
) -> None:
    return run_step_reflection_loop(
        goal=goal,
        env=env,
        model=model,
        log_path=log_path,
        max_input_tokens=max_input_tokens,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        initial_messages=initial_messages,
        start_turn=start_turn,
        start_day=start_day,
        initial_memory=initial_memory,
        log_dir=log_dir,
        max_days=max_days,
        max_steps_per_day=max_steps_per_day,
        client=client,
        generate_reflection_fn=None,
        reflection_gate=_end_of_day_gate,
    )


def build_log_path(base_dir: str = "logs") -> tuple[Path, str]:
    from datetime import datetime
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root = f"run_sparse_reflection_{timestamp}"
    return Path(os.path.join(base_dir, f"{root}/{root}.json")), os.path.join(base_dir, root)


def main() -> None:
    parser = argparse.ArgumentParser(description="A3 sparse-reflection ablation runner")
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--recover_turn", type=int)
    parser.add_argument("--recover_day", type=int)
    parser.add_argument("--checkpoint_interval", type=int, default=20)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--db_path", type=str, default=None)
    parser.add_argument("--config_type", type=str,
                        choices=["dynamic_hard", "dynamic_middle", "still_hard", "still_middle"],
                        default="still_middle")
    parser.add_argument("--max_input_tokens", type=int, default=50000)
    parser.add_argument("--max_days", type=int, default=30)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    args = parser.parse_args()

    builders = {
        "dynamic_hard": create_dynamic_hard_config,
        "dynamic_middle": create_dynamic_middle_config,
        "still_hard": create_still_hard_config,
        "still_middle": create_still_middle_config,
    }
    config = builders[args.config_type]()
    config["order_record_dir"] = args.db_path if args.db_path is not None else "model_run_time"

    log_path, env_log_path = build_log_path()
    goal = build_goal(DEFAULT_GOAL, config)
    config["log_dir"] = env_log_path

    os.makedirs(env_log_path, exist_ok=True)
    with open(os.path.join(env_log_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2, default=str)

    args_dict = {**vars(args), "api_key": "***" if args.api_key else None}
    (Path(env_log_path) / "args.json").write_text(json.dumps(args_dict, indent=2, default=str))

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else Path(env_log_path) / "checkpoints"
    client = create_openai_client(api_key=args.api_key, base_url=args.base_url)

    initial_messages = None
    start_turn = 0
    start_day = 1
    initial_memory: List[str] = []

    if args.recover_day is not None:
        env, initial_messages, initial_memory, start_day, start_turn = recover_from_day_checkpoint(
            checkpoint_dir, args.recover_day, config
        )
    elif args.recover_turn is not None:
        env, initial_messages = recover_from_checkpoint(checkpoint_dir, args.recover_turn, config)
        start_turn = args.recover_turn
    else:
        env = RetailEnvironment(config)

    run_sparse_reflection_loop(
        goal=goal,
        env=env,
        model=args.model,
        log_path=log_path,
        max_input_tokens=args.max_input_tokens,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        initial_messages=initial_messages,
        start_turn=start_turn,
        start_day=start_day,
        initial_memory=initial_memory,
        log_dir=Path(env_log_path),
        max_days=args.max_days,
        max_steps_per_day=args.max_steps,
        client=client,
    )


if __name__ == "__main__":
    main()
