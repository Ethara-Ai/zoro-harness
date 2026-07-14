#!/usr/bin/env python3
"""Paired-seed dispatcher for retail-agent evaluation.

One seed drives every (model, agent) cell so cross-cell differences are attributable
to the agent/model, not to environment stochasticity. Set of agents includes LLM
variants and non-LLM classical baselines; baselines skip the model axis.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from util.default_config import (
    create_dynamic_hard_config,
    create_dynamic_middle_config,
    create_still_hard_config,
    create_still_middle_config,
)


CONFIG_BUILDERS = {
    "dynamic_hard": create_dynamic_hard_config,
    "dynamic_middle": create_dynamic_middle_config,
    "still_hard": create_still_hard_config,
    "still_middle": create_still_middle_config,
}


def _load_llm_agent(module_path: str, loop_name: str) -> Callable:
    mod = __import__(module_path, fromlist=[loop_name])
    return getattr(mod, loop_name)


def _load_classical_agent(module_path: str, run_name: str) -> Callable:
    mod = __import__(module_path, fromlist=[run_name])
    return getattr(mod, run_name)


AGENT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "plan-and-act": {
        "kind": "llm",
        "module": "agents.run_plan_and_act",
        "loop": "run_plan_and_act_loop",
        "takes_goal": True,
        "loop_kwargs_extra": {"max_execution_turns_per_day": "max_execution_turns"},
        "log_prefix": "plan_and_act",
    },
    "step-reflection": {
        "kind": "llm",
        "module": "agents.run_step_reflection",
        "loop": "run_step_reflection_loop",
        "takes_goal": True,
        "loop_kwargs_extra": {"max_steps_per_day": "max_execution_turns"},
        "log_prefix": "step_reflection",
    },
    "sham-reflection": {
        "kind": "llm",
        "module": "agents.run_sham_reflection",
        "loop": "run_sham_reflection_loop",
        "takes_goal": True,
        "loop_kwargs_extra": {"max_steps_per_day": "max_execution_turns"},
        "log_prefix": "sham_reflection",
    },
    "sparse-reflection": {
        "kind": "llm",
        "module": "agents.run_sparse_reflection",
        "loop": "run_sparse_reflection_loop",
        "takes_goal": True,
        "loop_kwargs_extra": {"max_steps_per_day": "max_execution_turns"},
        "log_prefix": "sparse_reflection",
    },
    "ss-policy": {
        "kind": "classical",
        "module": "agents.run_ss_policy",
        "loop": "run_ss_policy",
        "log_prefix": "ss_policy",
    },
    "newsvendor": {
        "kind": "classical",
        "module": "agents.run_newsvendor",
        "loop": "run_newsvendor",
        "log_prefix": "newsvendor",
    },
}


MODELS: Dict[str, Dict[str, Optional[str]]] = {
    "deepseek-v3.2": {
        "model": "deepseek-v3.2-exp",
        "display": "DeepSeek-V3.2",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "default_base_url": "https://api.deepseek.com/v1",
    },
    "glm-4.6": {
        "model": "glm-4.6",
        "display": "GLM-4.6",
        "api_key_env": "GLM_API_KEY",
        "base_url_env": "GLM_BASE_URL",
        "default_base_url": "https://open.bigmodel.cn/api/paas/v4",
    },
    "kimi-k2-thinking": {
        "model": "kimi-k2-thinking",
        "display": "Kimi-K2 (Thinking)",
        "api_key_env": "KIMI_API_KEY",
        "base_url_env": "KIMI_BASE_URL",
        "default_base_url": "https://api.moonshot.cn/v1",
    },
    "claude-sonnet-cc": {
        "model": "sonnet",
        "display": "Claude Sonnet (Code)",
        "api_key_env": "ZORO_CC_BRIDGE_SECRET",
        "base_url_env": "ZORO_CC_BRIDGE_URL",
        "default_base_url": "http://127.0.0.1:8787/v1",
    },
    "claude-opus-cc": {
        "model": "opus",
        "display": "Claude Opus (Code)",
        "api_key_env": "ZORO_CC_BRIDGE_SECRET",
        "base_url_env": "ZORO_CC_BRIDGE_URL",
        "default_base_url": "http://127.0.0.1:8787/v1",
    },
}


CLASSICAL_DISPLAY: Dict[str, str] = {
    "ss-policy": "(s,S) Policy",
    "newsvendor": "Newsvendor",
}


def _arm_key(agent: str, model_key: Optional[str]) -> str:
    return f"{agent}__{model_key}" if model_key else agent


def _display_name(agent: str, model_key: Optional[str]) -> str:
    if model_key:
        model_disp = MODELS.get(model_key, {}).get("display") or model_key
        return f"{agent} \u00d7 {model_disp}"
    return CLASSICAL_DISPLAY.get(agent, agent)


DEFAULT_GOAL = (
    "Please optimize inventory assortment and turnover for long-term store viability: "
    "minimize stockouts, shrink, and cash risk while covering rent/operating costs "
    "and growing gross margin via data-driven, proactive decisions."
)


def build_config(config_type: str, seed: int, db_root: Path, run_tag: str) -> Dict[str, Any]:
    builder = CONFIG_BUILDERS[config_type]
    config = builder()
    config["global_random_seed"] = int(seed)
    per_run_db = db_root / run_tag
    per_run_db.mkdir(parents=True, exist_ok=True)
    config["order_record_dir"] = str(per_run_db)
    return config


def resolve_client(model_key: str):
    from openai import OpenAI

    spec = MODELS[model_key]
    api_key = os.getenv(spec["api_key_env"] or "", "") or ""
    base_url = os.getenv(spec["base_url_env"] or "", spec.get("default_base_url") or "")
    return OpenAI(api_key=api_key, base_url=base_url), spec["model"]


def _cell_run_tag(agent: str, model_key: Optional[str], seed: int) -> str:
    model_slug = model_key.replace(".", "_").replace("-", "_") if model_key else "none"
    return f"{agent}__{model_slug}__seed{seed}"


def _build_goal(config: Dict[str, Any]) -> str:
    context = (
        f"You are operating store {config.get('store_id', '')} with initial funds "
        f"of {config.get('initial_funds', '')}. Your available operational data ranges "
        f"from {config.get('data_begin_time', '')} to {config.get('data_end_time', '')}. "
        f"Today is {config.get('store_begin_time', '')}. Daily Rent is {config.get('everyday_rent', 0)}."
    )
    return f"{context}\n\n{DEFAULT_GOAL}"


def _extract_metrics_from_env(env: Any) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {"final_funds": None, "final_net_worth": None, "current_date": None}
    try:
        metrics["final_funds"] = float(getattr(env, "funds", 0.0))
    except Exception:
        pass
    try:
        metrics["final_net_worth"] = float(getattr(env, "net_worth", metrics["final_funds"]))
    except Exception:
        pass
    try:
        metrics["current_date"] = str(getattr(env, "current_date", ""))
    except Exception:
        pass
    return metrics


_WARNED_ANALYZER_IMPORT = False
_WARNED_TOOL_CALLS_MISSING = False
_WARNED_ANALYZER_FAILED = False


def _warn_once(flag_name: str, message: str) -> None:
    if globals().get(flag_name):
        return
    globals()[flag_name] = True
    print(f"[WARN] {message}", file=sys.stderr)


def _compute_cell_aggregates(cell_log_dir: Path) -> Dict[str, Any]:
    tool_calls_path = cell_log_dir / "tool_calls.jsonl"
    if not tool_calls_path.exists():
        _warn_once(
            "_WARNED_TOOL_CALLS_MISSING",
            f"tool_calls.jsonl missing (first miss: {tool_calls_path}); "
            "cell aggregates (profit/service level/etc.) will be blank. "
            "Downstream analyzer will treat these cells as no-metrics.",
        )
        return {}
    try:
        from analysis.analyze_experiment_data.analyze_paper_data_final import analyze_tool_calls
    except Exception as exc:
        _warn_once(
            "_WARNED_ANALYZER_IMPORT",
            f"analyze_paper_data_final not importable ({type(exc).__name__}: {exc}); "
            "cell aggregates will be blank for every remaining cell.",
        )
        return {}
    try:
        raw = analyze_tool_calls(str(tool_calls_path))
    except Exception as exc:
        _warn_once(
            "_WARNED_ANALYZER_FAILED",
            f"analyze_tool_calls raised {type(exc).__name__}: {exc} "
            f"(first failure on {tool_calls_path}). Cell aggregates will be blank.",
        )
        return {}
    if not raw:
        return {}
    return {
        "avg_daily_profit": float(raw.get("avg_daily_profit", 0.0) or 0.0),
        "avg_daily_sales": float(raw.get("avg_daily_sales", 0.0) or 0.0),
        "days_survived": int(raw.get("run_days", 0) or 0),
        "expiry_ratio": float(raw.get("expired_ratio", 0.0) or 0.0),
        "return_ratio": float(raw.get("return_ratio", 0.0) or 0.0),
        "inventory_turnover": float(raw.get("inventory_turnover", 0.0) or 0.0),
        "holding_units_days": float(raw.get("holding_units_days", 0.0) or 0.0),
        "mean_service_level": float(raw.get("mean_service_level", 1.0) or 1.0),
        "stockout_days_rate": float(raw.get("stockout_days_rate", 0.0) or 0.0),
        "orders_per_day": float(raw.get("orders_per_day", 0.0) or 0.0),
        "total_ordered": int(raw.get("total_ordered", 0) or 0),
        "total_sold": int(raw.get("total_sold", 0) or 0),
        "total_expired": int(raw.get("total_expired", 0) or 0),
        "total_returns": int(raw.get("total_returns", 0) or 0),
        "tool_error_rate": float(raw.get("tool_error_rate", 0.0) or 0.0),
    }


def _run_cell(
    agent: str,
    model_key: Optional[str],
    seed: int,
    config_type: str,
    max_days: int,
    max_execution_turns: int,
    output_dir: Path,
    db_root: Path,
) -> Dict[str, Any]:
    from retail_environment import RetailEnvironment

    entry = AGENT_REGISTRY[agent]
    run_tag = _cell_run_tag(agent, model_key, seed)
    cell_log_dir = output_dir / run_tag
    cell_log_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(config_type, seed, db_root, run_tag)
    config["log_dir"] = str(cell_log_dir)
    (cell_log_dir / "config.json").write_text(
        json.dumps({k: v for k, v in config.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))},
                   indent=2, default=str, ensure_ascii=False)
    )

    print(f"\n{'='*70}\nCELL: {run_tag}\n{'='*70}")
    started = datetime.utcnow().isoformat() + "Z"

    result: Dict[str, Any] = {
        "agent": agent,
        "model_key": model_key,
        "seed": seed,
        "config_type": config_type,
        "started": started,
        "run_tag": run_tag,
        "status": "pending",
    }

    try:
        env = RetailEnvironment(config)

        if entry["kind"] == "llm":
            loop_fn = _load_llm_agent(entry["module"], entry["loop"])
            client, resolved_model = resolve_client(model_key) if model_key else (None, None)

            loop_kwargs: Dict[str, Any] = {
                "env": env,
                "model": resolved_model,
                "log_path": cell_log_dir / f"{entry['log_prefix']}_history.json",
                "log_dir": cell_log_dir,
                "max_days": max_days,
                "client": client,
            }
            if entry.get("takes_goal"):
                loop_kwargs["goal"] = _build_goal(config)
            for kw, source_kw in entry.get("loop_kwargs_extra", {}).items():
                if source_kw == "max_execution_turns":
                    loop_kwargs[kw] = max_execution_turns

            loop_fn(**loop_kwargs)
        else:
            run_fn = _load_classical_agent(entry["module"], entry["loop"])
            run_fn(env=env, max_days=max_days, log_dir=cell_log_dir, seed=seed)

        result.update(_extract_metrics_from_env(env))
        result["status"] = "success"
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        print(f"[ERROR] {run_tag}: {result['error']}")

    result["arm_key"] = _arm_key(agent, model_key)
    result["display_name"] = _display_name(agent, model_key)
    result.update(_compute_cell_aggregates(cell_log_dir))
    result["finished"] = datetime.utcnow().isoformat() + "Z"
    (cell_log_dir / "cell_result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def _to_analyzer_row(result: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(result)
    row["model_key"] = result.get("arm_key") or _arm_key(result["agent"], result.get("model_key"))
    row["model"] = result.get("display_name") or _display_name(result["agent"], result.get("model_key"))
    row["environment"] = result.get("config_type", "")
    row["framework"] = result.get("agent", "")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired-seed retail-agent experiment dispatcher")
    parser.add_argument("--agents", nargs="+", default=list(AGENT_REGISTRY.keys()),
                        choices=list(AGENT_REGISTRY.keys()),
                        help="Agent variants to run (default: all)")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        choices=list(MODELS.keys()),
                        help="Model keys (ignored for classical baselines)")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)),
                        help="Seeds; one seed drives every (model, agent) cell (default: 0..9)")
    parser.add_argument("--config_type", type=str, default="still_middle",
                        choices=list(CONFIG_BUILDERS.keys()))
    parser.add_argument("--max_days", type=int, default=30)
    parser.add_argument("--max_execution_turns", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="experiments/very_easy/results")
    parser.add_argument("--db_root", type=str, default="model_run_time/paired_seed")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the plan and exit without launching cells")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    db_root = Path(args.db_root)
    db_root.mkdir(parents=True, exist_ok=True)

    llm_agents = [a for a in args.agents if AGENT_REGISTRY[a]["kind"] == "llm"]
    classical_agents = [a for a in args.agents if AGENT_REGISTRY[a]["kind"] == "classical"]

    plan: List[Dict[str, Any]] = []
    for seed in args.seeds:
        for agent in llm_agents:
            for model_key in args.models:
                plan.append({"agent": agent, "model": model_key, "seed": seed})
        for agent in classical_agents:
            plan.append({"agent": agent, "model": None, "seed": seed})

    print("=" * 70)
    print(f"PAIRED-SEED EXPERIMENT PLAN ({len(plan)} cells)")
    print("=" * 70)
    print(f"Seeds: {args.seeds}")
    print(f"LLM agents:       {llm_agents}")
    print(f"Classical agents: {classical_agents}")
    print(f"Models:           {args.models}")
    print(f"config_type={args.config_type}, max_days={args.max_days}")
    print(f"Output: {output_dir}")

    if args.dry_run:
        for cell in plan:
            print(f"  - {_cell_run_tag(cell['agent'], cell['model'], cell['seed'])}")
        print("\n[dry-run] exiting without executing cells")
        return

    results: List[Dict[str, Any]] = []
    for cell in plan:
        results.append(_run_cell(
            agent=cell["agent"],
            model_key=cell["model"],
            seed=cell["seed"],
            config_type=args.config_type,
            max_days=args.max_days,
            max_execution_turns=args.max_execution_turns,
            output_dir=output_dir,
            db_root=db_root,
        ))

    finished_ts = datetime.utcnow().isoformat() + "Z"

    summary_path = output_dir / "paired_seed_summary.json"
    summary_path.write_text(json.dumps({
        "started": finished_ts,
        "config_type": args.config_type,
        "max_days": args.max_days,
        "seeds": args.seeds,
        "agents": args.agents,
        "models": args.models,
        "cells": results,
    }, indent=2, default=str))
    print(f"\nSummary: {summary_path}")

    analyzer_rows = [_to_analyzer_row(r) for r in results]
    arm_display_map = {r["arm_key"]: r["display_name"] for r in results if r.get("arm_key")}
    analyzer_path = output_dir / "very_easy_summary.json"
    analyzer_path.write_text(json.dumps({
        "config": {
            "config_type": args.config_type,
            "max_days": args.max_days,
            "seeds": args.seeds,
            "agents": args.agents,
            "models": args.models,
        },
        "models": arm_display_map,
        "results": analyzer_rows,
        "timestamp": finished_ts,
    }, indent=2, default=str))
    print(f"Analyzer-compatible summary: {analyzer_path}")

    ok = sum(1 for r in results if r["status"] == "success")
    print(f"\nDone. {ok}/{len(results)} cells succeeded.")


if __name__ == "__main__":
    main()
