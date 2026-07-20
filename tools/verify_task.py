#!/usr/bin/env python3
"""
verify_task.py — General oracle-based feasibility verifier for generated tasks (both modes).

This is the Gate-A feasibility verifier the whole dataset needs (both generation modes):
for a task config, it runs the quality-oracle reference policy across a seed sweep and decides
whether the task is oracle-feasible, i.e. the reference survives the full horizon on every seed
with comfortable headroom (worst-seed final net worth >= 1.30 x initial funds).

For each config and each seed s, it runs the oracle on a COPY of the config with
config["global_random_seed"] = s (mirroring the eval harness, which overrides ONLY that key),
collects final_net_worth[s] and days_completed[s], then emits a <task_id>.verification.json
sidecar next to the config matching the §10.6.1 schema exactly.

CLI:
    python tools/verify_task.py --task <config.json>  [--out <scratch>]
    python tools/verify_task.py --dataset <dir>       [--out <scratch>]
    python tools/verify_task.py --task <config.json> --smoke   # 1 config x 2 seeds

Policy knobs (held constant; NOT part of the task): --days 180, --sample_size 2,
--bulk_qty_multiplier 4, --seeds (default range(10)).
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from retail_environment import simulate_quality_based_environment  # noqa: E402
from generator.validators import infer_archetype, match_economic_tier  # noqa: E402
# Single-source the oracle label + provenance hashing from run_oracle (the canonical runner),
# so a sidecar can prove WHICH oracle code and WHICH data files produced its numbers.
from tools.run_oracle import (  # noqa: E402
    ORACLE_VERSION,
    _hash_policy_source,
    _hash_external_inputs,
)

# Generic, meaningful schema name (this file verifies ANY task, in either generation mode;
# no run-specific codename belongs in shipped sidecars). "-v1" is a real schema version, bumped on
# any structural change to this record.
SCHEMA_VERSION = "task-verification-v1"
# Self-describing gate id: the reference must GROW to >= 1.30x initial funds (worst seed). Replaces
# the old "gateA-S3-v1", whose "S3" referenced an internal design-doc section, not anything in-repo.
GATE_VERSION = "growth-1.30x-v1"
FEASIBILITY_MULTIPLE = 1.30

# The reference (oracle) and the model bankrupt at different consecutive-negative-day counts in the
# harness (retail_environment.py vs run_env.py); recording both makes that asymmetry auditable rather
# than hidden when the numbers are later compared.
_KNOWN_ASYMMETRY = {
    "reference_bankruptcy_negative_days": 10,
    "model_bankruptcy_negative_days": 5,
}


def _generator_version() -> str:
    """Short git SHA of the repo, suffixed '-dirty' when the working tree has uncommitted changes
    (so the sidecar never points to a commit that predates the generator code that made it)."""
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        if not sha:
            return "uncommitted"
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().strip())
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return "uncommitted"


def _abs_c(config: dict) -> float:
    """Max |category_effect| over the traded (selected) categories."""
    ce = config.get("category_effects", {})
    selected = config.get("selected_categories", [])
    vals = []
    for u in selected:
        spaced = u.replace("_", " ")
        if spaced in ce:
            vals.append(abs(ce[spaced]))
    if not vals:
        vals = [abs(v) for v in ce.values()] or [0.0]
    return round(max(vals), 4)


def _news_intensity(config: dict) -> float:
    """Proxy for news pressure: base_scale x daily_count when the news channel is on, else 0."""
    if not config.get("enable_new", False):
        return 0.0
    return round(
        float(config.get("news_impact_base_scale", 0.0))
        * float(config.get("news_daily_count", 0)),
        4,
    )


def _run_one_seed(config: dict, seed: int, days: int, sample_size: int,
                  bulk_qty_multiplier: int, log_dir: Path) -> tuple[float, int]:
    """Run the oracle on a copy of config with global_random_seed=seed. Returns (net_worth, days)."""
    cfg = copy.deepcopy(config)
    cfg["global_random_seed"] = int(seed)  # override ONLY this key (mirror eval harness)
    log_dir.mkdir(parents=True, exist_ok=True)
    result = simulate_quality_based_environment(
        env_config=cfg,
        days=days,
        sample_size=sample_size,
        bulk_qty_multiplier=bulk_qty_multiplier,
        log_dir=str(log_dir),
    )
    return float(result["final_net_worth"]), int(result["days_completed"])


def verify_config(
    config_path: Path,
    days: int,
    sample_size: int,
    bulk_qty_multiplier: int,
    seeds: list[int],
    scratch: Path,
) -> dict:
    """Run the seed sweep for one config and return the sidecar dict (also written to disk)."""
    config = json.loads(config_path.read_text())
    task_id = config_path.stem
    F = float(config.get("initial_funds", 0) or 0)

    per_seed_nw: dict[str, float] = {}
    per_seed_days: dict[str, int] = {}
    for s in seeds:
        log_dir = scratch / task_id / f"seed_{s}"
        nw, dc = _run_one_seed(config, s, days, sample_size, bulk_qty_multiplier, log_dir)
        per_seed_nw[str(s)] = nw
        per_seed_days[str(s)] = dc

    worst = min(per_seed_nw.values())
    median = statistics.median(per_seed_nw.values())
    all_survived = all(dc == days for dc in per_seed_days.values())
    worst_multiple = (worst / F) if F else 0.0
    median_multiple = (median / F) if F else 0.0
    passed = (worst_multiple >= FEASIBILITY_MULTIPLE) and all_survived

    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": _generator_version(),
        "gate_version": GATE_VERSION,
        "horizon_H": days,
        "oracle_version": ORACLE_VERSION,
        # The verifiable identity behind "oracle_version": a SHA-256 of the oracle's own source,
        # and of every external data file the task reads. If either changes, these change — that is
        # what makes the label mean something.
        "oracle_source_sha256": _hash_policy_source(),
        "external_data_sha256": _hash_external_inputs(config),
        "reference_policy": {
            "name": "quality_oracle_only",
            "sample_size": sample_size,
            "bulk_qty_multiplier": bulk_qty_multiplier,
        },
        "verify_seeds": list(seeds),
        "floor_inputs": {
            "initial_funds": config.get("initial_funds"),
            "everyday_rent": config.get("everyday_rent"),
        },
        "reference_networth": {
            # best_of == quality_oracle: the oracle is the only reference policy here.
            "best_of": dict(per_seed_nw),
            "quality_oracle": dict(per_seed_nw),
        },
        "reference_days_completed": dict(per_seed_days),
        "feasibility_verdict": {
            "reference": "quality_oracle",
            "median_multiple": round(median_multiple, 4),
            "worst_seed_multiple": round(worst_multiple, 4),
            "all_survived_horizon": all_survived,
            "pass": passed,
        },
        "difficulty": {
            "archetype": infer_archetype(config),
            "tier": match_economic_tier(config),
            "proxy_features": {
                "abs_c": _abs_c(config),
                "news_intensity": _news_intensity(config),
                "traded_categories": len(config.get("selected_categories", [])),
            },
            "predicted_rank_in_archetype": None,
            "measured_winrate": None,
        },
        "known_asymmetry": dict(_KNOWN_ASYMMETRY),
    }

    sidecar_path = config_path.parent / f"{task_id}.verification.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    return sidecar


def main() -> None:
    ap = argparse.ArgumentParser(description="Oracle feasibility verifier (Gate-A) for tasks")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--task", help="Single task config json")
    src.add_argument("--dataset", help="Directory of task config jsons")
    ap.add_argument("--days", type=int, default=180, help="Horizon H (oracle policy; default 180)")
    ap.add_argument("--sample_size", type=int, default=2)
    ap.add_argument("--bulk_qty_multiplier", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)),
                    help="Seed sweep (default range(10))")
    ap.add_argument("--smoke", action="store_true",
                    help="Fast check: 1 config x 2 seeds")
    ap.add_argument("--out", default=None, help="Scratch dir for oracle logs")
    args = ap.parse_args()

    scratch = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="verify_task_"))

    if args.task:
        configs = [Path(args.task)]
    else:
        d = Path(args.dataset)
        configs = sorted(
            p for p in d.glob("*.json")
            if p.name != "manifest.json" and not p.name.endswith(".verification.json")
        )

    seeds = list(args.seeds)
    if args.smoke:
        configs = configs[:1]
        seeds = seeds[:2] if len(seeds) >= 2 else [0, 1]

    if not configs:
        print("No configs to verify.", file=sys.stderr)
        sys.exit(2)

    n_pass = 0
    n_error = 0
    for cfg_path in configs:
        # Isolate each config: over a large --dataset run, one crashing oracle result must not
        # abort the whole pass. Failures are logged and counted; the run exits non-zero if any.
        try:
            sidecar = verify_config(
                cfg_path, args.days, args.sample_size, args.bulk_qty_multiplier, seeds, scratch,
            )
        except Exception as exc:  # noqa: BLE001 — one bad task should not sink the batch
            n_error += 1
            print(f"[ERROR] {cfg_path.stem[:8]}  {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        v = sidecar["feasibility_verdict"]
        status = "PASS" if v["pass"] else "FAIL"
        if v["pass"]:
            n_pass += 1
        print(
            f"[{status}] {cfg_path.stem[:8]}  "
            f"worst={v['worst_seed_multiple']:.3f}x median={v['median_multiple']:.3f}x  "
            f"survived_all={v['all_survived_horizon']}  "
            f"-> {cfg_path.parent / (cfg_path.stem + '.verification.json')}"
        )

    print(f"\nVerify complete: {n_pass}/{len(configs)} feasible (>= {FEASIBILITY_MULTIPLE}x, all seeds survived)"
          + (f"; {n_error} errored" if n_error else ""))
    if n_error:
        sys.exit(1)


if __name__ == "__main__":
    main()
