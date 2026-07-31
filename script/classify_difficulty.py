#!/usr/bin/env python3
"""
classify_difficulty.py — Compute a task's difficulty level and stamp it into task.toml.

Difficulty model (swarm-synthesized; economic headroom dominant, load secondary):

    score = 100 * (0.50*E + 0.32*L + 0.10*C + 0.08*T)      # frozen anchors, clip to [0,1]

      E = clip((6.0 - oracle_multiple) / 4.38)              # economic tightness (PRIMARY, inverse)
      L = 0.75*clip((n_categories - 4)/16) + 0.25*(enable_new?1:0)   # operational load
      C = clip((|cannibalization| - 0.20)/0.20)             # cross-SKU pricing sensitivity
      T = 1 if archetype in {still_middle, dynamic_middle} else 0     # middle behavioral-trap floor

    bands:  trivial <14.6 | easy 14.6-32.2 | medium 32.2-53.5 | hard 53.5-66.95 | expert >=66.95

`oracle_multiple` is the dominant (0.50) input. Until a real per-task oracle run exists,
it is decoded from the generator's grid tables (INDICATIVE) via the task's global_random_seed
(seed = base + rank). Stamped rows carry difficulty_basis="indicative_grid" so the provisional
status is explicit. When measured oracle multiples exist, pass them in and set basis="measured".

Usage:
    python script/classify_difficulty.py                 # stamp default dataset dir
    python script/classify_difficulty.py --src <dir> --dry-run
    python script/classify_difficulty.py --only <task_id>
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from curate_harbor import DEFAULT_SRC, derive_config, render_task_toml  # noqa: E402
from generate_truth import read_json  # noqa: E402

# --- frozen scoring anchors (generator design bounds, NOT this batch's min/max) ---
OM_MIN, OM_MAX = 1.62, 6.0
NCAT_MIN, NCAT_MAX = 4, 20
C_MIN, C_MAX = 0.20, 0.40
W_E, W_L, W_C, W_T = 0.50, 0.32, 0.10, 0.08
BANDS = [(14.6, "trivial"), (32.2, "easy"), (53.5, "medium"), (66.95, "hard"), (float("inf"), "expert")]

# --- generator grid tables: (tier, |c|, indicative_oracle_multiple) ordered by rank ---
_GRID_HARD_DYNAMIC = [2.09, 2.37, 2.65, 2.92, 3.27, 3.80, 4.20, 4.5]
_GRID_HARD_STILL = [1.62, 1.81, 2.47, 2.6, 2.86, 3.1, 3.43, 3.7]
_GRID_MIDDLE = [2.2, 2.5, 3.0, 4.3, 5.0, 5.5, 6.0]  # dynamic_middle AND still_middle share it
_SEED_BASE = {100_000: "dynamic_hard", 200_000: "still_hard",
              300_000: "dynamic_middle", 400_000: "still_middle"}


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def decode_oracle_multiple(dataset: dict) -> tuple[float | None, str]:
    """Decode the INDICATIVE oracle multiple from the grid seed (seed = base + rank).

    Returns (multiple, basis). basis='indicative_grid' on success, 'unknown' if the seed
    is not a recognizable grid seed (e.g. a random spread-mode task)."""
    seed = dataset.get("global_random_seed")
    if not isinstance(seed, int):
        return None, "unknown"
    base = (seed // 100_000) * 100_000
    arch = _SEED_BASE.get(base)
    rank = seed - base
    if arch is None or rank < 1:
        return None, "unknown"
    if arch == "dynamic_hard":
        table = _GRID_HARD_DYNAMIC
    elif arch == "still_hard":
        table = _GRID_HARD_STILL
    else:
        table = _GRID_MIDDLE
    if rank > len(table):
        return None, "unknown"
    return table[rank - 1], "indicative_grid"


def score_task(dataset: dict, oracle_multiple: float) -> tuple[float, str, dict]:
    """Return (score, level, term_breakdown) for a task given its oracle_multiple."""
    cats = dataset.get("selected_categories") or []
    n_cat = len(cats)
    ce = dataset.get("category_effects") or {}
    c_abs = abs(next(iter(ce.values()))) if ce else C_MIN
    enable_new = bool(dataset.get("enable_new"))
    is_middle = n_cat <= 10  # middle = reduced assortment (4-7); hard = full 20

    E = _clip((OM_MAX - oracle_multiple) / (OM_MAX - OM_MIN))
    L = 0.75 * _clip((n_cat - NCAT_MIN) / (NCAT_MAX - NCAT_MIN)) + 0.25 * (1 if enable_new else 0)
    C = _clip((c_abs - C_MIN) / (C_MAX - C_MIN))
    T = 1 if is_middle else 0

    score = 100 * (W_E * E + W_L * L + W_C * C + W_T * T)
    score = round(max(0.0, min(100.0, score)), 2)
    level = next(name for cut, name in BANDS if score < cut)
    return score, level, {"E": round(E, 4), "L": round(L, 4), "C": round(C, 4), "T": T}


def classify(dataset: dict, oracle_multiple: float | None = None,
             basis: str | None = None) -> dict:
    """Full classification for a task. If oracle_multiple is None, decode the indicative one."""
    if oracle_multiple is None:
        oracle_multiple, basis = decode_oracle_multiple(dataset)
        if oracle_multiple is None:
            raise ValueError(
                "cannot decode an indicative oracle_multiple from the seed; "
                "pass --oracle-multiple or run the oracle for this task"
            )
    else:
        basis = basis or "measured"
    score, level, terms = score_task(dataset, oracle_multiple)
    return {"level": level, "score": score, "oracle_multiple": oracle_multiple,
            "basis": basis, "terms": terms}


def main() -> int:
    ap = argparse.ArgumentParser(description="Classify + stamp difficulty into each task.toml.")
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC, help=f"dataset dir (default: {DEFAULT_SRC})")
    ap.add_argument("--only", default=None, help="one task id only")
    ap.add_argument("--dry-run", action="store_true", help="print classification; do not write")
    args = ap.parse_args()

    if not args.src.is_dir():
        print(f"error: not a directory: {args.src}", file=sys.stderr)
        return 1

    task_dirs = sorted(p for p in args.src.iterdir()
                       if p.is_dir() and (p / "task.toml").exists() and (p / "environment" / "dataset.json").exists())
    if args.only:
        task_dirs = [p for p in task_dirs if p.name == args.only]
    if not task_dirs:
        print(f"no task dirs found in {args.src}", file=sys.stderr)
        return 1

    from collections import Counter
    dist: Counter = Counter()
    n = 0
    for td in task_dirs:
        dataset = read_json(td / "environment" / "dataset.json")
        try:
            diff = classify(dataset)
        except ValueError as e:
            print(f"  SKIP {td.name}: {e}", file=sys.stderr)
            continue
        dist[diff["level"]] += 1
        n += 1
        line = (f"{td.name[:8]}  {diff['level']:7} score={diff['score']:5}  "
                f"om={diff['oracle_multiple']:>4} ({diff['basis']})  "
                f"E={diff['terms']['E']} L={diff['terms']['L']} C={diff['terms']['C']} T={diff['terms']['T']}")
        print(("[dry] " if args.dry_run else "      ") + line)
        if args.dry_run:
            continue
        # re-render task.toml with difficulty block, preserving the existing description
        toml_text = (td / "task.toml").read_text()
        m = re.search(r'description\s*=\s*"([^"]*)"', toml_text)
        desc = m.group(1) if m else ""
        cfg = derive_config(dataset, td.name)
        (td / "task.toml").write_text(render_task_toml(cfg, desc, difficulty=diff))

    ordered = {k: dist[k] for k in ["trivial", "easy", "medium", "hard", "expert"]}
    print(f"\n{'[dry-run] ' if args.dry_run else ''}classified {n} tasks — distribution: {ordered}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
