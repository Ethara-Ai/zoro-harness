"""
generate_datasets.py — Batch generator for RetailBench dataset tasks.

Usage:
    python zoro/harness/generator/generate_datasets.py \
        --archetype {dynamic_hard|dynamic_middle|still_hard|still_middle|all} \
        --mode {spread|grid} \
        --n 10000 \
        --out zoro/harness/dataset/ \
        --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # zoro/harness/ → util.* importable

from generator.templates.archetypes import ARCHETYPES
from generator.validators import (
    ALL_CATEGORIES,
    ELASTICITY_PROFILES,
    ECONOMIC_TIERS,
    compute_task_id,
    load_known_ids,
    validate_dataset,
)
from tools.assert_categories import assert_all_categories

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NEWS_RATIO_PROFILES: dict[str, dict[str, float]] = {
    "mostly_noise":   {"neutral": 0.90, "single_category": 0.02, "macro_all": 0.03, "sku_level": 0.05},
    "balanced":       {"neutral": 0.65, "single_category": 0.10, "macro_all": 0.05, "sku_level": 0.20},
    "sku_heavy":      {"neutral": 0.50, "single_category": 0.08, "macro_all": 0.07, "sku_level": 0.35},
    "category_focus": {"neutral": 0.55, "single_category": 0.25, "macro_all": 0.12, "sku_level": 0.08},
    "macro_shock":    {"neutral": 0.40, "single_category": 0.10, "macro_all": 0.40, "sku_level": 0.10},
}

_MIDDLE_TIER_NAMES: list[str] = ["micro", "small", "medium"]
_HARD_TIER_NAMES: list[str] = ["standard", "large", "flagship"]

# Per-archetype task count for --archetype all in spread mode (sums to 10 000)
ARCHETYPE_ALLOCATION: dict[str, int] = {
    "still_middle":   3000,
    "dynamic_middle": 3000,
    "still_hard":     2000,
    "dynamic_hard":   2000,
}

# Fixed order in which the remainder is distributed one-each for the equal-split
# (grid-mode) allocation when n is not divisible by 4.
_GRID_ALLOCATION_ORDER: list[str] = [
    "dynamic_hard", "still_hard", "dynamic_middle", "still_middle",
]

_PAPER_DATA_DIR = Path(__file__).resolve().parent.parent / "paper_data"

# ---------------------------------------------------------------------------
# Grid-mode tables (deterministic; ordered by ascending oracle multiple).
# ---------------------------------------------------------------------------

# HARD tables — SPLIT per archetype because the two archetypes have unequal revenue
# engines: only dynamic_hard runs the news/new-product engine (enable_new=True), so it
# earns more per day than still_hard on the same cell. A single shared table pushed
# still_hard's large tier below the 1.30x growth gate. Both tables are ordered by
# ascending oracle multiple; every rung is intended oracle-feasible (worst-of-10 >= 1.30x),
# which the oracle-verify step (tools/verify_task.py) is the source of truth for.
# (tier, abs_c, elasticity_profile_name)
# NOTE: the inline "~X.XXx" per-cell comments are INDICATIVE oracle multiples as-measured on a
# 2026-07 run — guidance for a reader, NOT asserted by any test; re-verify with the oracle if the
# policy/data/tiers change. See docs for the recorded numbers.

# dynamic_hard — the richer archetype; keeps the large tier at every |c| (all verified pass).
_HARD_GRID_DYNAMIC: list[tuple[str, float, str]] = [
    ("large",    0.30, "uniform_030"),       # ~2.09x
    ("large",    0.25, "uniform_025"),       # ~2.37x
    ("large",    0.20, "uniform_baseline"),  # ~2.65x
    ("standard", 0.40, "uniform_040"),       # ~2.92x
    ("standard", 0.35, "all_elastic"),       # ~3.27x
    ("standard", 0.30, "uniform_030"),       # ~3.80x
    ("standard", 0.25, "uniform_025"),       # ~4.20x
    ("standard", 0.20, "uniform_baseline"),  # ~4.5x
]

# still_hard — the leaner archetype; large tier is only feasible at |c|=0.20, so the rest
# is a finer standard-tier |c| ladder (standard clears the gate at every |c|).
_HARD_GRID_STILL: list[tuple[str, float, str]] = [
    ("large",    0.20, "uniform_baseline"),  # ~1.62x  (leanest still cell — highest rent)
    ("standard", 0.40, "uniform_040"),       # ~1.81x
    ("standard", 0.35, "all_elastic"),       # ~2.47x
    ("standard", 0.325, "uniform_0325"),     # ~2.6x
    ("standard", 0.30, "uniform_030"),       # ~2.86x
    ("standard", 0.275, "uniform_0275"),     # ~3.1x
    ("standard", 0.25, "uniform_025"),       # ~3.43x
    ("standard", 0.225, "uniform_0225"),     # ~3.7x
]

# MIDDLE table (identical for dynamic_middle AND still_middle). 7 cells, ordered by ascending
# oracle multiple. The `medium` tier (18k/450) is DROPPED: it structurally needs ~480/day
# margin but N<=10 categories at ~60/cat can't reach it, so the oracle only treads water
# (verified: all 4 medium cells failed the gate). Middle-archetype separation rides the model's
# review-misreading (FM2), which micro/small exercise regardless of oracle margin.
# (tier, abs_c, preset, N). selected_categories = N highest-SKU canonical subset (from disk).
_MIDDLE_GRID: list[tuple[str, float, str, int]] = [
    ("small",  0.30, "uniform_030",      5),  # ~2.2x  (leanest middle cell)
    ("small",  0.25, "uniform_025",      5),  # ~2.5x
    ("small",  0.20, "uniform_baseline", 5),  # ~3.0x
    ("micro",  0.40, "uniform_040",      4),  # ~4.3x
    ("micro",  0.35, "all_elastic",      4),  # ~5.0x
    ("micro",  0.30, "uniform_030",      4),  # ~5.5x
    ("micro",  0.25, "uniform_025",      4),  # ~6.0x
]

# Per-archetype base offset for the deterministic global_random_seed (task_id
# uniqueness only; grid cells are otherwise fully determined by the table).
_GRID_SEED_BASE: dict[str, int] = {
    "dynamic_hard":   100_000,
    "still_hard":     200_000,
    "dynamic_middle": 300_000,
    "still_middle":   400_000,
}

# Data root used to recompute the canonical SKU-count ranking (dynamic == still).
_SIM_DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "dynamic" / "simulate_data" / "15"


def _canonical_category_ranking(sim_root: Path = _SIM_DATA_ROOT) -> list[str]:
    """Underscored categories ranked by on-disk SKU-json count (desc), then name (asc).

    Recomputed from disk so the middle subset is never a hardcoded guess.
    """
    counts: dict[str, int] = {}
    for cat in (c.replace(" ", "_") for c in ALL_CATEGORIES):
        cat_dir = sim_root / cat
        counts[cat] = len(list(cat_dir.glob("*.json"))) if cat_dir.is_dir() else 0
    missing = [c for c, n in counts.items() if n == 0]
    if missing:
        raise RuntimeError(
            f"Cannot rank categories — no SKU JSONs found for {missing} under {sim_root}"
        )
    return sorted(counts, key=lambda c: (-counts[c], c))


def _grid_cells(archetype: str, k: int) -> list[dict]:
    """Deterministically yield the first k grid-table variable-key dicts for archetype.

    Returns dicts in the SAME shape as _sample_variable_keys (including the internal
    _elasticity_profile_name / _economic_tier_name used for task-id computation).
    Raises a clear error if k exceeds the table size (do not silently wrap).
    """
    is_hard = archetype.endswith("_hard")
    hard_grid = _HARD_GRID_DYNAMIC if archetype == "dynamic_hard" else _HARD_GRID_STILL
    grid_len = len(hard_grid) if is_hard else len(_MIDDLE_GRID)
    if k > grid_len:
        raise ValueError(
            f"grid mode exhausted for archetype={archetype!r}: requested {k} cells "
            f"but the table has only {grid_len} rungs. Extend the {'HARD' if is_hard else 'MIDDLE'} "
            f"table with more rungs to generate more."
        )

    seed_base = _GRID_SEED_BASE[archetype]

    # News knobs: balanced ratios held constant across cells.
    news_sample_ratios = dict(NEWS_RATIO_PROFILES["balanced"])
    if archetype == "dynamic_hard":
        news_impact_base_scale, news_daily_count = 0.9, 30
    else:
        # still_hard + both middle archetypes: inert-but-valid news knobs.
        news_impact_base_scale, news_daily_count = 0.8, 20

    if is_hard:
        all_20 = sorted(c.replace(" ", "_") for c in ALL_CATEGORIES)
    else:
        ranking = _canonical_category_ranking()  # compute once (disk glob), reused per middle rung

    cells: list[dict] = []
    for rank in range(1, k + 1):
        if is_hard:
            tier_name, _abs_c, ep_name = hard_grid[rank - 1]
            selected = list(all_20)
        else:
            tier_name, _abs_c, ep_name, n_cats = _MIDDLE_GRID[rank - 1]
            selected = sorted(ranking[:n_cats])

        tier = ECONOMIC_TIERS[tier_name]
        category_effects = dict(ELASTICITY_PROFILES[ep_name])

        cells.append({
            "global_random_seed":       seed_base + rank,
            "selected_categories":      selected,
            "category_effects":         category_effects,
            "initial_funds":            tier["initial_funds"],
            "everyday_rent":            tier["everyday_rent"],
            "inventory_capacity":       tier["inventory_capacity"],
            "review_ratio":             0.02,
            "news_impact_base_scale":   news_impact_base_scale,
            "news_sample_ratios":       news_sample_ratios,
            "news_daily_count":         news_daily_count,
            "news_random_seed":         777,
            "initial_inventory":        {},
            # internal: used for ID computation, stripped before writing
            "_elasticity_profile_name": ep_name,
            "_economic_tier_name":      tier_name,
        })
    return cells

# ---------------------------------------------------------------------------
# Variable key sampler
# ---------------------------------------------------------------------------

def _sample_variable_keys(archetype: str, rng: random.Random) -> dict:
    """Sample all 12 variable keys for one task. Internal _* keys are stripped before writing."""
    is_hard = archetype.endswith("_hard")

    # selected_categories: hard = all 20; middle = N ∈ {4,5,6,7}
    if is_hard:
        selected = sorted(c.replace(" ", "_") for c in ALL_CATEGORIES)
    else:
        n = rng.choices([4, 5, 6, 7], weights=[0.2, 0.4, 0.3, 0.1])[0]
        chosen = rng.sample(ALL_CATEGORIES, n)
        selected = sorted(c.replace(" ", "_") for c in chosen)

    # global_random_seed: exclude 42 (paper default)
    seed = rng.randint(1, 999_999)
    while seed == 42:
        seed = rng.randint(1, 999_999)

    # elasticity profile
    ep_name = rng.choice(list(ELASTICITY_PROFILES.keys()))
    category_effects = dict(ELASTICITY_PROFILES[ep_name])

    # economic tier (band-constrained)
    tier_name = rng.choice(_HARD_TIER_NAMES if is_hard else _MIDDLE_TIER_NAMES)
    tier = ECONOMIC_TIERS[tier_name]

    # review_ratio: 0.02 is official default; full set keeps comparability
    review_ratio = rng.choice([0.01, 0.02, 0.03, 0.05])

    # news_impact_base_scale
    if archetype == "dynamic_hard":
        # enable_new=True: wider, more aggressive range
        news_impact_base_scale = rng.choice([0.3, 0.4, 0.6, 0.9])
    else:
        # enable_new=False (dynamic_middle, still_hard, still_middle): conservative range
        news_impact_base_scale = rng.choice([0.6, 0.8, 1.0])

    # news_sample_ratios profile
    nrp_name = rng.choice(list(NEWS_RATIO_PROFILES.keys()))
    news_sample_ratios = dict(NEWS_RATIO_PROFILES[nrp_name])

    # news_daily_count: 20 is official default
    news_daily_count = rng.choice([10, 15, 20, 25, 30])

    # news_random_seed: exclude 42
    news_seed = rng.randint(1, 99_999)
    while news_seed == 42:
        news_seed = rng.randint(1, 99_999)

    return {
        "global_random_seed":       seed,
        "selected_categories":      selected,
        "category_effects":         category_effects,
        "initial_funds":            tier["initial_funds"],
        "everyday_rent":            tier["everyday_rent"],
        "inventory_capacity":       tier["inventory_capacity"],
        "review_ratio":             review_ratio,
        "news_impact_base_scale":   news_impact_base_scale,
        "news_sample_ratios":       news_sample_ratios,
        "news_daily_count":         news_daily_count,
        "news_random_seed":         news_seed,
        "initial_inventory":        {},
        # internal: used for ID computation, stripped before writing
        "_elasticity_profile_name": ep_name,
        "_economic_tier_name":      tier_name,
    }

# ---------------------------------------------------------------------------
# Core generation loop
# ---------------------------------------------------------------------------

_MAX_RESAMPLE_RETRIES = 1000


def generate(
    archetype: str,
    n: int,
    out: Path,
    rng_seed: int,
    paper_data_dir: Path,
    mode: str = "spread",
) -> list[dict]:
    """
    Generate n unique tasks for the given archetype, writing one {task_id}.json per task.
    Returns a list of manifest entries: [{task_id, archetype}, ...].

    mode == "spread" : the existing random sampler (10k path, default; unchanged).
    mode == "grid"   : a deterministic enumerator of the economic-tier x |c| tables, ordered
                       by ascending oracle multiple. Fully reproducible; no RNG in the task keys.
    """
    out.mkdir(parents=True, exist_ok=True)

    known_ids = load_known_ids(out, paper_data_dir) if paper_data_dir.exists() else set(
        f.stem for f in out.glob("*.json")
        if f.name != "manifest.json" and not f.name.endswith(".verification.json")
    )

    # In grid mode, precompute the deterministic cell list (raises if the table is exhausted).
    grid_cells: list[dict] = _grid_cells(archetype, n) if mode == "grid" else []

    rng = random.Random(rng_seed)
    generated = 0
    manifest_entries: list[dict] = []

    while generated < n:
        if mode == "grid":
            var = grid_cells[generated]
        else:
            var = _sample_variable_keys(archetype, rng)
        task_id = compute_task_id(
            archetype,
            var["global_random_seed"],
            var["selected_categories"],
            var["_elasticity_profile_name"],
            var["_economic_tier_name"],
        )

        # Deduplication: resample up to _MAX_RESAMPLE_RETRIES times on collision.
        # Grid mode is deterministic — a collision means a duplicate cell / prior
        # output already on disk, which resampling cannot fix, so fail loud instead.
        if task_id in known_ids and mode == "grid":
            raise RuntimeError(
                f"grid-mode task_id collision for archetype={archetype!r} "
                f"(task_id={task_id}). The output dir already contains this task, or two "
                f"grid cells collapsed to the same id. Use a clean --out."
            )
        if task_id in known_ids:
            retries = _MAX_RESAMPLE_RETRIES
            while task_id in known_ids and retries > 0:
                var = _sample_variable_keys(archetype, rng)
                task_id = compute_task_id(
                    archetype,
                    var["global_random_seed"],
                    var["selected_categories"],
                    var["_elasticity_profile_name"],
                    var["_economic_tier_name"],
                )
                retries -= 1
            if task_id in known_ids:
                raise RuntimeError(
                    f"Could not find unique task_id after {_MAX_RESAMPLE_RETRIES} retries "
                    f"for archetype={archetype!r}. Task space may be exhausted."
                )

        # Flat env_config: merge archetype locked keys + sampled variable keys (strip internal _ keys)
        env_config = {
            **ARCHETYPES[archetype],
            **{k: v for k, v in var.items() if not k.startswith("_")},
        }

        # Sanity-check env_config before writing (should never fail if sampling is correct)
        errs = validate_dataset(env_config, task_id=task_id)
        if errs:
            raise RuntimeError(
                f"Generated task failed validation (bug in sampler): {errs}\n"
                f"archetype={archetype}, task_id={task_id}"
            )

        # Flat schema: file IS env_config (m1151/m1182)

        out_path = out / f"{task_id}.json"
        out_path.write_text(json.dumps(env_config, indent=2))
        known_ids.add(task_id)
        manifest_entries.append({"task_id": task_id, "archetype": archetype})
        generated += 1

        if generated % 500 == 0 or generated == n:
            print(f"  [{archetype}] {generated}/{n}")

    return manifest_entries

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate RetailBench dataset tasks")
    ap.add_argument(
        "--archetype",
        choices=["dynamic_hard", "dynamic_middle", "still_hard", "still_middle", "all"],
        required=True,
        help="Archetype to generate, or 'all' to generate across all archetypes",
    )
    ap.add_argument("--n", type=int, default=10_000,
                    help="Total tasks to generate (split per ARCHETYPE_ALLOCATION when "
                         "--archetype all in spread mode; equal split in grid mode)")
    ap.add_argument("--out", default="zoro/harness/dataset/",
                    help="Output directory for {task_id}.json files")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for the generator (not task global_random_seed)")
    ap.add_argument("--paper_data_dir", default=None,
                    help="paper_data/ directory for dedup against paper runs (default: auto-detected)")
    ap.add_argument("--mode", choices=["spread", "grid"], default="spread",
                    help="spread = random sampler (10k path, default); "
                         "grid = deterministic enumeration of the economic-tier x |c| tables")
    args = ap.parse_args()

    # Load-assert: all 20 canonical categories must load under BOTH data modes before
    # generating anything (guards the middle canonical subset + hard N=20 selection).
    assert_all_categories()

    out = Path(args.out)
    paper_data_dir = Path(args.paper_data_dir) if args.paper_data_dir else _PAPER_DATA_DIR

    if not paper_data_dir.exists():
        print(
            f"Warning: paper_data_dir not found ({paper_data_dir}) — "
            "deduplication against paper runs disabled"
        )

    all_manifest: list[dict] = []

    if args.archetype == "all":
        if args.mode == "grid":
            # Equal split: floor(n/4) each, remainder distributed one-each in fixed order.
            allocation = {name: args.n // 4 for name in _GRID_ALLOCATION_ORDER}
            for name in _GRID_ALLOCATION_ORDER[: args.n % 4]:
                allocation[name] += 1
        else:
            total_weight = sum(ARCHETYPE_ALLOCATION.values())
            allocation = {}
            remaining = args.n
            names = list(ARCHETYPE_ALLOCATION.keys())
            for name in names[:-1]:
                count = round(args.n * ARCHETYPE_ALLOCATION[name] / total_weight)
                allocation[name] = count
                remaining -= count
            allocation[names[-1]] = remaining  # absorb rounding remainder

        for offset, (arch, count) in enumerate(allocation.items()):
            print(f"\nGenerating {count} tasks for {arch} (mode={args.mode}) ...")
            entries = generate(
                archetype=arch,
                n=count,
                out=out,
                rng_seed=args.seed + offset,   # distinct seed per archetype run
                paper_data_dir=paper_data_dir,
                mode=args.mode,
            )
            all_manifest.extend(entries)
    else:
        print(f"\nGenerating {args.n} tasks for {args.archetype} (mode={args.mode}) ...")
        entries = generate(
            archetype=args.archetype,
            n=args.n,
            out=out,
            rng_seed=args.seed,
            paper_data_dir=paper_data_dir,
            mode=args.mode,
        )
        all_manifest.extend(entries)

    manifest_path = out / "manifest.json"
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    existing.extend(all_manifest)
    manifest_path.write_text(json.dumps(existing, indent=2))
    print(f"\nDone. {len(all_manifest)} tasks written to {out}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
