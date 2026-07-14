"""
generate_datasets.py — Batch generator for RetailBench dataset tasks.

Usage:
    python zoro/harness/generator/generate_datasets.py \
        --archetype {dynamic_hard|dynamic_middle|still_hard|still_middle|all} \
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

# Per-archetype task count for --archetype all (sums to 10 000)
ARCHETYPE_ALLOCATION: dict[str, int] = {
    "still_middle":   3000,
    "dynamic_middle": 3000,
    "still_hard":     2000,
    "dynamic_hard":   2000,
}

_PAPER_DATA_DIR = Path(__file__).resolve().parent.parent / "paper_data"

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
) -> list[dict]:
    """
    Generate n unique tasks for the given archetype, writing one {task_id}.json per task.
    Returns a list of manifest entries: [{task_id, archetype}, ...].
    """
    out.mkdir(parents=True, exist_ok=True)

    known_ids = load_known_ids(out, paper_data_dir) if paper_data_dir.exists() else set(
        f.stem for f in out.glob("*.json") if f.name != "manifest.json"
    )

    rng = random.Random(rng_seed)
    generated = 0
    manifest_entries: list[dict] = []

    while generated < n:
        var = _sample_variable_keys(archetype, rng)
        task_id = compute_task_id(
            archetype,
            var["global_random_seed"],
            var["selected_categories"],
            var["_elasticity_profile_name"],
            var["_economic_tier_name"],
        )

        # Deduplication: resample up to _MAX_RESAMPLE_RETRIES times on collision
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
        help="Archetype to generate, or 'all' to generate across all archetypes per allocation table",
    )
    ap.add_argument("--n", type=int, default=10_000,
                    help="Total tasks to generate (split per ARCHETYPE_ALLOCATION when --archetype all)")
    ap.add_argument("--out", default="zoro/harness/dataset/",
                    help="Output directory for {task_id}.json files")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for the generator (not task global_random_seed)")
    ap.add_argument("--paper_data_dir", default=None,
                    help="paper_data/ directory for dedup against paper runs (default: auto-detected)")
    args = ap.parse_args()

    out = Path(args.out)
    paper_data_dir = Path(args.paper_data_dir) if args.paper_data_dir else _PAPER_DATA_DIR

    if not paper_data_dir.exists():
        print(
            f"Warning: paper_data_dir not found ({paper_data_dir}) — "
            "deduplication against paper runs disabled"
        )

    all_manifest: list[dict] = []

    if args.archetype == "all":
        total_weight = sum(ARCHETYPE_ALLOCATION.values())
        allocation: dict[str, int] = {}
        remaining = args.n
        names = list(ARCHETYPE_ALLOCATION.keys())
        for name in names[:-1]:
            count = round(args.n * ARCHETYPE_ALLOCATION[name] / total_weight)
            allocation[name] = count
            remaining -= count
        allocation[names[-1]] = remaining  # absorb rounding remainder

        for offset, (arch, count) in enumerate(allocation.items()):
            print(f"\nGenerating {count} tasks for {arch} ...")
            entries = generate(
                archetype=arch,
                n=count,
                out=out,
                rng_seed=args.seed + offset,   # distinct seed per archetype run
                paper_data_dir=paper_data_dir,
            )
            all_manifest.extend(entries)
    else:
        print(f"\nGenerating {args.n} tasks for {args.archetype} ...")
        entries = generate(
            archetype=args.archetype,
            n=args.n,
            out=out,
            rng_seed=args.seed,
            paper_data_dir=paper_data_dir,
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
