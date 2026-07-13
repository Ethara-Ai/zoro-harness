"""
validators.py — Schema, semantic, and deduplication validation for generated dataset tasks.

Gates:
  1 — Key schema: 27 expected top-level env_config keys, correct sub-dict structure.
  2 — Category format: spaced keys in category_effects, underscored in selected_categories.
  3 — News ratios: news_sample_ratios values sum to 1.0 (±0.001).
  4 — Paper-run exclusion: task_id must not match any paper_data config fingerprint.
  5 — Survival floor: initial_funds / everyday_rent >= 30.
  6 — Capacity ratio: 0.5 <= inventory_capacity / initial_funds <= 2.0.
  7 — category_effects completeness: all 20 spaced keys present.
  8 — Economic band: middle archetypes initial_funds <= 20000; hard >= 25000.

Also exports shared constants (ALL_CATEGORIES, ELASTICITY_PROFILES, ECONOMIC_TIERS, NAMESPACE,
compute_task_id, load_known_ids, infer_archetype) consumed by generate_datasets.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # zoro/harness/ → util.* importable

from util.default_config import create_dynamic_hard_config

# ---------------------------------------------------------------------------
# Shared constants (imported by generate_datasets.py)
# ---------------------------------------------------------------------------

NAMESPACE = uuid.UUID("c7e0fde0-0000-5000-8000-000000000000")

ALL_CATEGORIES: list[str] = [
    "Bathroom Tissues", "Beer", "Bottled Juices", "Canned Soup", "Canned Tuna",
    "Cereals", "Cheeses", "Cigarettes", "Cookies", "Crackers", "Dish Detergent",
    "Fabric Softeners", "Front end candies", "Frozen Entrees", "Frozen Juices",
    "Oatmeal", "Paper Towels", "Snack Crackers", "Soft Drinks", "Toothpastes",
]

CATEGORY_NAMES_SPACED: frozenset[str] = frozenset(ALL_CATEGORIES)
CATEGORY_NAMES_UNDERSCORED: frozenset[str] = frozenset(c.replace(" ", "_") for c in ALL_CATEGORIES)

ELASTICITY_PROFILES: dict[str, dict[str, float]] = {
    "uniform_baseline": {c: -0.20 for c in ALL_CATEGORIES},
    "staples_inelastic": {
        **{c: -0.20 for c in ALL_CATEGORIES},
        "Cigarettes": -0.05,
        "Canned Soup": -0.12,
        "Oatmeal": -0.10,
        "Canned Tuna": -0.13,
    },
    "treats_elastic": {
        **{c: -0.20 for c in ALL_CATEGORIES},
        "Beer": -0.42,
        "Soft Drinks": -0.38,
        "Cookies": -0.40,
        "Snack Crackers": -0.35,
        "Front end candies": -0.44,
    },
    "mixed_volatile": {
        **{c: -0.20 for c in ALL_CATEGORIES},
        "Beer": -0.45,
        "Cigarettes": -0.05,
        "Frozen Entrees": -0.30,
        "Cereals": -0.12,
        "Soft Drinks": -0.37,
        "Oatmeal": -0.10,
    },
    "all_elastic": {c: -0.35 for c in ALL_CATEGORIES},
}

ECONOMIC_TIERS: dict[str, dict[str, int]] = {
    # middle-band tiers
    "micro":    {"initial_funds": 5_000,  "everyday_rent": 120,   "inventory_capacity": 6_000},
    "small":    {"initial_funds": 10_000, "everyday_rent": 250,   "inventory_capacity": 10_000},
    "medium":   {"initial_funds": 18_000, "everyday_rent": 450,   "inventory_capacity": 16_000},
    # hard-band tiers
    "standard": {"initial_funds": 30_000, "everyday_rent": 700,   "inventory_capacity": 25_000},
    "large":    {"initial_funds": 50_000, "everyday_rent": 1_000, "inventory_capacity": 40_000},
    "flagship": {"initial_funds": 80_000, "everyday_rent": 1_600, "inventory_capacity": 65_000},
}

# ---------------------------------------------------------------------------
# Expected env_config key set (ground truth from factory function)
# ---------------------------------------------------------------------------

EXPECTED_KEYS: frozenset[str] = frozenset(create_dynamic_hard_config().keys())

_NEWS_SAMPLE_RATIO_KEYS: frozenset[str] = frozenset({"neutral", "single_category", "macro_all", "sku_level"})
_NEWS_IMPACT_WEIGHT_KEYS: frozenset[str] = frozenset({"neutral", "macro_all", "single_category", "sku_level"})

# Auto-detected paper_data directory (zoro/harness/paper_data/)
_HARNESS_DIR = Path(__file__).resolve().parent.parent   # zoro/harness/
PAPER_DATA_DIR: Path = _HARNESS_DIR / "paper_data"

# ---------------------------------------------------------------------------
# UUID / identity helpers (also used by generate_datasets.py)
# ---------------------------------------------------------------------------

def compute_task_id(
    archetype: str,
    global_random_seed: int,
    selected_categories: list[str],
    elasticity_profile: str,
    economic_tier: str,
) -> str:
    content = json.dumps(
        {
            "archetype": archetype,
            "seed": global_random_seed,
            "categories": sorted(selected_categories),
            "elasticity_profile": elasticity_profile,
            "economic_tier": economic_tier,
        },
        sort_keys=True,
    )
    return str(uuid.uuid5(NAMESPACE, content))


def infer_archetype(cfg: dict) -> str:
    data_mode = "dynamic" if "dynamic" in cfg.get("data_dir", "") else "still"
    difficulty = "hard" if len(cfg.get("selected_categories", [])) > 10 else "middle"
    return f"{data_mode}_{difficulty}"


def match_elasticity_profile(effects: dict) -> str:
    for name, profile in ELASTICITY_PROFILES.items():
        if all(abs(effects.get(c, -0.2) - profile[c]) < 0.001 for c in ALL_CATEGORIES):
            return name
    return "custom"


def match_economic_tier(cfg: dict) -> str:
    for name, tier in ECONOMIC_TIERS.items():
        if (
            tier["initial_funds"] == cfg.get("initial_funds")
            and tier["everyday_rent"] == cfg.get("everyday_rent")
        ):
            return name
    return "custom"


def _paper_run_ids_from_dir(paper_data_dir: Path) -> set[str]:
    """Compute UUID5 fingerprints for every config.json under paper_data_dir."""
    known: set[str] = set()
    for cfg_path in paper_data_dir.glob("*/*/run_env_*/config.json"):
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            continue
        archetype = infer_archetype(cfg)
        ep_name = match_elasticity_profile(cfg.get("category_effects", {}))
        tier_name = match_economic_tier(cfg)
        known.add(
            compute_task_id(
                archetype,
                cfg.get("global_random_seed"),
                cfg.get("selected_categories", []),
                ep_name,
                tier_name,
            )
        )
    return known


def load_known_ids(dataset_dir: Path, paper_data_dir: Path) -> set[str]:
    """Return all known task IDs: previously generated tasks + all paper-run fingerprints."""
    known: set[str] = set()

    # Previously generated tasks (filename == task_id)
    for f in dataset_dir.glob("*.json"):
        if f.name != "manifest.json":
            known.add(f.stem)

    # Live scan of paper_data config.json files
    known.update(_paper_run_ids_from_dir(paper_data_dir))
    return known


# ---------------------------------------------------------------------------
# Validation gates
# ---------------------------------------------------------------------------

def validate_schema(ds: dict) -> list[str]:
    errors: list[str] = []
    cfg = ds  # flat schema: ds IS the env_config

    # Gate 1b: env_config key set
    cfg_keys = set(cfg.keys())
    missing_cfg = EXPECTED_KEYS - cfg_keys
    extra_cfg = cfg_keys - EXPECTED_KEYS
    if missing_cfg:
        errors.append(f"Gate1: env_config missing keys: {sorted(missing_cfg)}")
    if extra_cfg:
        errors.append(f"Gate1: env_config extra keys: {sorted(extra_cfg)}")

    # Gate 1d: news_sample_ratios sub-dict
    nsr = cfg.get("news_sample_ratios", {})
    missing_nsr = _NEWS_SAMPLE_RATIO_KEYS - set(nsr.keys())
    if missing_nsr:
        errors.append(f"Gate1: news_sample_ratios missing keys: {sorted(missing_nsr)}")

    # Gate 1e: news_impact_mode_weights sub-dict
    niw = cfg.get("news_impact_mode_weights", {})
    missing_niw = _NEWS_IMPACT_WEIGHT_KEYS - set(niw.keys())
    if missing_niw:
        errors.append(f"Gate1: news_impact_mode_weights missing keys: {sorted(missing_niw)}")

    # Gate 2a: category_effects keys must use spaced names
    ce = cfg.get("category_effects", {})
    bad_ce = set(ce.keys()) - CATEGORY_NAMES_SPACED
    if bad_ce:
        errors.append(f"Gate2: category_effects has non-spaced keys: {sorted(bad_ce)}")

    # Gate 2b: selected_categories must use underscored names
    sc = cfg.get("selected_categories", [])
    bad_sc = set(sc) - CATEGORY_NAMES_UNDERSCORED
    if bad_sc:
        errors.append(f"Gate2: selected_categories has non-underscored names: {sorted(bad_sc)}")

    # Gate 3: news_sample_ratios values must sum to 1.0 ±0.001
    if nsr:
        ratio_sum = sum(nsr.values())
        if abs(ratio_sum - 1.0) > 0.001:
            errors.append(
                f"Gate3: news_sample_ratios sum={ratio_sum:.6f} (must be 1.0 ±0.001)"
            )

    return errors


def validate_semantic(ds: dict) -> list[str]:
    errors: list[str] = []
    cfg = ds  # flat schema
    initial_funds: float = cfg.get("initial_funds", 0)
    everyday_rent: float = cfg.get("everyday_rent", 1)
    inventory_capacity: float = cfg.get("inventory_capacity", 0)

    # Gate 5: survival floor — at least 30 days runway
    if everyday_rent > 0:
        runway = initial_funds / everyday_rent
        if runway < 30:
            errors.append(
                f"Gate5: survival floor violation — "
                f"initial_funds({initial_funds}) / everyday_rent({everyday_rent}) "
                f"= {runway:.2f} < 30"
            )

    # Gate 6: inventory capacity ratio in [0.5, 2.0]
    if initial_funds > 0:
        cap_ratio = inventory_capacity / initial_funds
        if not (0.5 <= cap_ratio <= 2.0):
            errors.append(
                f"Gate6: capacity ratio={cap_ratio:.4f} "
                f"(inventory_capacity={inventory_capacity} / initial_funds={initial_funds}) "
                f"must be in [0.5, 2.0]"
            )

    # Gate 7: category_effects must have all 20 spaced keys
    ce = cfg.get("category_effects", {})
    missing_ce = CATEGORY_NAMES_SPACED - set(ce.keys())
    if missing_ce:
        errors.append(
            f"Gate7: category_effects missing {len(missing_ce)} keys: {sorted(missing_ce)}"
        )

    return errors


def validate_archetype_band(ds: dict) -> list[str]:
    errors: list[str] = []
    cfg = ds  # flat schema
    initial_funds: float = cfg.get("initial_funds", 0)
    archetype = infer_archetype(cfg)

    # Gate 8: economic band consistency
    if "middle" in archetype and initial_funds > 20_000:
        errors.append(
            f"Gate8: middle archetype ({archetype}) but initial_funds={initial_funds} > 20000"
        )
    elif "hard" in archetype and initial_funds < 25_000:
        errors.append(
            f"Gate8: hard archetype ({archetype}) but initial_funds={initial_funds} < 25000"
        )

    return errors


def validate_dataset(ds: dict, paper_run_ids: set[str] | None = None, task_id: str | None = None) -> list[str]:
    """Run all gates against a flat config dict. Returns list of error strings (empty = pass)."""
    errors: list[str] = []
    errors.extend(validate_schema(ds))
    errors.extend(validate_semantic(ds))
    errors.extend(validate_archetype_band(ds))

    # Gate 4: paper-run exclusion
    if paper_run_ids is not None and task_id is not None:
        if task_id in paper_run_ids:
            errors.append(f"Gate4: task_id {task_id!r} matches a paper-run config fingerprint")

    return errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Validate generated dataset.json files")
    ap.add_argument("--datasets", required=True,
                    help="Directory containing {task_id}.json files (and optional manifest.json)")
    ap.add_argument("--strict", action="store_true",
                    help="Exit 1 if any task fails validation")
    ap.add_argument("--paper_data_dir", default=None,
                    help="paper_data/ directory (default: auto-detected from repo root)")
    args = ap.parse_args()

    datasets_dir = Path(args.datasets)
    if not datasets_dir.is_dir():
        print(f"ERROR: --datasets is not a directory: {datasets_dir}")
        sys.exit(2)

    paper_data_dir = Path(args.paper_data_dir) if args.paper_data_dir else PAPER_DATA_DIR
    paper_run_ids: set[str] | None = None
    if paper_data_dir.exists():
        paper_run_ids = _paper_run_ids_from_dir(paper_data_dir)
        print(f"Loaded {len(paper_run_ids)} paper-run fingerprints from {paper_data_dir}")
    else:
        print(f"Warning: paper_data_dir not found ({paper_data_dir}) — Gate 4 skipped")

    task_files = sorted(p for p in datasets_dir.glob("*.json") if p.name != "manifest.json")
    if not task_files:
        print("No dataset files found.")
        sys.exit(0)

    n_total = len(task_files)
    n_failed = 0
    all_errors: dict[str, list[str]] = {}

    for path in task_files:
        try:
            ds = json.loads(path.read_text())
        except Exception as exc:
            all_errors[path.name] = [f"JSON parse error: {exc}"]
            n_failed += 1
            continue

        errs = validate_dataset(ds, paper_run_ids, task_id=path.stem)
        if errs:
            all_errors[path.stem] = errs
            n_failed += 1

    if all_errors:
        shown = list(all_errors.items())[:50]
        for task_id, errs in shown:
            print(f"\nFAIL {task_id}:")
            for e in errs:
                print(f"  {e}")
        if len(all_errors) > 50:
            print(f"\n  ... and {len(all_errors) - 50} more failures (first 50 shown)")

    print(f"\nValidation complete: {n_total - n_failed}/{n_total} passed")

    if args.strict and n_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
