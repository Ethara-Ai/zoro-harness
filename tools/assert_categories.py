#!/usr/bin/env python3
"""
assert_categories.py — Build-time load-assert for the dataset generator (both modes).

Asserts that all 20 canonical categories load with real, parseable SKU JSONs under BOTH
  data/dynamic/simulate_data/15  AND  data/still/simulate_data/15
Each category dir must contain >= 1 *.json that parses. Fails loud (RuntimeError / exit 1)
if not 20/20 under either data mode. Called at generator startup so neither the spread
(random) nor the grid (deterministic) path can silently emit tasks that reference a
missing/unparseable category. Does NOT hardcode 19 (Fabric_Softeners N=20 is real).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from generator.validators import ALL_CATEGORIES  # noqa: E402

DATA_ROOTS = {
    "dynamic": REPO_ROOT / "data" / "dynamic" / "simulate_data" / "15",
    "still":   REPO_ROOT / "data" / "still" / "simulate_data" / "15",
}

EXPECTED_UNDERSCORED = sorted(c.replace(" ", "_") for c in ALL_CATEGORIES)


def _category_ok(cat_dir: Path) -> bool:
    if not cat_dir.is_dir():
        return False
    jsons = sorted(cat_dir.glob("*.json"))
    if not jsons:
        return False
    for jp in jsons:
        try:
            json.loads(jp.read_text())
            return True  # at least one parseable SKU JSON
        except Exception:
            continue
    return False


def assert_all_categories() -> dict:
    """Return per-mode {category: sku_json_count}. Raise RuntimeError if any mode != 20/20."""
    summary: dict[str, dict[str, int]] = {}
    for mode, root in DATA_ROOTS.items():
        if not root.is_dir():
            raise RuntimeError(f"load-assert FAIL: data root missing for {mode}: {root}")
        counts: dict[str, int] = {}
        loaded = []
        for cat in EXPECTED_UNDERSCORED:
            cat_dir = root / cat
            if _category_ok(cat_dir):
                counts[cat] = len(list(cat_dir.glob("*.json")))
                loaded.append(cat)
        missing = [c for c in EXPECTED_UNDERSCORED if c not in loaded]
        if len(loaded) != 20:
            raise RuntimeError(
                f"load-assert FAIL [{mode}]: {len(loaded)}/20 categories loaded; "
                f"missing/unparseable: {missing}"
            )
        summary[mode] = counts
    return summary


def main() -> None:
    summary = assert_all_categories()
    for mode, counts in summary.items():
        print(f"[{mode}] 20/20 categories OK (SKU json counts): "
              f"min={min(counts.values())} max={max(counts.values())}")
    print("load-assert PASS: 20/20 under both dynamic and still")


if __name__ == "__main__":
    main()
