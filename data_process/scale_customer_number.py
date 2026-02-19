from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO_ROOT / "data" / "customer_number"


def scale_custcoun(obj: Any, factor: float) -> Tuple[Any, bool]:
    """Scale custcoun if numeric; return (maybe-updated, changed_flag)."""
    if isinstance(obj, dict) and "custcoun" in obj:
        val = obj.get("custcoun")
        if isinstance(val, (int, float)):
            obj = dict(obj)
            obj["custcoun"] = val * factor
            return obj, True
    return obj, False


def process_file(path: Path, factor: float) -> int:
    """Multiply custcoun by factor for all entries in a JSON array."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[SKIP] {path} (failed to load: {exc})")
        return 0

    if not isinstance(data, list):
        print(f"[SKIP] {path} (unexpected root type {type(data).__name__})")
        return 0

    changed = 0
    new_items = []
    for item in data:
        new_item, is_changed = scale_custcoun(item, factor)
        new_items.append(new_item)
        changed += int(is_changed)

    if changed:
        path.write_text(json.dumps(new_items, ensure_ascii=False, indent=2), encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Scale custcoun in customer_number JSON files.")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_DIR,
        help="Root directory containing customer_number store folders.",
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=7.0,
        help="Multiply custcoun by this factor.",
    )
    args = parser.parse_args()

    root = args.root
    factor = args.factor
    if not root.exists():
        raise SystemExit(f"Root directory not found: {root}")

    total_files = 0
    total_changed = 0
    for json_file in sorted(root.rglob("*.json")):
        total_files += 1
        changed = process_file(json_file, factor)
        total_changed += changed
        if changed:
            print(f"[OK] {json_file} (updated {changed})")
        else:
            print(f"[OK] {json_file} (no numeric custcoun)")

    print(f"Done. Files scanned: {total_files}, custcoun updated: {total_changed}")


if __name__ == "__main__":
    main()
