"""
Normalize category names in UPC metadata:
- Remove underscores in CATEGORY strings.

Reads:
  - data/simulate_data/15/upc.json
Writes:
  - data/simulate_data/15/upc_normalized.json
"""

import json
from pathlib import Path


def normalize_category(cat: str) -> str:
    return cat.replace("_", " ")


def main() -> None:
    src = Path("data/simulate_data/15/upc.json")
    dst = Path("data/simulate_data/15/upc.json")
    dst.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(src.read_text(encoding="utf-8"))
    for row in data:
        cat = row.get("CATEGORY")
        if isinstance(cat, str):
            row["CATEGORY"] = normalize_category(cat)

    dst.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Normalized UPC categories written to {dst}")


if __name__ == "__main__":
    main()
