"""
Normalize category names in review data:
- Remove underscores in category strings (e.g., "Frozen_Entrees" -> "Frozen Entrees")

Reads:
  - data/review/all_category_reviews.jsonl
Writes:
  - data/review/all_category_reviews_normalized.jsonl
"""

from pathlib import Path
import json


def normalize_category(cat: str) -> str:
    return cat.replace("_", " ")


def main() -> None:
    src = Path("/Users/linghuazhang/Desktop/Project/RetailBench/data/review/all_category_reviews_mp.jsonl")
    dst = Path("data/review/all_category_reviews.jsonl")
    dst.parent.mkdir(parents=True, exist_ok=True)

    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cat = obj.get("CATEGORY")
            if isinstance(cat, str):
                obj["CATEGORY"] = normalize_category(cat)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Normalized categories written to {dst}")


if __name__ == "__main__":
    main()
