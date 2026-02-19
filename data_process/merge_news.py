"""
Unify neutral / generated / SKU news into one JSONL with consistent fields.

Default inputs (store 15):
- data/simulate_data/15/neutral_news.jsonl
- data/simulate_data/15/generated_news.jsonl
- data/simulate_data/15/generated_sku_news.jsonl

Usage:
    python script/merge_news.py \
        --output data/simulate_data/15/news_merged.jsonl

You can override inputs with repeated --input flags.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List
from uuid import uuid4


STANDARD_FIELDS = [
    "mode",
    "title",
    "content",
    "impact_scope",
    "impact_categories",
    "impact_strength",
    "impact_duration_days",
    "impact_direction",
    "impact_factor",
    "target_category",
    "target_sku_upc",
    "sku_meta",
    "source_file",
]


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_row(row: Dict[str, Any], source_file: str) -> Dict[str, Any]:
    """Return a normalized record with standard fields."""
    mode = (
        row.get("MODE")
        or row.get("mode")
        or ("sku_level" if row.get("SKU_UPC") else "single_category" if row.get("TARGET_CATEGORY") else "neutral")
    )

    title = row.get("TITLE") or row.get("title") or ""
    content = row.get("CONTENT") or row.get("content") or ""

    impact_scope = row.get("IMPACT_SCOPE") or row.get("impact_scope")
    impact_categories = row.get("IMPACT_CATEGORIES") or row.get("impact_categories") or []
    impact_strength = row.get("IMPACT_STRENGTH") or row.get("impact_strength")
    impact_duration_days = row.get("IMPACT_DURATION_DAYS") or row.get("impact_duration_days")
    impact_direction = row.get("IMPACT_DIRECTION") or row.get("impact_direction")
    impact_factor = row.get("IMPACT_FACTOR") or row.get("impact_factor")

    target_category = row.get("TARGET_CATEGORY") or row.get("target_category") or row.get("SKU_CATEGORY")
    target_sku_upc = row.get("TARGET_SKU_UPC") or row.get("target_sku_upc") or row.get("SKU_UPC")
    sku_meta = row.get("SKU_META") or row.get("sku_meta")

    # Neutral news usually lacks impact info; mark scope explicitly.
    if not impact_scope:
        impact_scope = "neutral"

    normalized = {
        "id": uuid4().hex,
        "mode": mode,
        "title": title,
        "content": content,
        "impact_scope": impact_scope,
        "impact_categories": impact_categories,
        "impact_strength": impact_strength,
        "impact_duration_days": impact_duration_days,
        "impact_direction": impact_direction,
        "impact_factor": impact_factor,
        "target_category": None if target_sku_upc else target_category,
        "target_sku_upc": target_sku_upc,
        "sku_meta": sku_meta,
        "source_file": source_file,
    }
    return normalized

from collections import Counter
from typing import List, Dict, Any

def stat_mode_distribution(merged: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    统计不同 mode 的分布数量
    """
    return Counter(item["mode"] for item in merged)
def merge_files(input_files: List[Path]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for p in input_files:
        for row in iter_jsonl(p):
            merged.append(normalize_row(row, source_file=str(p)))

    merged = [item for item in merged if item['mode'] != 'local_event']
   
    
    return merged


def write_jsonl(rows: List[Dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge news files into a unified JSONL format.")
    parser.add_argument(
        "--input",
        action="append",
        help="Input JSONL file path. Can repeat. Defaults to the three known news files under store 15.",
    )
    parser.add_argument(
        "--output",
        default="data/simulate_data/15/news_merged.jsonl",
        help="Output JSONL path.",
    )
    args = parser.parse_args()

    default_inputs = [
        Path("data/simulate_data/15/neutral_news.jsonl"),
        Path("data/simulate_data/15/single_category.jsonl"),
        Path("data/simulate_data/15/sku_new.jsonl"),
    ]
    input_files = [Path(p) for p in args.input] if args.input else default_inputs

    missing = [str(p) for p in input_files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input files: {', '.join(missing)}")

    merged = merge_files(input_files)
    mode_dist = stat_mode_distribution(merged)

    for mode, cnt in mode_dist.items():
        print(f"{mode}: {cnt}")
    
    output_path = Path(args.output)
    write_jsonl(merged, output_path)
    print(f"Wrote {len(merged)} rows to {output_path}")


if __name__ == "__main__":
    main()
