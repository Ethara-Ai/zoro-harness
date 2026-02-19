"""Generate initial ratings for SKUs under a folder using category begin points."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict

from util.file import load_json, dump_json

# --- 固定配置 ---
# 输入：包含各品类 sku.csv 的目录（形如 store/category/sku.csv）
INPUT_DIR = Path("/Users/linghuazhang/Desktop/Project/RetailBench/data/simulate_data")
# 品类 → begin_point 配置
BEGIN_POINT_PATH = Path("data/review/begin_effect_point.json")
# 输出：评分文件（store -> category -> sku -> rating）
OUTPUT_PATH = Path("data/review/simulated_ratings.json")
# 随机种子
SEED = 42


def scan_skus(root: Path):
    """Yield tuples of (store, category, sku_id)."""
    for csv_path in root.rglob("*.csv"):
        rel = csv_path.relative_to(root)
        parts = rel.parts
        if len(parts) < 3:
            # 期望 store/category/sku.csv
            continue
        store, category, sku_file = parts[0], parts[1], parts[-1]
        sku_id = Path(sku_file).stem
        yield store, category, sku_id


def generate_ratings(begin_points: Dict[str, float]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Sample rating per SKU using category-specific ranges."""
    random.seed(SEED)
    ratings: Dict[str, Dict[str, Dict[str, float]]] = {}

    for store, category, sku_id in scan_skus(INPUT_DIR):
        category = category.replace("_", " ")
        begin = begin_points.get(category)
        if begin is None:
            print(f"[WARN] Missing begin_point for category: {category}, skip {store}/{category}/{sku_id}")
            continue

        low = max(begin + 0.2, 4.4)
        high = min(begin + 1.2, 4.8)
        rating = random.uniform(low, high)
        rating = max(0.0, min(5.0, rating))

        ratings.setdefault(store, {}).setdefault(category, {})[sku_id] = round(rating, 3)

    return ratings


def main() -> None:
    begin_points = load_json(str(BEGIN_POINT_PATH))
    if not isinstance(begin_points, dict):
        raise SystemExit("begin_effect_point.json 应为 {category: begin_point} 的映射")

    ratings = generate_ratings({k: float(v) for k, v in begin_points.items()})
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump_json(ratings, str(OUTPUT_PATH))
    print(f"Generated {len(ratings)} ratings → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
