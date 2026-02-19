"""
统计 data/filter_beta_data（或指定目录）下各门店 big/middle/small SKU 的占比。
判定规则：路径包含 filtered_big_data / filtered_middle_data / filtered_small_data（大小写不敏感）。
去重方式：按 (store_id, sku_id, size_type) 去重，sku_id 取文件名 stem。
"""

import argparse
from pathlib import Path
from typing import Dict, Set, Tuple


def detect_size_type(path: Path) -> str | None:
    for part in path.parts:
        p = part.lower()
        if "filtered_big_data" in p or "_big_" in p or p.endswith("_big"):
            return "big"
        if "filtered_middle_data" in p or "filtered_mid_data" in p or "_middle_" in p or p.endswith("_middle") or "_mid_" in p:
            return "middle"
        if "filtered_small_data" in p or "_small_" in p or p.endswith("_small"):
            return "small"
    return None


def collect_stats(root: Path) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = {}
    seen: Set[Tuple[str, str, str]] = set()

    for store_dir in root.iterdir():
        if not store_dir.is_dir():
            continue
        store_id = store_dir.name
        stats.setdefault(store_id, {"big": 0, "middle": 0, "small": 0})

        for file_path in store_dir.rglob("*"):
            if not file_path.is_file():
                continue
            size_type = detect_size_type(file_path)
            if size_type is None:
                continue

            sku_id = file_path.stem.split("_")[0]  # 取文件名去扩展名，遇到 daily/suppliers 也能匹配
            key = (store_id, sku_id, size_type)
            if key in seen:
                continue
            seen.add(key)
            stats[store_id][size_type] += 1

    return stats


def print_stats(stats: Dict[str, Dict[str, int]]) -> None:
    for store_id, counts in sorted(stats.items()):
        total = sum(counts.values())
        if total == 0:
            print(f"[STORE {store_id}] 无匹配 SKU")
            continue
        def pct(v: int) -> float:
            return (v / total * 100) if total else 0.0
        print(f"[STORE {store_id}] total={total} | big={counts['big']} ({pct(counts['big']):.1f}%) | "
              f"middle={counts['middle']} ({pct(counts['middle']):.1f}%) | "
              f"small={counts['small']} ({pct(counts['small']):.1f}%)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计 big/middle/small SKU 占比")
    parser.add_argument("--input-dir", default="data/filter_beta_data", help="输入目录，包含各 store 子目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.input_dir)
    if not root.exists():
        raise FileNotFoundError(f"输入目录不存在: {root}")
    stats = collect_stats(root)
    print_stats(stats)


if __name__ == "__main__":
    main()
