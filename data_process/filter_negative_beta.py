"""
找出各门店 sku_model_parameter.json 中 beta 为负数的 SKU，
并将这些 SKU 的数据文件（保持原目录层级）复制到输出目录。
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Set


def load_params(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def match_sku_file(path: Path, sku_ids: Set[str]) -> bool:
    stem = path.stem
    for sku in sku_ids:
        if stem == sku or stem.startswith(f"{sku}_"):
            return True
    return False


def copy_negative_sku_files(store_dir: Path, sku_ids: Set[str], output_dir: Path) -> int:
    """复制匹配 SKU 的文件，保持相对路径，返回复制数量。"""
    copied = 0
    for file_path in store_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.name == "sku_model_parameter.json":
            continue
        if match_sku_file(file_path, sku_ids):
            rel = file_path.relative_to(store_dir)
            dest = output_dir / store_dir.name / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest)
            copied += 1
    return copied


def filter_and_copy(store_root: Path, output_dir: Path) -> None:
    stores = [d for d in store_root.iterdir() if d.is_dir()]
    total_skus = 0
    total_negative = 0
    total_copied = 0

    for store_dir in stores:
        param_file = store_dir / "sku_model_parameter.json"
        params = load_params(param_file)
        if not params:
            continue

        negative = {
            sku_id: p for sku_id, p in params.items()
            if p.get("beta0", 0) < 0
        }

        total_skus += len(params)
        total_negative += len(negative)

        if negative:
            # 保存参数子集（两份：集中放一份 + 保留目录层级一份）
            # out_param = output_dir / "params" / f"store_{store_dir.name}_negative_beta.json"
            # save_json(negative, out_param)
            out_param_store = output_dir / store_dir.name / "sku_model_parameter.json"
            save_json(negative, out_param_store)

            # 复制数据文件，保持层级
            copied = copy_negative_sku_files(store_dir, set(negative.keys()), output_dir)
            total_copied += copied
            print(f"[STORE {store_dir.name}] 负 beta SKU: {len(negative)} / {len(params)} | 复制文件 {copied}")
        else:
            print(f"[STORE {store_dir.name}] 无负 beta SKU （共 {len(params)} 条）")

    print(f"\n汇总：负 beta SKU {total_negative} / {total_skus}，复制文件总数 {total_copied}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy data files of SKUs with negative beta, preserving store directory structure.")
    parser.add_argument("--store-root", default="/Users/linghuazhang/Desktop/RetailBenchRubbish/filtered_source_data_by_category", help="门店参数所在根目录")
    parser.add_argument("--output-dir", default="/Users/linghuazhang/Desktop/RetailBenchRubbish/filtered_source_data_by_category/negative_beta_data", help="输出目录（保持原层级）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store_root = Path(args.store_root)
    output_dir = Path(args.output_dir)

    if not store_root.exists():
        raise FileNotFoundError(f"store_root 不存在: {store_root}")

    filter_and_copy(store_root, output_dir)


if __name__ == "__main__":
    main()
