"""
根据价格-销量模拟结果，过滤掉销量始终为 0 的 SKU。

输入：
    - data/negative_beta_data（或自定义 --input-dir）
    其中每个门店目录包含 sku_model_parameter.json 以及 SKU 相关文件。

过程：
    1) 调用 sku.simulate_all_store_skus 在指定价格区间内模拟销量。
    2) 对于销量全为 0 的 SKU，将其剔除。
    3) 保留的 SKU 文件按原目录层级复制到输出目录，并写出过滤后的 sku_model_parameter.json。

输出：
    - 默认 data/filter_beta_data（可通过 --output-dir 指定）
    - 额外保存每个门店过滤后的参数到 output_dir/params/store_<id>_filtered.json
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Set

from sku import simulate_all_store_skus


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


def copy_filtered_files(store_dir: Path, keep_skus: Set[str], output_dir: Path) -> int:
    """复制保留 SKU 的文件，保持相对路径，返回复制数量。"""
    copied = 0
    for file_path in store_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.name == "sku_model_parameter.json":
            # 参数文件后面单独写
            continue
        if match_sku_file(file_path, keep_skus):
            rel = file_path.relative_to(store_dir)
            dest = output_dir / store_dir.name / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest)
            copied += 1
    return copied


def filter_zero_sales(
    input_dir: Path,
    output_dir: Path,
    price_min: float,
    price_max: float,
    price_step: float,
    customer_count: int,
) -> None:
    # 先模拟销量
    store_sales = simulate_all_store_skus(
        store_root=str(input_dir),
        price_min=price_min,
        price_max=price_max,
        price_step=price_step,
        customer_count=customer_count,
        plot_dir=str(output_dir / "plots"),
        batch_size=20,
    )

    total_skus = 0
    total_filtered = 0
    total_copied = 0

    for store_dir in input_dir.iterdir():
        if not store_dir.is_dir():
            continue
        store_id = store_dir.name
        params_path = store_dir / "sku_model_parameter.json"
        params = load_params(params_path)
        if not params:
            continue

        sim_sales = store_sales.get(store_id, {})
        keep_skus = {
            sku_id for sku_id, sales in sim_sales.items()
            if any(s > 0 for s in sales)
        }
        total_skus += len(params)
        total_filtered += len(params) - len(keep_skus)

        filtered_params = {sku_id: params[sku_id] for sku_id in keep_skus if sku_id in params}

        # 保存过滤后的参数
        out_param_store = output_dir / store_id / "sku_model_parameter.json"
        out_param_flat = output_dir / "params" / f"store_{store_id}_filtered.json"
        save_json(filtered_params, out_param_store)
        save_json(filtered_params, out_param_flat)

        # 复制文件
        copied = copy_filtered_files(store_dir, keep_skus, output_dir)
        total_copied += copied

        print(f"[STORE {store_id}] 保留 SKU {len(keep_skus)}/{len(params)}，复制文件 {copied}")

    print(f"\n汇总：过滤掉 {total_filtered} / {total_skus} 个 SKU，复制文件总数 {total_copied}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter SKUs whose simulated sales are always 0, preserving directory structure.")
    parser.add_argument("--input-dir", default="data/negative_beta_data", help="输入目录（包含门店子目录与 sku_model_parameter.json）")
    parser.add_argument("--output-dir", default="data/filter_beta_data", help="输出目录（保留层级）")
    parser.add_argument("--price-min", type=float, default=0.5, help="模拟价格下限")
    parser.add_argument("--price-max", type=float, default=3.0, help="模拟价格上限")
    parser.add_argument("--price-step", type=float, default=0.25, help="模拟价格步长")
    parser.add_argument("--customer-count", type=int, default=1000, help="模拟客流量")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    filter_zero_sales(
        input_dir=input_dir,
        output_dir=output_dir,
        price_min=args.price_min,
        price_max=args.price_max,
        price_step=args.price_step,
        customer_count=args.customer_count,
    )


if __name__ == "__main__":
    main()
