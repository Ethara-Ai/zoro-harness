from pathlib import Path

import pandas as pd


def regroup_by_category(source_root: str, output_root: str, chunksize: int = 100_000):
    """
    把 data/filtered_source_data_by_store 下的文件按“类目”重新归类，
    同一类目同一 SKU 会合并到一个文件（按行追加），但不跨类目合并。

    输入目录示例:
        source_root/100/Cookies/1470001100.csv
        source_root/101/Cookies/1470001100.csv

    输出目录结构:
        output_root/Cookies/1470001100.csv
        output_root/Cookies/11470001100.csv
    """
    source_path = Path(source_root)
    output_path = Path(output_root)

    if not source_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {source_path}")

    # category -> sku -> list(csv paths)
    category_map: dict[str, dict[str, list[Path]]] = {}

    for store_dir in sorted(p for p in source_path.iterdir() if p.is_dir()):
        for category_dir in sorted(p for p in store_dir.iterdir() if p.is_dir()):
            for csv_path in sorted(category_dir.glob("*.csv")):
                sku = csv_path.stem
                category_map.setdefault(category_dir.name, {}).setdefault(sku, []).append(csv_path)

    if not category_map:
        print("未找到任何 csv 文件")
        return

    for category, sku_map in sorted(category_map.items()):
        target_dir = output_path / category
        target_dir.mkdir(parents=True, exist_ok=True)

        for sku, files in sorted(sku_map.items()):
            target_file = target_dir / f"{sku}.csv"
            if target_file.exists():
                target_file.unlink()

            header_written = False
            for csv_path in files:
                for chunk in pd.read_csv(csv_path, chunksize=chunksize):
                    chunk.to_csv(
                        target_file,
                        mode="a",
                        index=False,
                        header=not header_written,
                    )
                    header_written = True

        print(f"{category}: {sum(len(v) for v in sku_map.values())} 个文件合并到 {target_dir}")


if __name__ == "__main__":
    regroup_by_category(
        source_root="data/filtered_source_data_by_store",
        output_root="data/filtered_source_data_by_category",
        chunksize=100_000,
    )
