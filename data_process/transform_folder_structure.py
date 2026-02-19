import shutil
from pathlib import Path


def rearrange_csv_files(input_root: str, output_root: str):
    """
    把所有形如
        input_root/folder_name/category/store/sku.csv
    的文件复制到
        output_root/store/category/folder_name/sku.csv

    参数:
    - input_root: 原始 data 根目录，例如 "data"
    - output_root: 输出根目录，例如 "outputdir"
    """
    input_root_path = Path(input_root).resolve()
    output_root_path = Path(output_root).resolve()

    if not input_root_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_root_path}")

    # 遍历所有 csv 文件
    for csv_path in input_root_path.rglob("*.csv"):
        # 计算相对路径，例如:
        # data/filtered_middle_data/Analgesics/12/3828161001.csv
        # -> rel_parts = ("filtered_middle_data", "Analgesics", "12", "3828161001.csv")
        rel = csv_path.relative_to(input_root_path)
        parts = rel.parts

        # 要求至少有 4 层: folder_name/category/store/sku.csv
        if len(parts) < 4:
            print(f"[跳过] 层级不足 4: {rel}")
            continue

        folder_name = parts[0]
        category = parts[1]
        store = parts[2]
        filename = parts[-1]  # sku.csv

        # 目标路径: output_root/store/category/folder_name/sku.csv
        target_path = output_root_path / store / category / folder_name / filename
        target_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(csv_path, target_path)
        # print(f"[复制] {csv_path} -> {target_path}")


if __name__ == "__main__":
    # 举例：你的结构类似于 data/filtered_middle_data/Analgesics/12/xxx.csv
    # 那就把 input_root 设为 data
    rearrange_csv_files(
        input_root="data",
        output_root="post_data"
    )
