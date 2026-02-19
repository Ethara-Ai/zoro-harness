import os
import pandas as pd

def add_cost_price_column(root_dir: str):
    """
    递归遍历 root_dir 下所有 CSV 文件，
    根据 COST_PRICE = PRICE * (PROFIT / 100)
    计算并修复缺省值为 0.0
    """

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.lower().endswith(".csv"):
                continue

            csv_path = os.path.join(dirpath, filename)
            # print(f"处理文件: {csv_path}")

            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"  CSV 读取失败：{e}")
                continue

            # 检查必须列
            if "PRICE" not in df.columns or "PROFIT" not in df.columns:
                print("  缺少 PRICE 或 PROFIT 列，跳过")
                continue

            # 转为数值，无法转换的变为 NaN
            df["PRICE"] = pd.to_numeric(df["PRICE"], errors="coerce")
            df["PROFIT"] = pd.to_numeric(df["PROFIT"], errors="coerce")

            # 修复 NaN → 0.0
            df["PRICE"] = df["PRICE"].fillna(0.0)
            df["PROFIT"] = df["PROFIT"].fillna(0.0)

            # 计算 COST_PRICE
            df["COST_PRICE"] = df["PRICE"] * (df["PROFIT"] / 100.0)

            # 保存回原 CSV
            try:
                df.to_csv(csv_path, index=False)
                # print("  写回成功")
            except Exception as e:
                print(f"  写回失败：{e}")


if __name__ == "__main__":
    # 这里换成你的数据目录
    root_dir = "filtered_post_data"
    add_cost_price_column(root_dir)