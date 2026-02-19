import os
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 根目录，改成你自己的，比如 "./filtered_post_data"
ROOT_DIR = "/Users/linghuazhang/Desktop/Project/RetailBench/data/simulate_data"
WEEK1_START = datetime.strptime("09/14/89", "%m/%d/%y")

def process_one_csv(csv_path: str):
    # print(f"processing: {csv_path}")
    try:
        df = pd.read_csv(csv_path)

        # 必须要有 COST_PRICE 列
        if "COSTPRICE" not in df.columns:
            print(f"  -> skip: no 'COSTPRICE' column")
            return

        if "WEEK" not in df.columns:
            print(f"  -> skip: no 'WEEK' column")
            return

        # ========= 1. 先做周级平滑 =========
        cost = df["COSTPRICE"].astype(float)

        # 把 0 当成缺失
        cost = cost.replace(0, np.nan)

        # 线性插值补全
        cost = cost.interpolate(method="linear", limit_direction="both")

        # EWMA 平滑（周级）
        smoothed = cost.ewm(span=10, adjust=False).mean()
        df["smoothed_cost_price"] = smoothed

        # ========= 2. 根据 WEEK 做时间映射，生成日级数据 =========
        def week_to_dates(week_value):
            try:
                week_int = int(week_value)
            except (TypeError, ValueError):
                return (pd.NaT, pd.NaT)
            start = WEEK1_START + timedelta(days=(week_int - 1) * 7)
            end = start + timedelta(days=6)
            return (start, end)

        mapped = df["WEEK"].apply(week_to_dates)
        df["date_start"] = [pair[0] for pair in mapped]
        df["date_end"] = [pair[1] for pair in mapped]
        df = df.dropna(subset=["date_start", "date_end"])

        # 转成 datetime 并排序
        df["date_start"] = pd.to_datetime(df["date_start"])
        df["date_end"]   = pd.to_datetime(df["date_end"])
        df = df.sort_values("date_start")

        # 同一周可能有多条记录，先聚合避免重复索引导致 reindex 报错
        agg_fields = {"smoothed_cost_price": "mean"}
        if "STORE" in df.columns:
            agg_fields["STORE"] = "first"
        if "UPC" in df.columns:
            agg_fields["UPC"] = "first"
        df = df.groupby(["date_start", "date_end"], as_index=False).agg(agg_fields)

        # 周级时间序列：索引 = date_start，值 = smoothed_cost_price
        weekly_series = pd.Series(
            df["smoothed_cost_price"].values,
            index=df["date_start"]
        )

        # 构造完整的【日级】时间索引：从最早的 start 到最后一行的 end
        full_index = pd.date_range(
            start=weekly_series.index.min(),
            end=df["date_end"].max(),
            freq="D"
        )

        # 把周级数据放到日级索引上（周起始那天有值，其余先是 NaN）
        daily = weekly_series.reindex(full_index)

        # 按时间插值，把相邻两周之间的变化均匀拆到每天
        daily = daily.interpolate(method="time")

        # ========= 3. 组装日级 DataFrame =========
        daily_df = pd.DataFrame({
            "date": full_index,
            "smoothed_cost_price": daily.values,
        })

        # 如果你想带上 STORE / UPC 信息（每个文件其实只有一个 sku）
        if "STORE" in df.columns:
            daily_df["STORE"] = df["STORE"].iloc[0]
        if "UPC" in df.columns:
            daily_df["UPC"] = df["UPC"].iloc[0]

        # ========= 4. 周级 & 日级分别导出 json =========
        base_path = os.path.splitext(csv_path)[0]

        # （可选）周级 json，保留你之前的结构，如果不需要可以注释掉
        weekly_json_path = base_path + ".json"
        df.to_json(
            weekly_json_path,
            orient="records",
            force_ascii=False,
            indent=2
        )

        daily_df["date"] = daily_df["date"].dt.strftime("%Y-%m-%d")

        # 日级 json：xxx_daily.json —— 这是你要的“日期级别数据”
        daily_json_path = base_path + "_daily.json"
        daily_df.to_json(
            daily_json_path,
            orient="records",
            force_ascii=False,
            indent=2
        )

        # print(f"  -> saved weekly: {weekly_json_path}")
        # print(f"  -> saved daily : {daily_json_path}")

    except Exception as e:
        print(f"  !! error processing {csv_path}: {e}")

def walk_and_process(root_dir: str):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                csv_path = os.path.join(dirpath, filename)
                process_one_csv(csv_path)

def sample_and_plot_json(root_dir: str, sample_size: int = 100):
    # 1. 收集所有 json 文件路径
    json_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".json"):
                json_files.append(os.path.join(dirpath, filename))

    if not json_files:
        print("No json files found, skip plotting.")
        return

    # 2. 随机采样
    if len(json_files) > sample_size:
        sampled_files = random.sample(json_files, sample_size)
    else:
        sampled_files = json_files

    print(f"Total json files: {len(json_files)}, sampled: {len(sampled_files)}")

    # 3. 绘图
    plt.figure(figsize=(12, 8))

    for path in sampled_files:
        try:
            # 对应 to_json(orient="records") 的读取
            df = pd.read_json(path, orient="records")

            if "smoothed_cost_price" not in df.columns:
                continue

            series = df["smoothed_cost_price"].astype(float)

            # 每条线用很低的 alpha，这样 100 条线叠在一起也不至于太糊
            plt.plot(series.values, alpha=0.15)
        except Exception as e:
            print(f"  !! error reading {path}: {e}")

    plt.title("Sampled smoothed_cost_price curves (up to 100 SKUs)")
    plt.xlabel("Time index")
    plt.ylabel("smoothed_cost_price")
    plt.tight_layout()

    # 4. 保存图片
    out_path = os.path.join(root_dir, "smoothed_cost_price_samples.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Plot saved to: {out_path}")


if __name__ == "__main__":
    # 先处理所有 csv -> json
    walk_and_process(ROOT_DIR)

    # 再随机采样 100 个 json 画图
    sample_and_plot_json(ROOT_DIR, sample_size=100)
