import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def stat_file_avg_move_distribution(
    base_dir,
    output_dir,
    subfolders_to_process=None,
):
    """
    对 base_dir 下指定子目录(subfolders_to_process)中的 CSV 文件：
        - 计算每个文件的“有效行平均 MOVE”
        - 输出该批文件的 avg_move 分布（CSV + 直方图 + 箱线图）
    """

    base_path = Path(base_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 如果没指定子目录，就报错（避免一次跑全部）
    # 如果没有指定 subfolders_to_process，则自动获取所有子目录（一次跑全部）
    if subfolders_to_process is None:
        subfolders_to_process = [
            d.name for d in base_path.iterdir() if d.is_dir()
        ]
        subfolders_to_process.sort()

        print(f"[STAT] Processing subfolders: {subfolders_to_process}")

    file_stats = []

    for subdir_name in subfolders_to_process:
        subdir = base_path / subdir_name

        if not subdir.exists():
            print(f"[STAT] Skip non-existing folder: {subdir_name}")
            continue

        csv_files = list(subdir.rglob("*.csv"))
        print(f"[STAT] Found {len(csv_files)} CSV files in {subdir_name}")

        for csv_file in tqdm(csv_files, desc=f"[STAT] {subdir_name}", unit="file"):
            try:
                df = pd.read_csv(csv_file)

                if "MOVE" not in df.columns or "PRICE" not in df.columns:
                    continue

                valid_rows = df[
                    df["MOVE"].notna() &
                    df["PRICE"].notna() &
                    (df["MOVE"] > 0) &
                    (df["PRICE"] > 0)
                ]

                if len(valid_rows) == 0:
                    continue

                avg_move = valid_rows["MOVE"].mean()

                file_stats.append({
                    "file_path": str(csv_file),
                    "subdir": subdir_name,
                    "valid_count": len(valid_rows),
                    "avg_move": avg_move
                })

            except Exception as e:
                print(f"  - ERROR in {csv_file}: {e}")

    if not file_stats:
        print("[STAT] No valid files in this batch.")
        return

    stats_df = pd.DataFrame(file_stats)

    # 保存 CSV
    csv_path = out_path / "file_move_distribution.csv"
    stats_df.to_csv(csv_path, index=False)
    print(f"[STAT] Saved CSV: {csv_path}")

    # 画直方图
    plt.figure(figsize=(10, 6))
    plt.hist(stats_df["avg_move"], bins=50)
    plt.xlabel("Per-file avg MOVE")
    plt.ylabel("Number of files")
    plt.title("Distribution of per-file average MOVE")
    plt.tight_layout()
    plt.savefig(out_path / "file_avg_move_hist.png")
    plt.close()

    # 画箱线图
    plt.figure(figsize=(6, 6))
    plt.boxplot(stats_df["avg_move"], vert=True)
    plt.ylabel("Per-file avg MOVE")
    plt.title("Boxplot of per-file average MOVE")
    plt.tight_layout()
    plt.savefig(out_path / "file_avg_move_boxplot.png")
    plt.close()

    print(f"[STAT] Batch finished: {output_dir}")

stat_file_avg_move_distribution(
    base_dir='/Users/linghuazhang/Desktop/grocery/data/dominicks/source_data_processed',
    output_dir="raw_data/stat_move_distribution",
)