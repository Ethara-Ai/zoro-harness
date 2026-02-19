import os
import pandas as pd
import csv
import shutil
from pathlib import Path
from tqdm import tqdm  # 需要先 pip install tqdm


def filter_csv_files(base_dir, output_dir, subfolders_to_process=None, skip_subfolders=None):
    """
    遍历 base_dir 下所有 CSV 文件，按以下规则筛选：
    1. 有效数据行数（总行数） >= 350
    2. MOVE 列的非空值平均值 >= 10

    对于满足条件的文件：
        - 将原 CSV 复制到 output_dir 中对应的相对路径位置
    同时在 output_dir 下生成：
        - valid_files.csv：记录通过筛选的文件
        - filtered_files.csv：记录被过滤掉的文件及原因

    支持按子文件夹分批执行：
        - subfolders_to_process: 要处理的子文件夹列表（相对于base_dir），为 None 时表示处理全部
        - skip_subfolders: 要跳过的子文件夹列表（相对于base_dir），优先级高于 subfolders_to_process
    """
    base_path = Path(base_dir)
    output_path = Path(output_dir)

    # 创建输出根目录
    output_path.mkdir(parents=True, exist_ok=True)

    # 获取所有子目录
    all_subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    # 过滤要处理的子目录
    if skip_subfolders:
        all_subdirs = [d for d in all_subdirs if d.name not in skip_subfolders]
    
    if subfolders_to_process:
        all_subdirs = [d for d in all_subdirs if d.name in subfolders_to_process]
    
    print(f"Found {len(all_subdirs)} subdirectories to process")
    
    valid_files = []
    filtered_files = []

    # 逐个处理子目录
    for subdir in all_subdirs:
        print(f"\nProcessing subdirectory: {subdir.name}")
        
        # 获取当前子目录下的所有CSV文件
        csv_files = list(subdir.rglob("*.csv"))
        print(f"Found {len(csv_files)} CSV files in {subdir.name}")
        
        # 用 tqdm 显示进度条
        for csv_file in tqdm(csv_files, desc=f"Processing {subdir.name}", unit="file"):
            try:
                # 读取 CSV
                df = pd.read_csv(csv_file)

                valid_rows = df[
                    df['MOVE'].notna() &
                    df['PRICE'].notna() &
                    (df['MOVE'] > 0) &
                    (df['PRICE'] > 0)
                ]

                valid_data_count = len(valid_rows)

                # 计算平均 MOVE
                if 'MOVE' in df.columns:
                    move_values = df['MOVE'].dropna()
                    avg_move = move_values.mean() if len(move_values) > 0 else 0
                else:
                    # 用 tqdm.write 避免打乱进度条
                    tqdm.write(f"  - Warning: No 'MOVE' column found in {csv_file}")
                    avg_move = 0

                # 条件判断
                meets_valid_data_threshold = valid_data_count >= 300
                meets_avg_move_threshold = avg_move >= 20

                if meets_valid_data_threshold and meets_avg_move_threshold:
                    # 记录为有效文件
                    valid_files.append({
                        'file_path': str(csv_file),
                        'valid_data_count': valid_data_count,
                        'avg_move': avg_move
                    })

                    # 计算相对路径，并复制到 output_dir 下同样的结构
                    relative_path = csv_file.relative_to(base_path)
                    out_file = output_path / relative_path
                    out_file.parent.mkdir(parents=True, exist_ok=True)

                    # 使用拷贝保持文件内容完全一致（也可以用 df.to_csv）
                    shutil.copy2(csv_file, out_file)

                    # tqdm.write(f"  - VALID   | rows: {valid_data_count}, avg MOVE: {avg_move:.2f}, copied to {out_file}")
                else:
                    # 构造过滤原因
                    reason_parts = []
                    if not meets_valid_data_threshold:
                        reason_parts.append(f"Low data count ({valid_data_count})")
                    if not meets_avg_move_threshold:
                        reason_parts.append(f"Low avg MOVE ({round(avg_move, 2)})")
                    reason = ", ".join(reason_parts) if reason_parts else "Unknown reason"

                    filtered_files.append({
                        'file_path': str(csv_file),
                        'valid_data_count': valid_data_count,
                        'avg_move': avg_move,
                        'reason': reason
                    })

                    # tqdm.write(f"  - FILTERED| rows: {valid_data_count}, avg MOVE: {avg_move:.2f}, reason: {reason}")

            except Exception as e:
                tqdm.write(f"  - ERROR   | file: {csv_file}, err: {e}")
                continue

    # 汇总信息
    print("\nFiltering Summary (all subdirectories processed):")
    print(f"Valid files: {len(valid_files)}")
    print(f"Filtered files: {len(filtered_files)}")

    # 在 output_dir 下保存筛选结果列表
    if valid_files:
        valid_df = pd.DataFrame(valid_files)
        valid_list_path = output_path / "valid_files.csv"
        valid_df.to_csv(valid_list_path, index=False)
        print(f"Valid files list saved to: {valid_list_path}")

    if filtered_files:
        filtered_df = pd.DataFrame(filtered_files)
        filtered_list_path = output_path / "filtered_files.csv"
        filtered_df.to_csv(filtered_list_path, index=False)
        print(f"Filtered files list saved to: {filtered_list_path}")

    return valid_files, filtered_files

 
if __name__ == "__main__":

    base_directory = '/Users/linghuazhang/Desktop/Project/RetailBench/data/source_data_by_store'
    output_directory = '/Users/linghuazhang/Desktop/Project/RetailBench/data/filtered_source_data_by_store'

    # Process subfolders one by one
    valid_files, filtered_files = filter_csv_files(
        base_directory,
        output_directory
    )

    print("\nFiltering process completed!")