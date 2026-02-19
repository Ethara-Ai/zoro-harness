#!/usr/bin/env python3
"""
Plot funds and net worth over time from a tool_calls.json or tool_calls.jsonl log.
Supports both JSON array format and JSONL (JSON Lines) format.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def load_log(path: Path):
    """
    Load log entries from either JSON array format (.json) or JSONL format (.jsonl).
    
    Args:
        path: Path to the log file
        
    Returns:
        List of log entries (dictionaries)
    """
    if path.suffix == ".jsonl":
        # JSONL format: each line is a JSON object
        entries = []
        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}", file=sys.stderr)
                    continue
        return entries
    else:
        # JSON array format
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Log file must contain a JSON array.")
        return data


def extract_series(entries):
    points = []
    for entry in entries:
        date_raw = entry.get("current_date")
        if not date_raw:
            continue
        try:
            dt = datetime.fromisoformat(str(date_raw))
        except Exception:
            try:
                dt = datetime.strptime(str(date_raw), "%Y-%m-%d")
            except Exception:
                continue
        funds = entry.get("funds")
        net_worth = entry.get("net_worth")
        if funds is None or net_worth is None:
            continue
        points.append((dt, float(funds), float(net_worth)))
    points.sort(key=lambda x: x[0])
    return points


def aggregate_by_date(points):
    """
    Aggregate points by date, taking the last value of each day.
    
    Args:
        points: List of (datetime, funds, net_worth) tuples
        
    Returns:
        List of (date, funds, net_worth) tuples, one per day
    """
    from collections import defaultdict
    
    # Group by date (YYYY-MM-DD)
    daily_data = defaultdict(list)
    for dt, funds, net_worth in points:
        date_key = dt.date()
        daily_data[date_key].append((dt, funds, net_worth))
    
    # For each day, take the last value (most recent)
    aggregated = []
    for date_key in sorted(daily_data.keys()):
        day_points = daily_data[date_key]
        # Sort by datetime and take the last one
        day_points.sort(key=lambda x: x[0])
        last_point = day_points[-1]
        aggregated.append((date_key, last_point[1], last_point[2]))
    
    return aggregated


def plot_series(points):
    dates = [p[0] for p in points]
    funds = [p[1] for p in points]
    networth = [p[2] for p in points]

    x = list(range(1, len(points) + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(x, funds, label="Funds", marker="o")
    plt.plot(x, networth, label="Net Worth", marker="x")

    # Annotate date on the first action of each day (funds curve only)
    seen_dates = set()
    for xi, yi, dt in zip(x, funds, dates):
        label = dt.strftime("%Y-%m-%d")
        if label in seen_dates:
            continue
        seen_dates.add(label)
        plt.text(xi, yi, label, fontsize=8, rotation=45, ha="right", va="bottom")

    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("Funds and Net Worth Across Steps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_by_date(points):
    """
    Plot funds and net worth aggregated by date.
    Each date shows the last value of that day.
    """
    daily_points = aggregate_by_date(points)
    
    if not daily_points:
        print("No data to plot by date.")
        return
    
    dates = [p[0] for p in daily_points]
    funds = [p[1] for p in daily_points]
    networth = [p[2] for p in daily_points]
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates, funds, label="Funds", marker="o", linewidth=2, markersize=6)
    plt.plot(dates, networth, label="Net Worth", marker="x", linewidth=2, markersize=6)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")
    
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Funds and Net Worth by Date (Daily Aggregated)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_multi_networth_by_date(series_by_label: Dict[str, List[Tuple[datetime, float, float]]]) -> None:
    """
    对比多个日志文件的 net_worth 轨迹（按日期聚合，一天一个点，只画 net_worth）。
    series_by_label: {label: [(date, funds, net_worth), ...]}
    """
    if not series_by_label:
        print("No data to plot.")
        return

    plt.figure(figsize=(12, 6))

    for label, points in series_by_label.items():
        if not points:
            continue
        # 聚合到按天的数据
        daily_points = aggregate_by_date(points)
        if not daily_points:
            continue
        dates = [p[0] for p in daily_points]
        networth = [p[2] for p in daily_points]
        plt.plot(dates, networth, marker="o", linewidth=2, markersize=4, label=label)

    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Date")
    plt.ylabel("Net Worth")
    plt.title("Net Worth by Date (Multiple Runs)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def find_log_files(logs_dir: Path = None) -> List[Path]:
    """
    从 logs 文件夹中查找所有 tool_calls.jsonl 文件。
    
    Args:
        logs_dir: logs 文件夹路径，默认为当前目录下的 logs/
        
    Returns:
        找到的所有 tool_calls.jsonl 文件路径列表，按时间戳排序
    """
    if logs_dir is None:
        logs_dir = Path("env_logs")
    else:
        logs_dir = Path(logs_dir)
    
    if not logs_dir.exists():
        print(f"Logs directory not found: {logs_dir}")
        return []
    
    log_files = []
    for subdir in sorted(logs_dir.iterdir()):
        if not subdir.is_dir():
            continue
        tool_calls_file = subdir / "tool_calls.jsonl"
        if tool_calls_file.exists():
            log_files.append(tool_calls_file)
    
    # 按时间戳（目录名）排序
    log_files.sort(key=lambda p: p.parent.name)
    
    return log_files


def main():
    # 如果没有命令行参数，自动从 logs 文件夹读取
    if len(sys.argv) < 2:
        print("No arguments provided, scanning 'logs/' directory for tool_calls.jsonl files...")
        paths = find_log_files()
        
        if not paths:
            print("No log files found in logs/ directory.")
            print("\nUsage:")
            print("  python plot_logs.py                    # 自动扫描 logs/ 文件夹")
            print("  python plot_logs.py <log1.jsonl>       # 单个日志：资金/净值曲线")
            print("  python plot_logs.py <log1.jsonl> <log2.jsonl> [...]  # 多个日志：对比净值曲线")
            sys.exit(1)
        
        print(f"Found {len(paths)} log file(s):")
        for p in paths:
            print(f"  - {p}")
    else:
        # 使用命令行参数指定的文件
        paths = [Path(p) for p in sys.argv[1:]]
        for p in paths:
            if not p.exists():
                print(f"File not found: {p}")
                sys.exit(1)

    # 单文件：保持原有行为（既画 step，也画按日期）
    if len(paths) == 1:
        log_path = paths[0]
        entries = load_log(log_path)
        points = extract_series(entries)
        if not points:
            print("No valid data to plot.")
            sys.exit(0)

        print(f"\nPlotting single log file: {log_path}")
        print("Plotting by step...")
        plot_series(points)
        
        print("Plotting by date...")
        plot_by_date(points)
        return

    # 多文件：按日期对比 net_worth
    print(f"\nLoading {len(paths)} log file(s)...")
    series_by_label: Dict[str, List[Tuple[datetime, float, float]]] = {}
    for p in paths:
        print(f"  Loading {p}...")
        entries = load_log(p)
        pts = extract_series(entries)
        if not pts:
            print(f"    Warning: no valid data in {p}, skip.")
            continue
        # 用父目录名（时间戳）作为 label，便于区分不同实验
        label = p.parent.name or p.stem
        # 若同一标签多次出现（不太可能），则合并
        series_by_label.setdefault(label, []).extend(pts)
        print(f"    Loaded {len(pts)} data points")

    if not series_by_label:
        print("No valid data from any log files.")
        sys.exit(0)

    print(f"\nPlotting net worth by date for {len(series_by_label)} run(s)...")
    plot_multi_networth_by_date(series_by_label)


if __name__ == "__main__":
    main()
