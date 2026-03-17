import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

import matplotlib.pyplot as plt

from util.default_config import create_default_config


def _parse_date(raw: Any):
    """尽量把字符串解析成日期，不合法则返回 None。"""
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.date()
    s = str(raw)
    for fmt in ("%Y-%m-%d", "%m/%d/%y", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            continue
    return None


def load_customer_series(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    根据 config 定位 customer_number 文件并返回有效记录列表。
    - 文件路径:  {customer_data_path}/{store_id}/data.json
    - 过滤掉 custcoun 或 date 为空的记录
    """
    base_dir = Path(config.get("customer_data_path", "data/customer_number"))
    store_id = str(config.get("store_id", ""))
    data_path = base_dir / store_id / "data.json"

    if not data_path.exists():
        raise FileNotFoundError(f"customer_number 文件不存在: {data_path}")

    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    series: List[Dict[str, Any]] = []
    for row in data:
        cnt = row.get("custcoun")
        d = _parse_date(row.get("date"))
        if cnt is None or d is None:
            continue
        try:
            y = float(cnt)
        except Exception:
            continue
        series.append({"date": d, "value": y})

    series.sort(key=lambda r: r["date"])
    return series


def plot_customer_series(series: List[Dict[str, Any]], title: str = "Customer count over time"):
    """
    按年份分组展示客户数量变化。
    每年一个子图，方便对比不同年份的趋势。
    """
    if not series:
        raise ValueError("没有可用的 customer_number 数据可以绘图。")

    # 按年份分组
    by_year: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for record in series:
        year = record["date"].year
        by_year[year].append(record)

    if not by_year:
        raise ValueError("没有有效的年份数据。")

    # 按年份排序
    years = sorted(by_year.keys())
    n_years = len(years)

    # 创建子图：每年一个子图
    fig, axes = plt.subplots(n_years, 1, figsize=(14, 4 * n_years), sharex=True)
    
    # 如果只有一年，axes 不是数组，需要转换
    if n_years == 1:
        axes = [axes]

    for idx, year in enumerate(years):
        year_data = by_year[year]
        xs = [r["date"] for r in year_data]
        ys = [r["value"] for r in year_data]

        ax = axes[idx]
        ax.plot(xs, ys, marker="o", markersize=2, linewidth=1.2, label=f"Year {year}")
        ax.set_ylabel("Customer Count")
        ax.set_title(f"{title} - Year {year}")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # 设置 x 轴标签（只在最后一个子图显示）
    axes[-1].set_xlabel("Date")
    axes[-1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def main():
    """
    入口函数：
    1. 读取默认 config
    2. 定位对应门店的 customer_number 文件
    3. 画出时间序列曲线
    """
    config = create_default_config()
    series = load_customer_series(config)

    store_id = config.get("store_id", "")
    title = f"Store {store_id} customer count"
    plot_customer_series(series, title=title)


if __name__ == "__main__":
    main()

