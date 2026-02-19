"""
Aggregate `sales_by_sku` changes from end_today tool calls.

Usage:
    python script/analyze_end_today_sales.py --file logs/2025-12-14_04-48-13/tool_calls.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any


def load_end_today_entries(file_path: Path) -> List[Dict[str, Any]]:
    """Return log entries whose tool is end_today, sorted by current_date."""
    data = json.loads(file_path.read_text(encoding="utf-8"))
    end_calls = [row for row in data if row.get("tool") == "end_today"]

    def _date_key(row: Dict[str, Any]) -> str:
        result = (row.get("result") or {}).get("result") or {}
        return str(result.get("current_date") or row.get("current_date") or "")

    return sorted(end_calls, key=_date_key)


def build_sales_series(entries: List[Dict[str, Any]]) -> Dict[str, List[Tuple[str, int]]]:
    """Build per-SKU timeseries: [(date, units_sold)]."""
    series: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for row in entries:
        result = (row.get("result") or {}).get("result") or {}
        date_str = str(result.get("current_date") or row.get("current_date") or "")
        sales_map = result.get("sales_by_sku") or {}
        for sku, units in sales_map.items():
            series[str(sku)].append((date_str, int(units)))
    return series


def add_deltas(series: Dict[str, List[Tuple[str, int]]]) -> Dict[str, List[Tuple[str, int, int]]]:
    """Add per-day delta compared to previous entry."""
    result: Dict[str, List[Tuple[str, int, int]]] = {}
    for sku, values in series.items():
        enriched: List[Tuple[str, int, int]] = []
        prev_units = None
        for date_str, units in values:
            delta = units if prev_units is None else units - prev_units
            enriched.append((date_str, units, delta))
            prev_units = units
        result[sku] = enriched
    return result


def print_report(series_with_delta: Dict[str, List[Tuple[str, int, int]]], total_entries: int) -> None:
    print(f"Total end_today entries: {total_entries}")
    print(f"SKUs with sales records: {len(series_with_delta)}\n")

    for sku, rows in sorted(series_with_delta.items()):
        print(f"SKU {sku}:")
        for date_str, units, delta in rows:
            print(f"  {date_str}: sold={units}, delta={delta}")
        print()


def plot_series(series_with_delta: Dict[str, List[Tuple[str, int, int]]], plot_dir: Path) -> None:
    """Plot per-SKU sales curves (units sold) and save as PNG."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for plotting. Install with `pip install matplotlib`.") from exc

    plot_dir.mkdir(parents=True, exist_ok=True)

    for sku, rows in series_with_delta.items():
        dates = []
        units = []
        for date_str, unit, _ in rows:
            try:
                # Most dates are ISO formatted (YYYY-MM-DD)
                from datetime import datetime
                dates.append(datetime.fromisoformat(str(date_str)))
            except Exception:
                dates.append(str(date_str))
            units.append(unit)

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(dates, units, marker="o", linewidth=1.5)
        ax.set_title(f"SKU {sku} sales_by_sku")
        ax.set_xlabel("Date")
        ax.set_ylabel("Units sold")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.autofmt_xdate(rotation=30)
        out_path = plot_dir / f"{sku}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize sales_by_sku changes from end_today logs.")
    parser.add_argument(
        "--file",
        required=True,
        help="Path to tool_calls.json to parse.",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Optional directory to output per-SKU plots. If omitted, plotting is skipped.",
    )
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        raise FileNotFoundError(f"Log file not found: {file_path}")

    entries = load_end_today_entries(file_path)
    sales_series = build_sales_series(entries)
    sales_with_delta = add_deltas(sales_series)
    print_report(sales_with_delta, len(entries))

    if args.plot_dir:
        plot_dir = Path(args.plot_dir)
        plot_series(sales_with_delta, plot_dir)
        print(f"Saved plots to {plot_dir}")


if __name__ == "__main__":
    main()
