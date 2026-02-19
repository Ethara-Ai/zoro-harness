"""Scan *_daily.json files, report their date ranges, and print the maximal intersection."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path("data/simulate_data")
DATE_FIELD = "date"


def parse_date(value: str):
    """Parse ISO-like date string to datetime.date; return None on failure."""
    try:
        return datetime.fromisoformat(str(value)).date()
    except Exception:
        return None


def range_for_file(path: Path) -> Optional[Tuple[str, str]]:
    """Return (min_date, max_date) as ISO strings for a daily JSON file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[SKIP] {path}: load error ({exc})")
        return None

    dates = []
    for row in data if isinstance(data, list) else []:
        date_val = row.get(DATE_FIELD)
        dt = parse_date(date_val)
        if dt:
            dates.append(dt)

    if not dates:
        print(f"[SKIP] {path}: no valid dates")
        return None

    return (min(dates).isoformat(), max(dates).isoformat())


def main() -> None:
    daily_files: List[Path] = sorted(ROOT.rglob("*_daily.json"))
    if not daily_files:
        raise SystemExit(f"No *_daily.json found under {ROOT}")

    ranges: List[Tuple[Path, str, str]] = []
    for file_path in daily_files:
        rng = range_for_file(file_path)
        if rng:
            start, end = rng
            ranges.append((file_path, start, end))

    if not ranges:
        raise SystemExit("No valid date ranges collected.")

    # Report per file
    for file_path, start, end in ranges:
        print(f"{file_path}: {start} -> {end}")

    # Maximal intersection across all collected ranges
    latest_start = max(start for _, start, _ in ranges)
    earliest_end = min(end for _, _, end in ranges)
    print(f"\nMax intersection: {latest_start} -> {earliest_end}")


if __name__ == "__main__":
    main()
