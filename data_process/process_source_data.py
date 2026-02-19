import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd


def build_customer_counts(customer_root: Path) -> Dict[str, Dict[float, float]]:
    """Build mapping: store_id -> {week_number -> total customer count}.

    The customer files are stored under ``data/customer_number/store_x/store_x_data.json``.
    Each record has ``week`` (float) and ``custcoun`` (float). We sum over all days in
    the same week to align with the weekly sales data.
    """
    lookup: Dict[str, Dict[float, float]] = {}
    for store_dir in customer_root.glob("store_*"):
        store_id = store_dir.name.split("_", 1)[1]
        json_path = store_dir / f"{store_dir.name}_data.json"
        if not json_path.exists():
            continue
        try:
            records = json.loads(json_path.read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"[customer] failed to read {json_path}: {exc}")
            continue

        week_sum: Dict[float, float] = {}
        for rec in records:
            week = rec.get("week")
            count = rec.get("custcoun")
            if week is None or count is None:
                continue
            try:
                week = float(week)
                count = float(count)
            except (TypeError, ValueError):
                continue
            week_sum[week] = week_sum.get(week, 0.0) + count

        lookup[store_id] = week_sum
    return lookup


def compute_cost_price(price: Any, qty: Any, profit_pct: Any) -> float:
    """Compute cost price per *item* using margin percent and bundle size.

    - ``price``: total price for the bundle (or single item if ``qty`` == 1)
    - ``qty``: bundle size. If zero/NaN, we treat as 1 to avoid division by zero.
    - ``profit_pct``: gross margin percent (e.g., 25.3 means 25.3%)

    Cost per item = (price / qty) * (1 - profit_pct / 100)
    """
    try:
        p = float(price)
        q = float(qty)
        profit = float(profit_pct)
    except (TypeError, ValueError):
        return 0.0

    if q <= 0:
        q = 1.0

    return (p / q) * (1 - profit / 100.0)


def process_category(category_path: Path, output_root: Path, customer_counts: Dict[str, Dict[float, float]]):
    """Process one category folder containing a single w*.csv file."""
    w_files = list(category_path.glob("w*.csv"))
    if not w_files:
        return

    csv_path = w_files[0]
    df = pd.read_csv(csv_path)

    # Ensure numeric types for required fields
    for col in ["STORE", "UPC", "WEEK", "MOVE", "QTY", "PRICE", "PROFIT"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute CUSTOMCOUNT per row via (store, week)
    def lookup_custom_count(row: pd.Series) -> float:
        store_id = str(int(row["STORE"])) if pd.notna(row["STORE"]) else None
        week = float(row["WEEK"]) if pd.notna(row["WEEK"]) else None
        if store_id is None or week is None:
            return float("nan")
        return customer_counts.get(store_id, {}).get(week, float("nan"))

    df["CUSTOMCOUNT"] = df.apply(lookup_custom_count, axis=1)

    # Compute COSTPRICE per item using bundle-aware formula
    df["COSTPRICE"] = df.apply(
        lambda r: compute_cost_price(r.get("PRICE"), r.get("QTY"), r.get("PROFIT")),
        axis=1,
    )

    # Write out per store / per SKU
    for store_id, store_df in df.groupby("STORE"):
        store_str = str(int(store_id)) if pd.notna(store_id) else "unknown"
        for upc, upc_df in store_df.groupby("UPC"):
            upc_str = str(int(upc)) if pd.notna(upc) else "unknown"
            target = output_root / store_str / category_path.name / f"{upc_str}.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            upc_df.to_csv(target, index=False)


def main():
    source_root = Path("data/source_data")
    customer_root = Path("data/customer_number")
    output_root = Path("data/source_data_by_store")

    customer_counts = build_customer_counts(customer_root)
    print(f"Loaded customer counts for {len(customer_counts)} stores")

    for category_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        process_category(category_dir, output_root, customer_counts)
        print(f"processed {category_dir.name}")


if __name__ == "__main__":
    main()
