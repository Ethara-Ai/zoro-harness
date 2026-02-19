"""
Filter descriptions for store 15 SKUs without CLI args.

Reads:
  - data/simulate_data/15/sku_model_parameter.json
  - data/upc/description.json
Writes:
  - post_data/store_15_descriptions.json
"""

import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    store_id = "15"
    
    sku_model_path = Path("data/simulate_data") / store_id / "sku_model_parameter.json"
    description_path = Path("/Users/linghuazhang/Desktop/Project/RetailBench/data/simulate_data/15/upc.json")
    output_path = Path("post_data") / f"store_{store_id}_descriptions.json"

    sku_params = load_json(sku_model_path)
    target_upcs = set(sku_params.keys())

    descriptions = load_json(description_path)
    filtered = [row for row in descriptions if str(row.get("UPC")) in target_upcs]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Filtered {len(filtered)} descriptions for store {store_id} -> {output_path}")


if __name__ == "__main__":
    main()
