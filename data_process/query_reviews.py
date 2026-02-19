"""
Quick helper to query reviews via env.sales_record_manager without writing files.
Edit SKU_IDS below to target specific SKUs.
"""

from datetime import date
from pprint import pprint

from retail_environment import RetailEnvironment
from util.default_config import create_default_config

# Configure target SKUs here
SKU_IDS = ["1111325381"]  # replace with your SKU ids


def main() -> None:
    config = create_default_config()
    env = RetailEnvironment(config)

    for sku_id in SKU_IDS:
        reviews = env.sales_record_manager.read_reviews(
            sku_id=sku_id,
            start_date=None,  # or date(1991, 12, 15)
            end_date=None,
        )
        print(f"SKU {sku_id}: {len(reviews)} reviews")
        pprint([r.to_dict() for r in reviews])


if __name__ == "__main__":
    main()
