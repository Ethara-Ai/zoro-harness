"""
Print each SKU's initial rating (if available) and current average rating
queried via the environment tool.

Runs with default config; no file output.
"""

from retail_environment import RetailEnvironment
from util.default_config import create_default_config


def get_initial_rating(sku) -> float | None:
    """Try to read initial_rating from attribute or attributes dict."""
    if hasattr(sku, "initial_rating"):
        return getattr(sku, "initial_rating")
    if isinstance(getattr(sku, "attributes", None), dict):
        return sku.attributes.get("initial_rating")
    return None


def main() -> None:
    config = create_default_config()
    env = RetailEnvironment(config)

    sku_ids = [sku.sku_id for sku in env.skus_list]

    # Query average ratings via tool
    avg_resp = env.exec_tools(
        "view_sku_avg_ratings",
        sku_ids=sku_ids,
        start_date=None,
        end_date=None,
    )
    avg_map = avg_resp.get("result", {}) or {}

    print("SKU\tInitialRating\tAvgRating")
    for sku in env.skus_list:
        init_rate = get_initial_rating(sku)
        avg_rate = avg_map.get(sku.sku_id)
        print(f"{sku.sku_id}\t{init_rate}\t{avg_rate}")


if __name__ == "__main__":
    main()
