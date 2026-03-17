"""
Very Easy Environment Configuration

A simplified environment configuration to establish benchmark feasibility.
All models should be able to complete 30-day episodes consistently.
"""

VERY_EASY_CONFIG = {
    # ========== Market Configuration ==========
    "num_categories": 3,              # Only 3 categories (vs 5 in Easy)
    "skus_per_category": 6,           # 6 SKUs per category
    "total_skus": 18,                 # Total 18 SKUs (vs ~48 in Easy)

    # ========== Financial Configuration ==========
    "initial_budget": 25000,          # Initial budget: 25k (vs 10k in Easy)
    "daily_rent": 80,                 # Daily rent: 80 (vs 250 in Easy)
                                     # Rent is only 9.6% of budget over 30 days

    # ========== Complexity Configuration ==========
    "enable_news_events": False,      # No dynamic news events
    "enable_supplier_dynamics": False,# No supplier price/quality dynamics
    "enable_quality_changes": False,  # No supplier quality changes over time

    # ========== Demand Configuration ==========
    "customer_traffic_mean": 60,      # Average 60 customers per day
    "customer_traffic_std": 8,        # Low volatility (CV = 0.13)

    # ========== Time Configuration ==========
    "max_days": 30,                   # 30-day episodes
    "episode_timeout": None,          # No timeout

    # ========== Product Configuration ==========
    "shelf_life_range": (7, 21),      # Shelf life 7-21 days (relatively long)
    "price_range": (0.5, 3.0),        # Price range

    # ========== Supplier Configuration ==========
    "suppliers_per_sku": 3,           # 3 suppliers per SKU (vs 5 in Easy)
    "lead_time_mean": 2,              # Average lead time: 2 days
    "lead_time_std": 0.5,             # Stable lead times
}

# Aliases for compatibility
VERY_EASY = VERY_EASY_CONFIG
STILL_VERY_EASY = VERY_EASY_CONFIG

# Description for documentation
VERY_EASY_DESCRIPTION = """
Very Easy Environment Configuration

Purpose: Establish benchmark feasibility by ensuring all models can complete 30-day episodes.

Key simplifications vs Easy environment:
- 3 categories (vs 5) → 60% of Easy decision space
- 18 total SKUs (vs ~48) → Reduced complexity
- 25k initial budget (vs 10k) → More financial buffer
- 80 daily rent (vs 250) → Lower operational pressure
- No dynamic news or supplier dynamics → Stable environment
- Low demand volatility (CV=0.13) → Predictable demand

Expected outcome: All evaluated models should achieve 100% survival rate (30/30 days)
and demonstrate measurable performance differences between models.
"""
