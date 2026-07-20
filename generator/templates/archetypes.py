# generator/templates/archetypes.py
# 15-key base configs, one per archetype.
# Variable keys (global_random_seed, selected_categories, category_effects,
# initial_funds, everyday_rent, inventory_capacity, review_ratio,
# news_impact_base_scale, news_sample_ratios, news_daily_count,
# news_random_seed, initial_inventory) are NOT here — the generator adds them.

DYNAMIC_HARD = {
    "debug": False,
    "data_dir": "data/dynamic/simulate_data",
    "customer_data_path": "data/dynamic/customer_number",
    "init_sql_path": "data/dynamic/simulate_data/15/records.sql",
    "data_begin_time": "06/06/91",
    "data_end_time": "12/31/95",
    "store_begin_time": "09/07/91",
    "store_id": "15",
    "order_record_dir": "order_records",        # overridden at runtime by run_env.py
    "enable_review": True,
    "review_model_path": "data/dynamic/review/review_star_smooth_params.json",
    "review_source_path": "data/dynamic/review/all_category_reviews.jsonl",
    "enable_new": True,
    "news_source_path": "data/dynamic/simulate_data/15/news_merged.jsonl",
    "news_impact_mode_weights": {
        "neutral": 0.0,
        "macro_all": 1.0,
        "single_category": 1.0,
        "sku_level": 1.2,
    },
}

DYNAMIC_MIDDLE = {
    **DYNAMIC_HARD,
    "enable_review": True,
    "enable_new": False,
}

STILL_HARD = {
    **DYNAMIC_HARD,
    "data_dir": "data/still/simulate_data",
    "customer_data_path": "data/still/customer_number",
    "init_sql_path": "data/still/simulate_data/15/records.sql",
    "review_model_path": "data/still/review/review_star_smooth_params.json",
    "review_source_path": "data/still/review/all_category_reviews.jsonl",
    "enable_new": False,
    "news_source_path": "data/still/simulate_data/15/news_merged.jsonl",
}

STILL_MIDDLE = {
    **STILL_HARD,
    # Same locked keys as still_hard; difficulty comes from variable keys only.
}

ARCHETYPES = {
    "dynamic_hard":   DYNAMIC_HARD,
    "dynamic_middle": DYNAMIC_MIDDLE,
    "still_hard":     STILL_HARD,
    "still_middle":   STILL_MIDDLE,
}
