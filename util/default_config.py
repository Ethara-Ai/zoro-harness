
def create_dynamic_hard_config():
    """
    Create a default configuration for the retail environment.
    """
    return {
        "debug": True,

        "data_dir": "data/dynamic/simulate_data",
        "customer_data_path": "data/dynamic/customer_number",
        "init_sql_path": 'data/dynamic/simulate_data/15/records.sql',
        "data_begin_time": "06/06/91",
        "data_end_time": "12/31/95",
        "store_begin_time": "09/07/91",
        "store_id": "15",  # Using store 8 as an example
        "order_record_dir": "order_records",

        "enable_review": True,
        "review_model_path": "data/dynamic/review/review_star_smooth_params.json",
        "review_source_path": "data/dynamic/review/all_category_reviews.jsonl",
        "review_ratio": 0.02,

        "enable_new": True,
        "news_source_path": "data/dynamic/simulate_data/15/news_merged.jsonl",
        "news_impact_base_scale": 0.4,
        "news_sample_ratios": {
            "neutral": 0.9,
            "single_category": 0.02,
            "macro_all": 0.03,
            "sku_level": 0.05,
        },
        "news_daily_count": 20,
        "news_random_seed": 42,
        "news_impact_mode_weights": {
            # 对所有品类 / 全局的新闻，一般偏弱一些
            "neutral": 0.0,
            "macro_all": 1.0,
            "single_category": 1.0,
            "sku_level": 1.2,
        },

        "initial_funds": 50000,
        "everyday_rent": 1000,
        "inventory_capacity": 40000,
        "initial_inventory": {},
        # 全局随机种子：用于锁定 retail_environment 内部的随机行为（如价格初始化等）
        # 如需完全关闭固定种子，可将该值设为 None
        "global_random_seed": 42,
        "selected_categories": [
            "Bathroom_Tissues",
            "Beer",
            "Bottled_Juices",
            "Canned_Soup",
            "Canned_Tuna",
            "Cereals",
            "Cheeses",
            "Cigarettes",
            "Cookies",
            "Crackers",
            "Dish_Detergent",
            "Fabric_Softeners",
            "Front_end_candies",
            "Frozen_Entrees",
            "Frozen_Juices",
            "Oatmeal",
            "Paper_Towels",
            "Snack_Crackers",
            "Soft_Drinks",
            "Toothpastes"
        ],
        "category_effects": {
            "Bathroom Tissues": -0.2,
            "Beer": -0.2,
            "Bottled Juices": -0.2,
            "Canned Soup": -0.2,
            "Canned Tuna": -0.2,
            "Cereals": -0.2,
            "Cheeses": -0.2,
            "Cigarettes": -0.2,
            "Cookies": -0.2,
            "Crackers": -0.2,
            "Dish Detergent": -0.2,
            "Fabric Softeners": -0.2,
            "Front end candies": -0.2,
            "Frozen Entrees": -0.2,
            "Frozen Juices": -0.2,
            "Oatmeal": -0.2,
            "Paper Towels": -0.2,
            "Snack Crackers": -0.2,
            "Soft Drinks": -0.2,
            "Toothpastes": -0.2,
        },
    }



def create_dynamic_middle_config():
    """
    Create a default configuration for the retail environment.
    """
    return {
        "debug": False,

        "data_dir": "data/dynamic/simulate_data",
        "customer_data_path": "data/dynamic/customer_number",
        "init_sql_path": 'data/dynamic/simulate_data/15/records.sql',
        "data_begin_time": "06/06/91",
        "data_end_time": "12/31/95",
        "store_begin_time": "09/07/91",
        "store_id": "15",  # Using store 8 as an example
        "order_record_dir": "order_records",

        "enable_review": False,
        "review_model_path": "data/dynamic/review/review_star_smooth_params.json",
        "review_source_path": "data/dynamic/review/all_category_reviews.jsonl",
        "review_ratio": 0.02,

        "enable_new": False,
        "news_source_path": "data/simulate_data/15/news_merged.jsonl",
        "news_impact_base_scale": 0.8,
        "news_sample_ratios": {
            "neutral": 0.65,
            "single_category": 0.1,
            "macro_all": 0.05,
            "sku_level": 0.2,
        },
        "news_daily_count": 20,
        "news_random_seed": 42,
        "news_impact_mode_weights": {
            # 对所有品类 / 全局的新闻，一般偏弱一些
            "neutral": 0.0,
            "macro_all": 1.0,
            "single_category": 1.0,
            "sku_level": 1.2,
        },

        "initial_funds": 10000,
        "everyday_rent": 250,
        "inventory_capacity": 10000,
        "initial_inventory": {},
        # 全局随机种子（简单配置下也保持可复现实验）
        "global_random_seed": 42,
        "selected_categories": [
            "Bathroom_Tissues",
            # "Beer",
            # "Bottled_Juices",
            "Canned_Soup",
            # "Canned_Tuna",
            # "Cereals",
            # "Cheeses",
            "Cigarettes",
            # "Cookies",
            # "Crackers",
            # "Dish_Detergent",
            # "Fabric_Softeners",
            "Front_end_candies",
            # "Frozen_Entrees",
            # "Frozen_Juices",
            # "Oatmeal",
            # "Paper_Towels",
            # "Snack_Crackers",
            "Soft_Drinks",
            # "Toothpastes"
        ],
        "category_effects": {
            "Bathroom Tissues": -0.2,
            "Beer": -0.2,
            "Bottled Juices": -0.2,
            "Canned Soup": -0.2,
            "Canned Tuna": -0.2,
            "Cereals": -0.2,
            "Cheeses": -0.2,
            "Cigarettes": -0.2,
            "Cookies": -0.2,
            "Crackers": -0.2,
            "Dish Detergent": -0.2,
            "Fabric Softeners": -0.2,
            "Front end candies": -0.2,
            "Frozen Entrees": -0.2,
            "Frozen Juices": -0.2,
            "Oatmeal": -0.2,
            "Paper Towels": -0.2,
            "Snack Crackers": -0.2,
            "Soft Drinks": -0.2,
            "Toothpastes": -0.2,
        },
    }




def create_still_hard_config():
    """
    Create a default configuration for the retail environment.
    """
    return {
        "debug": True,

        "data_dir": "data/still/simulate_data",
        "customer_data_path": "data/still/customer_number",
        "init_sql_path": 'data/still/simulate_data/15/records.sql',
        "data_begin_time": "06/06/91",
        "data_end_time": "12/31/95",
        "store_begin_time": "09/07/91",
        "store_id": "15",  # Using store 8 as an example
        "order_record_dir": "order_records",

        "enable_review": True,
        "review_model_path": "data/still/review/review_star_smooth_params.json",
        "review_source_path": "data/still/review/all_category_reviews.jsonl",
        "review_ratio": 0.02,

        "enable_new": False,
        "news_source_path": "data/still/simulate_data/15/news_merged.jsonl",
        "news_impact_base_scale": 0.8,
        "news_sample_ratios": {
            "neutral": 0.65,
            "single_category": 0.1,
            "macro_all": 0.05,
            "sku_level": 0.2,
        },
        "news_daily_count": 20,
        "news_random_seed": 42,
        "news_impact_mode_weights": {
            # 对所有品类 / 全局的新闻，一般偏弱一些
            "neutral": 0.0,
            "macro_all": 1.0,
            "single_category": 1.0,
            "sku_level": 1.2,
        },

        "initial_funds": 50000,
        "everyday_rent": 1000,
        "inventory_capacity": 40000,
        "initial_inventory": {},
        # 全局随机种子：用于锁定 retail_environment 内部的随机行为（如价格初始化等）
        # 如需完全关闭固定种子，可将该值设为 None
        "global_random_seed": 42,
        "selected_categories": [
            "Bathroom_Tissues",
            "Beer",
            "Bottled_Juices",
            "Canned_Soup",
            "Canned_Tuna",
            "Cereals",
            "Cheeses",
            "Cigarettes",
            "Cookies",
            "Crackers",
            "Dish_Detergent",
            "Fabric_Softeners",
            "Front_end_candies",
            "Frozen_Entrees",
            "Frozen_Juices",
            "Oatmeal",
            "Paper_Towels",
            "Snack_Crackers",
            "Soft_Drinks",
            "Toothpastes"
        ],
        "category_effects": {
            "Bathroom Tissues": -0.2,
            "Beer": -0.2,
            "Bottled Juices": -0.2,
            "Canned Soup": -0.2,
            "Canned Tuna": -0.2,
            "Cereals": -0.2,
            "Cheeses": -0.2,
            "Cigarettes": -0.2,
            "Cookies": -0.2,
            "Crackers": -0.2,
            "Dish Detergent": -0.2,
            "Fabric Softeners": -0.2,
            "Front end candies": -0.2,
            "Frozen Entrees": -0.2,
            "Frozen Juices": -0.2,
            "Oatmeal": -0.2,
            "Paper Towels": -0.2,
            "Snack Crackers": -0.2,
            "Soft Drinks": -0.2,
            "Toothpastes": -0.2,
        },
    }


def create_still_middle_config():
    """
    Create a default configuration for the retail environment.
    """
    return {
        "debug": True,

        "data_dir": "data/still/simulate_data",
        "customer_data_path": "data/still/customer_number",
        "init_sql_path": 'data/still/simulate_data/15/records.sql',
        "data_begin_time": "06/06/91",
        "data_end_time": "12/31/95",
        "store_begin_time": "09/07/91",
        "store_id": "15",  # Using store 8 as an example
        "order_record_dir": "order_records",

        "enable_review": True,
        "review_model_path": "data/still/review/review_star_smooth_params.json",
        "review_source_path": "data/still/review/all_category_reviews.jsonl",
        "review_ratio": 0.02,

        "enable_new": False,
        "news_source_path": "data/simulate_data/15/news_merged.jsonl",
        "news_impact_base_scale": 0.8,
        "news_sample_ratios": {
            "neutral": 0.65,
            "single_category": 0.1,
            "macro_all": 0.05,
            "sku_level": 0.2,
        },
        "news_daily_count": 20,
        "news_random_seed": 42,
        "news_impact_mode_weights": {
            # 对所有品类 / 全局的新闻，一般偏弱一些
            "neutral": 0.0,
            "macro_all": 1.0,
            "single_category": 1.0,
            "sku_level": 1.2,
        },

        "initial_funds": 10000,
        "everyday_rent": 250,
        "inventory_capacity": 10000,
        "initial_inventory": {},
        # 全局随机种子（中等配置：与 simple 一致，便于对比）
        "global_random_seed": 42,
        "selected_categories": [
            "Bathroom_Tissues",
            # "Beer",
            # "Bottled_Juices",
            "Canned_Soup",
            # "Canned_Tuna",
            # "Cereals",
            # "Cheeses",
            "Cigarettes",
            # "Cookies",
            # "Crackers",
            # "Dish_Detergent",
            # "Fabric_Softeners",
            "Front_end_candies",
            # "Frozen_Entrees",
            # "Frozen_Juices",
            # "Oatmeal",
            # "Paper_Towels",
            # "Snack_Crackers",
            "Soft_Drinks",
            # "Toothpastes"
        ],
        "category_effects": {
            "Bathroom Tissues": -0.2,
            "Beer": -0.2,
            "Bottled Juices": -0.2,
            "Canned Soup": -0.2,
            "Canned Tuna": -0.2,
            "Cereals": -0.2,
            "Cheeses": -0.2,
            "Cigarettes": -0.2,
            "Cookies": -0.2,
            "Crackers": -0.2,
            "Dish Detergent": -0.2,
            "Fabric Softeners": -0.2,
            "Front end candies": -0.2,
            "Frozen Entrees": -0.2,
            "Frozen Juices": -0.2,
            "Oatmeal": -0.2,
            "Paper Towels": -0.2,
            "Snack Crackers": -0.2,
            "Soft Drinks": -0.2,
            "Toothpastes": -0.2,
        },
    }