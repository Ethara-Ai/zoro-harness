import os
import json
from glob import glob
from model.sku_model import fit_logit_demand_model


def run_sku_models_for_stores(data_root):
    """
    data_root: 整个数据根目录，比如 /.../filtered_post_data
    按 store 维度来处理，递归搜索该 store 目录下的所有 csv 文件。
    返回一个 dict: {store_id: {sku_id: {alpha, beta, sigma}}}
    """
    # 第一层：所有 store 目录（如 12, 13, 14 ...）
    stores = [
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    ]

    all_store_models = {}

    for store_id in stores:
        store_path = os.path.join(data_root, store_id)

        # 递归搜这个 store 目录下的所有 csv
        csv_files = glob(os.path.join(store_path, "**", "*.csv"), recursive=True)

        if not csv_files:
            print(f"[WARN] Store {store_id} 下没有找到任何 csv 文件")
            continue

        print(f"[INFO] Store {store_id} 找到 {len(csv_files)} 个 csv 文件")

        all_store_models[store_id] = {}

        for csv_file in csv_files:
            sku_filename = os.path.basename(csv_file)
            sku_id = os.path.splitext(sku_filename)[0]

            try:
                params = fit_logit_demand_model(csv_file)

                all_store_models[store_id][sku_id] = params

                # 你要的话可以打开这行调试打印
                # print(f"[OK] Store {store_id}, SKU {sku_id}: alpha={alpha}, beta={beta}, sigma={sigma}")

            except Exception as e:
                print(f"[ERROR] 处理 Store {store_id}, SKU {sku_id} 时出错: {e}")
                continue

    return all_store_models


def save_models_to_each_store_folder(data_root, store_models):
    """
    将每个 store 的模型参数保存到该 store 目录下的 sku_model_parameter.json

    也就是：
        data_root/store_id/sku_model_parameter.json
    """
    for store_id, sku_params in store_models.items():
        store_path = os.path.join(data_root, store_id)
        if not os.path.exists(store_path):
            # 理论上不会发生，打印一下提示
            print(f"[WARN] Store 目录不存在: {store_path}")
            continue

        output_file = os.path.join(store_path, "sku_model_parameter.json")

        with open(output_file, "w") as f:
            json.dump(sku_params, f, indent=4, ensure_ascii=False)

        print(f"[SAVE] {store_id} -> {output_file}")


if __name__ == "__main__":
    # 你给的是“数据文件夹”本身
    data_path = "/Users/linghuazhang/Desktop/RetailBenchRubbish/filtered_source_data_by_category"

    store_models = run_sku_models_for_stores(data_path)
    save_models_to_each_store_folder(data_path, store_models)