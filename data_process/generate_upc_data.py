import csv
import io
import json
import multiprocessing
import os
import random
from glob import glob
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Iterable

from openai import OpenAI

from util.file import load_json

dashscope_ak = ''
dashscope_baseurl = 'https://dashscope.aliyuncs.com/compatible-mode/v1'

client = OpenAI(api_key=dashscope_ak, base_url=dashscope_baseurl)

import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

from openai import OpenAI
from util.file import load_json


client = OpenAI(api_key=dashscope_ak, base_url=dashscope_baseurl)

MODEL_NAME = "qwen3-235b-a22b-instruct-2507"


# ===================== Anti-collision 摘要 =====================

_STOPWORDS = {
    "with", "from", "this", "that", "made", "fresh", "pack", "size", "oz", "lb", "count", "ct",
    "and", "or", "for", "the", "a", "an", "to", "in", "on", "of", "by", "as",
    "flavor", "flavoured", "flavored", "assorted"
}

def _norm_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _extract_keywords(desc: str, max_keywords: int = 8) -> List[str]:
    """
    从 DESCRIPTION 中抽取关键词（轻量），用于 anti-collision 约束。
    目标：提供“信号”，而不是复述全文。
    """
    text = _norm_text(desc)
    words = re.findall(r"[a-z]{3,}", text)
    kws: List[str] = []
    for w in words:
        if w in _STOPWORDS:
            continue
        if w.isdigit():
            continue
        if w not in kws:
            kws.append(w)
        if len(kws) >= max_keywords:
            break
    return kws

def summarize_generated_item(item: Dict[str, Any], max_keywords: int = 8) -> str:
    """
    生成一条 anti-collision 摘要：
    - BRAND
    - DESC_KEYWORDS（来自 DESCRIPTION）
    - CATEGORY（辅助模型做差异化）
    """
    brand = (item.get("BRAND") or "").strip()
    category = (item.get("CATEGORY") or "").strip()
    desc = (item.get("DESCRIPTION") or "").strip()
    kws = _extract_keywords(desc, max_keywords=max_keywords)
    kw_str = ", ".join(kws) if kws else ""
    # 保持简短
    return f'BRAND="{brand}" | CATEGORY="{category}" | DESC_KEYWORDS=[{kw_str}]'

def build_anticollision_block(summaries: List[str], top_k: int = 12) -> str:
    """
    将最近 top_k 条摘要拼成 prompt 的约束块。
    """
    if not summaries:
        return ""
    recent = summaries[-top_k:]
    lines = "\n".join([f"- {s}" for s in recent])
    return f"""
Already generated SKUs (anti-collision constraints):
{lines}

Constraints you MUST follow:
1) BRAND must NOT be identical or very similar to any BRAND listed above.
2) DESCRIPTION must NOT reuse the same phrasing pattern implied by DESC_KEYWORDS above.
3) To be different, vary at least 2 dimensions where applicable:
   - product type/form (e.g., chips vs bars), flavor/variety, packaging (bottle/can/box/bag),
     count/size wording, usage/occasion, material/function (for household items).
"""


# ===================== 分位数 / 价格统计 =====================

def _percentile(sorted_list: List[float], p: float) -> float:
    if not sorted_list:
        return 0.0
    p = min(max(p, 0.0), 1.0)
    n = len(sorted_list)
    if n == 1:
        return sorted_list[0]
    idx = p * (n - 1)
    low = int(idx)
    high = min(low + 1, n - 1)
    if low == high:
        return sorted_list[low]
    frac = idx - low
    return sorted_list[low] + frac * (sorted_list[high] - sorted_list[low])

def compute_price_stats(csv_path: str) -> Optional[Dict[str, float]]:
    prices: List[float] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                price = float(row.get("PRICE", "") or 0)
            except ValueError:
                continue
            if price <= 0:
                continue
            prices.append(price)

    if not prices:
        return None

    prices.sort()
    return {
        "price_min": min(prices),
        "price_max": max(prices),
        "price_p30": _percentile(prices, 0.3),
        "price_p70": _percentile(prices, 0.7),
    }


# ===================== CSV 遍历：sku -> category/path =====================

def list_csv_with_category(folder: str, recursive: bool = True) -> Dict[str, Dict[str, str]]:
    folder_path = Path(folder)
    mapping: Dict[str, Dict[str, str]] = {}

    csv_files = folder_path.rglob("*.csv") if recursive else folder_path.glob("*.csv")
    for csv_path in csv_files:
        sku_name = csv_path.stem
        rel = csv_path.relative_to(folder_path)
        parts = rel.parts

        # 你的原逻辑：上一级目录名（len>=3 -> parts[-3]）
        if len(parts) >= 3:
            category = parts[-3]
        elif len(parts) == 2:
            category = parts[-2]
        else:
            category = ""

        mapping[sku_name] = {"category": category, "path": str(csv_path)}
    return mapping


# ===================== LLM：单 SKU 生成（带 anti-collision 摘要） =====================

def call_llm_for_single_product(
    upc: str,
    info: Dict[str, Any],
    anti_summaries: List[str],
    model: str = MODEL_NAME,
    top_k: int = 12,
) -> Dict[str, Any]:
    """
    生成单个商品结构化 JSON，同时带入 Top-K anti-collision 摘要约束，减少跨 SKU 相似。
    """
    desc_raw = info.get("desc", "") or ""
    size = info.get("size", "") or ""
    com_code = info.get("com_code", "") or ""
    case = info.get("case", "") or ""
    nitem = info.get("nitem", "") or ""
    category = info.get("category", "") or ""

    anti_block = build_anticollision_block(anti_summaries, top_k=top_k)

    user_prompt = f"""
You are a retail product data generation expert.
Return STRICT JSON only (no explanations, no markdown).

{anti_block}

Input product info:
- UPC: {upc}
- DESCRIPTION_RAW: {desc_raw}
- SIZE: {size}
- COM_CODE: {com_code}
- CASE: {case}
- NITEM: {nitem}
- CATEGORY: {category}

Rules for PROMOTION_TIME:
1) Fresh Products: 7-10 days
2) Regular Food Products: 21-28 days
3) Frozen Food: 30-45 days
4) Daily Necessities: 30-45 days
5) Medicines / OTC: 60 days
Choose an integer days value.

Rules for DELEVERY_TIME:
- Output [min_days, max_days] as integers, reasonable for retail replenishment.

BRAND rules (strict):
1) Prefer extracting a real brand from DESCRIPTION_RAW if clearly present.
2) If you cannot confidently identify a real brand, you MUST create a specific, plausible English brand name:
   - 1 to 3 words (e.g., "RiverStone", "Oak Valley", "SunPeak")
   - Must look like a consumer brand (not a category, not a sentence)
   - Must NOT contain any of: "Generic", "Brand", "Store", "Private", "Label", "Unbranded", "Unknown"
   - Must NOT be overly broad (e.g., "Food", "Household", "OTC", "Pharmacy", "Grocery")
3) BRAND must be English and should be distinct from the brands in anti-collision constraints.

Output JSON schema (strict):
{{
  "UPC": "{upc}",
  "BRAND": "<English brand>",
  "DESCRIPTION": "<natural English product description>",
  "中文描述": "<Chinese translation of DESCRIPTION>",
  "PROMOTION_TIME": <integer>,
  "DELEVERY_TIME": [<int>, <int>]
}}

Hard constraints:
1) The JSON must be parseable by json.loads.
2) BRAND and DESCRIPTION should be realistic and not templated.
3) Follow the anti-collision constraints above.
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return strict JSON only."},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.6,
    )

    content = resp.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print("LLM JSON decode failed, raw content:")
        print(content)
        raise


def generate_upc_data_with_anticollision(
    upc_records: List[Dict[str, Any]],
    limit: int = 300,
    model: str = MODEL_NAME,
    top_k: int = 12,
) -> List[Dict[str, Any]]:
    """
    顺序生成：每生成一个 SKU，就把摘要加入 anti_summaries，影响后续 SKU。
    这是“第一次生成就避免跨 SKU 相似”的关键。
    """
    results: List[Dict[str, Any]] = []
    anti_summaries: List[str] = []

    raw_items = upc_records[:limit]
    total = len(raw_items)

    for idx, d in enumerate(raw_items):
        upc = (d.get("UPC") or d.get("upc") or "").strip()
        if not upc:
            continue

        info = {
            "com_code": (d.get("COM_CODE") or "").strip(),
            "upc": upc,
            "desc": (d.get("DESCRIP") or "").strip(),
            "size": (d.get("SIZE") or "").strip(),
            "case": (d.get("CASE") or "").strip(),
            "nitem": (d.get("NITEM") or "").strip(),
            "category": (d.get("CATEGORY") or "").strip(),
        }

        print(f"Processing {idx + 1}/{total} UPC={upc} ...")

        result = call_llm_for_single_product(
            upc=upc,
            info=info,
            anti_summaries=anti_summaries,
            model=model,
            top_k=top_k,
        )

        # 强制写入 UPC / CATEGORY
        result["UPC"] = upc
        result["CATEGORY"] = info.get("category", "")

        # 额外携带价格统计（如果有）
        for key in ("price_min", "price_max", "price_p30", "price_p70"):
            if key in d and d[key] is not None:
                result[key] = d[key]

        results.append(result)

        # 更新 anti-collision 摘要（只用生成结果本身，不塞全文）
        anti_summaries.append(summarize_generated_item(result, max_keywords=8))

    return results


# ===================== 主流程 =====================

def main() -> None:
    print("Starting UPC data generation (anti-collision enabled)...")

    store_record_path = "/Users/linghuazhang/Desktop/Project/RetailBench/data/simulate_data/15"
    upc_path = "/Users/linghuazhang/Desktop/Project/RetailBench/data/upc/upc.json"
    output_file = "/Users/linghuazhang/Desktop/Project/RetailBench/data/simulate_data/15/upc.json"

    # 1) sku 列表 + category 映射
    sku_meta = list_csv_with_category(store_record_path, recursive=True)
    sku_names = list(sku_meta.keys())
    print(f"Found {len(sku_names)} SKU CSV files in store_record.")

    # 2) 加载 upc.json
    upc_json = load_json(upc_path)

    # 3) 组装原始数据（补 CATEGORY + price stats）
    sku_upc_jsons: List[Dict[str, Any]] = []
    for sku_name in sku_names:
        if sku_name not in upc_json:
            continue

        record = upc_json[sku_name]

        # 价格统计
        csv_path = sku_meta[sku_name]["path"]
        price_stats = compute_price_stats(csv_path)
        if price_stats:
            record.update(price_stats)

        record["CATEGORY"] = sku_meta.get(sku_name, {}).get("category", "")
        sku_upc_jsons.append(record)

    # 4) 顺序生成（第一次就避免跨 SKU 相似）
    print("Generating complete UPC data with anti-collision constraints...")
    upc_data = generate_upc_data_with_anticollision(
        sku_upc_jsons,
        limit=200,
        model=MODEL_NAME,
        top_k=12,  # 只取最近 12 条摘要
    )

    # 5) 写 JSON
    print(f"Saving results to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(upc_data, f, indent=2, ensure_ascii=False)

    print(f"Successfully saved {len(upc_data)} UPC entries to {output_file}")


if __name__ == "__main__":
    main()