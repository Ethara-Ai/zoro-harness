from datetime import datetime
import json
import math
import random
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import matplotlib.pyplot as plt

# ---------- 配置区 ----------

# 根目录：你的 data 所在路径（按实际改）
DATA_ROOT = Path("/Users/linghuazhang/Desktop/Project/RetailBench/data/still/simulate_data/15")

# now_start_point 的 JSON 路径（按你实际文件名改）
NOW_START_JSON = Path("/Users/linghuazhang/Desktop/Project/RetailBench/data/still/review/simulated_ratings.json")

# 每个商品生成多少个供给商
N_SUPPLIERS_PER_PRODUCT = 5
# 是否在已有供应商中随机选一个作为“劣质低价”供应商（质量极低、价格极低）
INCLUDE_BAD_SUPPLIER = True

# 档位变动周期范围（天），每个 SKU 随机采样一个周期
CYCLE_DAYS_RANGE = (100000, 100000000)

# 随机种子，保证复现性，想要每次都不一样可以删掉这一行
random.seed(42)

# 相关系数范围不符时的最大重采样次数
MAX_RESAMPLE_ATTEMPTS = 50

# 价格档位（乘数）
PRICE_TIERS: Dict[int, Tuple[float, float]] = {
    1: (1.0, 1.05),
    2: (0.9, 1.0),
    3: (0.8, 0.9),
    4: (0.7, 0.8),
    5: (0.6, 0.7),
}

# 质量档位，基于 now_start_point 的偏移
QUALITY_OFFSETS: Dict[int, Tuple[float, float]] = {
    1: (0.2, 0.3),
    2: (0.1, 0.2),
    3: (0, 0.1),
    4: (-1.0, 0),
    5: (-2.0, -1.0),
    6: (-2.5, -2.0),
    7: (-3.0, -2.5),
}

# 低价低质档位设定
BAD_PRICE_TIER = 5  # 使用最低价格档 (0.6~0.7)
BAD_QUALITY_TIER = max(QUALITY_OFFSETS.keys())  # 使用最差质量档
# 高质档（价格可高可低，随机档位）
GOOD_QUALITY_TIER = min(QUALITY_OFFSETS.keys())
GOOD_PRICE_TIERS = [1, 2, 3, 4, 5]

# 运输时间范围（天），为每个供应商生成一个 [min, max]
TRANSPORT_DAYS_RANGE = (2, 7)

# 食物/非食物品类，用于控制价格-质量相关性
FOOD_CATEGORIES = {
    "Bottled Juices",
    "Cereals",
    "Cookies",
    "Crackers",
    "Cheeses",
    "Canned Soup",
    "Canned Tuna",
    "Front-end-candies",
    "Frozen Entrees",
    "Frozen Juices",
    "Snack Crackers",
    "Soft Drinks",
}

NON_FOOD_CORR_RANGE = (0.4, 0.5)  # 非食物：相关性控制在这一区间
FOOD_CORR_RANGE = (0.2, 0.3)      # 食物：相关性控制在这一区间


# ---------- 工具函数 ----------

def norm_cdf(x: float) -> float:
    """标准正态分布 CDF（math.erf 实现）。"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


QUALITY_TIERS = sorted(QUALITY_OFFSETS.keys())  # 1~7


def quality_tier_from_uniform(u: float) -> int:
    """
    将 [0,1) 均匀分位映射到质量档位（当前 7 档）。
    """
    u = max(0.0, min(0.999999, u))
    idx = int(math.floor(u * len(QUALITY_TIERS)))
    idx = min(idx, len(QUALITY_TIERS) - 1)
    return QUALITY_TIERS[idx]


def price_tier_from_uniform(u: float) -> int:
    """
    将 [0,1) 均匀分位映射到价格档位 1~5。
    单独函数方便后续调整价格档位边界。
    """
    u = max(0.0, min(0.999999, u))
    return int(math.floor(u * 5)) + 1


def is_food_category(category: str) -> bool:
    """判断类别是否归为食品/快消品。"""
    return category in FOOD_CATEGORIES


def sample_correlated_tiers(category: str, rng: random.Random) -> Tuple[int, int]:
    """
    为单个供给商生成 (quality_tier, price_tier)，
    使用高斯 copula 控制相关性：食品更弱相关，非食品更强相关。
    """
    corr_low, corr_high = FOOD_CORR_RANGE if is_food_category(category) else NON_FOOD_CORR_RANGE
    rho = rng.uniform(corr_low, corr_high)

    z1 = rng.gauss(0, 1)
    z2 = rho * z1 + math.sqrt(max(0.0, 1 - rho ** 2)) * rng.gauss(0, 1)

    q_tier = quality_tier_from_uniform(norm_cdf(z1))
    p_tier_base = price_tier_from_uniform(norm_cdf(z2))

    # 直接使用 copula 生成的价格档位，可在此叠加额外业务规则（若需要）
    p_tier = max(1, min(5, p_tier_base))

    return q_tier, p_tier


def sample_quality(now_start: float, q_tier: int, rng: Optional[random.Random] = None) -> float:
    """按质量档位，从对应区间采样质量分数"""
    rng = rng or random
    offset_low, offset_high = QUALITY_OFFSETS.get(q_tier, QUALITY_OFFSETS[max(QUALITY_OFFSETS)])
    raw = now_start + rng.uniform(offset_low, offset_high)
    # 限制在 [1, 5] 之间
    return max(1.0, min(5.0, raw))


def sample_supplier_price(base_price: float, p_tier: int, rng: Optional[random.Random] = None) -> float:
    """根据价格档位，从乘数区间采样供给商价格"""
    rng = rng or random
    low, high = PRICE_TIERS[p_tier]
    factor = rng.uniform(low, high)
    return base_price * factor


def compute_correlation(xs: List[float], ys: List[float]) -> Optional[float]:
    """
    简单皮尔逊相关系数；数据不足或方差为 0 时返回 None。
    """
    n = min(len(xs), len(ys))
    if n < 2:
        return None
    xs = xs[:n]
    ys = ys[:n]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return None
    return cov / math.sqrt(var_x * var_y)


def plot_corr_distribution(corrs: List[float], output_path: Path) -> None:
    """绘制相关系数分布并保存。"""
    if not corrs:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(corrs, bins=20, color="steelblue", edgecolor="black", alpha=0.8)
    plt.axvline(sum(corrs) / len(corrs), color="red", linestyle="--", linewidth=1, label="mean")
    plt.xlabel("corr(price, quality)")
    plt.ylabel("count")
    plt.title("Supplier Price-Quality Correlation Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] 相关系数分布图已保存: {output_path}")


def plot_supplier_tiers(records: Dict[str, List[Dict[str, float]]], output_path: Path) -> None:
    """
    绘制单个 SKU 随日期的“最优质量供应商”和“最差质量且最低价供应商”时间线。
    y 轴为供应商编号，便于查看每天是谁占据这两个位置。
    """
    supplier_keys = [k for k in records.keys() if not k.startswith("_")]
    if not supplier_keys:
        return

    # 将数据聚合到日期维度：每个日期记录所有供应商的数据
    date_records: Dict[datetime, List[Dict[str, float]]] = {}
    for sid in supplier_keys:
        for row in records.get(sid, []):
            date_str = row.get("date")
            if not date_str:
                continue
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d").date()
            except Exception:
                continue
            date_records.setdefault(dt, []).append(row)

    if not date_records:
        return

    def supplier_index(supplier_id: str) -> int:
        # supplier_1 -> 1 等
        try:
            return int(supplier_id.split("_")[-1])
        except Exception:
            return 0

    best_points: List[Tuple[datetime, int]] = []
    worst_points: List[Tuple[datetime, int]] = []

    for dt, rows in sorted(date_records.items(), key=lambda x: x[0]):
        best_candidates = [r for r in rows if r.get("quality_tier") == GOOD_QUALITY_TIER]
        worst_candidates = [
            r
            for r in rows
            if r.get("quality_tier") == BAD_QUALITY_TIER and r.get("price_tier") == BAD_PRICE_TIER
        ]

        # 若缺少严格命中的档位，则退化为“最好质量”或“最差质量”
        if not best_candidates:
            best_candidates = rows  # 用所有供应商里质量最佳的
        if not worst_candidates:
            worst_candidates = rows  # 用所有供应商里质量最差的

        # 最佳：优先质量档最低，其次质量分最高，再次价格最低
        best_row = min(
            best_candidates,
            key=lambda r: (
                r.get("quality_tier", float("inf")),
                -r.get("quality_score", -float("inf")),
                r.get("supplier_price", float("inf")),
            ),
        )
        # 最差：优先质量档最高，其次价格最低
        worst_row = min(
            worst_candidates,
            key=lambda r: (
                -r.get("quality_tier", -float("inf")),
                r.get("supplier_price", float("inf")),
            ),
        )
        best_points.append((dt, supplier_index(best_row.get("supplier_id", ""))))
        worst_points.append((dt, supplier_index(worst_row.get("supplier_id", ""))))

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    axes[0].plot([d for d, _ in best_points], [v for _, v in best_points], marker="o", linestyle="-", color="green")
    axes[1].plot([d for d, _ in worst_points], [v for _, v in worst_points], marker="o", linestyle="-", color="red")

    axes[0].set_title("Best Quality Supplier (by day)")
    axes[1].set_title("Worst Quality & Lowest Price Supplier (by day)")
    axes[1].set_xlabel("date")
    axes[0].set_ylabel("supplier #")
    axes[1].set_ylabel("supplier #")
    axes[0].set_yticks(range(1, N_SUPPLIERS_PER_PRODUCT + 1))
    axes[1].set_yticks(range(1, N_SUPPLIERS_PER_PRODUCT + 1))
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[1].grid(True, linestyle="--", alpha=0.3)
    fig.autofmt_xdate()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"[INFO] 供应商档位图已保存: {output_path}")


def plot_best_worst_tiers(records: Dict[str, List[Dict[str, float]]], output_path: Path) -> None:
    """
    绘制“最好/最差”供应商的质量档与价格档随时间变化。
    - 质量档：best/worst 的 quality_tier
    - 价格档：best/worst 的 price_tier
    """
    supplier_keys = [k for k in records.keys() if not k.startswith("_")]
    if not supplier_keys:
        return

    date_records: Dict[datetime, List[Dict[str, float]]] = {}
    for sid in supplier_keys:
        for row in records.get(sid, []):
            date_str = row.get("date")
            if not date_str:
                continue
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d").date()
            except Exception:
                continue
            date_records.setdefault(dt, []).append(row)

    if not date_records:
        return

    best_q: List[Tuple[datetime, int]] = []
    worst_q: List[Tuple[datetime, int]] = []
    best_p: List[Tuple[datetime, int]] = []
    worst_p: List[Tuple[datetime, int]] = []

    for dt, rows in sorted(date_records.items(), key=lambda x: x[0]):
        best_candidates = [r for r in rows if r.get("quality_tier") == GOOD_QUALITY_TIER]
        worst_candidates = [
            r
            for r in rows
            if r.get("quality_tier") == BAD_QUALITY_TIER and r.get("price_tier") == BAD_PRICE_TIER
        ]

        if not best_candidates:
            best_candidates = rows
        if not worst_candidates:
            worst_candidates = rows

        best_row = min(
            best_candidates,
            key=lambda r: (
                r.get("quality_tier", float("inf")),
                -r.get("quality_score", -float("inf")),
                r.get("supplier_price", float("inf")),
            ),
        )
        worst_row = min(
            worst_candidates,
            key=lambda r: (
                -r.get("quality_tier", -float("inf")),
                r.get("supplier_price", float("inf")),
            ),
        )

        best_q.append((dt, int(best_row.get("quality_tier", 0))))
        worst_q.append((dt, int(worst_row.get("quality_tier", 0))))
        best_p.append((dt, int(best_row.get("price_tier", 0))))
        worst_p.append((dt, int(worst_row.get("price_tier", 0))))

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    axes[0].plot([d for d, _ in best_q], [v for _, v in best_q], marker="o", color="green", label="best quality")
    axes[0].plot([d for d, _ in worst_q], [v for _, v in worst_q], marker="o", color="red", label="worst quality")
    axes[1].plot([d for d, _ in best_p], [v for _, v in best_p], marker="o", color="green", linestyle="--", label="best price tier")
    axes[1].plot([d for d, _ in worst_p], [v for _, v in worst_p], marker="o", color="red", linestyle="--", label="worst price tier")

    axes[0].set_ylabel("quality tier")
    axes[1].set_ylabel("price tier")
    axes[1].set_xlabel("date")
    axes[0].set_yticks(sorted(QUALITY_OFFSETS.keys()))
    axes[1].set_yticks(sorted(PRICE_TIERS.keys()))
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")
    fig.autofmt_xdate()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"[INFO] 最好/最差档位图已保存: {output_path}")


def plot_supplier_tier_fluctuations(records: Dict[str, List[Dict[str, float]]], base_path: Path) -> None:
    """
    绘制单个 SKU 下每个供应商的质量档 / 价格档随时间波动，每个供应商单独一张图。
    """
    supplier_keys = [k for k in records.keys() if not k.startswith("_")]
    if not supplier_keys:
        return

    for sid in sorted(supplier_keys):
        series: List[Tuple[datetime, int, int]] = []
        for row in records.get(sid, []):
            date_str = row.get("date")
            if not date_str:
                continue
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d").date()
            except Exception:
                continue
            q_tier = row.get("quality_tier")
            p_tier = row.get("price_tier")
            if q_tier is None or p_tier is None:
                continue
            series.append((dt, int(q_tier), int(p_tier)))
        series.sort(key=lambda x: x[0])

        if not series:
            continue

        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
        dates = [s[0] for s in series]
        q_tiers = [s[1] for s in series]
        p_tiers = [s[2] for s in series]

        axes[0].plot(dates, q_tiers, marker="o", color="steelblue", linewidth=1.2)
        axes[1].plot(dates, p_tiers, marker="o", color="darkorange", linewidth=1.2)

        axes[0].set_title(f"{sid} quality tier over time")
        axes[1].set_title(f"{sid} price tier over time")
        axes[1].set_xlabel("date")
        axes[0].set_ylabel("quality tier")
        axes[1].set_ylabel("price tier")
        axes[0].set_yticks(sorted(QUALITY_OFFSETS.keys()))
        axes[1].set_yticks(sorted(PRICE_TIERS.keys()))
        axes[0].grid(True, linestyle="--", alpha=0.3)
        axes[1].grid(True, linestyle="--", alpha=0.3)
        fig.autofmt_xdate()
        output_path = base_path.with_name(f"{base_path.stem}_{sid}_fluctuations.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
        print(f"[INFO] 供应商波动图已保存: {output_path}")


# ---------- 主逻辑 ----------

def load_now_start_points(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def process_daily_file(
    daily_path: Path,
    store_id: str,
    category: str,
    now_start_points,
) -> Optional[float]:
    upc = daily_path.stem.replace("_daily", "")

    # 取 now_start_point，如果没有就跳过
    try:
        now_start = now_start_points[str(store_id)][category][upc]
    except KeyError:
        print(f"[WARN] now_start_point 缺失，跳过: store={store_id}, category={category}, upc={upc}")
        return None

    with daily_path.open("r", encoding="utf-8") as f:
        daily_data = json.load(f)
    # 按日期排序，确保周期分段正确
    def _parse_dt(val: str):
        try:
            return datetime.strptime(val, "%m/%d/%y").date()
        except Exception:
            return None
    daily_data = [row for row in daily_data if row.get("date")]
    daily_data.sort(key=lambda r: _parse_dt(r["date"]) or datetime.min.date())

    target_range = FOOD_CORR_RANGE if is_food_category(category) else NON_FOOD_CORR_RANGE

    def simulate_once(seed_offset: int) -> Tuple[Dict[str, List[Dict[str, float]]], Optional[float]]:
        """
        单次尝试：为当前 SKU 生成供应商记录，并返回相关系数。
        """
        rng = random.Random(seed_offset)

        # 解析开始日期，用于周期分段；为该 SKU 随机周期
        valid_dates = [row.get("date") for row in daily_data if row.get("date")]
        if not valid_dates:
            return {}, None
        try:
            start_dt = min(datetime.strptime(d, "%Y-%m-%d").date() for d in valid_dates)
        except Exception:
            start_dt = None

        supplier_cycle_days = {s_idx: max(1, rng.randint(*CYCLE_DAYS_RANGE)) for s_idx in range(N_SUPPLIERS_PER_PRODUCT)}
        supplier_cycle_tiers: Dict[int, Dict[int, Tuple[int, int]]] = {}
        supplier_transport_days: Dict[int, Tuple[int, int]] = {}
        for s_idx in range(N_SUPPLIERS_PER_PRODUCT):
            a, b = rng.randint(*TRANSPORT_DAYS_RANGE), rng.randint(*TRANSPORT_DAYS_RANGE)
            low, high = sorted((a, b))
            if low == high:
                if high < TRANSPORT_DAYS_RANGE[1]:
                    high += 1
                elif low > TRANSPORT_DAYS_RANGE[0]:
                    low -= 1
            supplier_transport_days[s_idx] = (low, high)

        output_records = {f"supplier_{i+1}": [] for i in range(N_SUPPLIERS_PER_PRODUCT)}
        all_prices: List[float] = []
        all_qualities: List[float] = []

        for row in daily_data:
            base_price = row.get("smoothed_cost_price")
            date = row.get("date")

            if base_price is None or date is None:
                continue

            try:
                dt = datetime.strptime(date, "%Y-%m-%d").date()
            except Exception:
                continue

            day_cycle_idx: Dict[int, int] = {}
            for s_idx in range(N_SUPPLIERS_PER_PRODUCT):
                cycle_days = supplier_cycle_days[s_idx]
                cycle_idx = 0
                if start_dt is not None:
                    cycle_idx = (dt - start_dt).days // cycle_days

                day_cycle_idx[s_idx] = cycle_idx
                # 为每个周期重新抽样档位
                supplier_cycle_tiers.setdefault(s_idx, {})
                if cycle_idx not in supplier_cycle_tiers[s_idx]:
                    supplier_cycle_tiers[s_idx][cycle_idx] = sample_correlated_tiers(category, rng)

            # 确保每天至少出现一个“低价低质”与一个“高质量”供应商，且两者不互相覆盖
            daily_tiers = {s_idx: supplier_cycle_tiers[s_idx][day_cycle_idx[s_idx]] for s_idx in range(N_SUPPLIERS_PER_PRODUCT)}
            bad_indices = [
                idx for idx, (q_tier, p_tier) in daily_tiers.items() if q_tier == BAD_QUALITY_TIER and p_tier == BAD_PRICE_TIER
            ]
            good_indices = [idx for idx, (q_tier, _) in daily_tiers.items() if q_tier == GOOD_QUALITY_TIER]

            bad_idx = bad_indices[0] if bad_indices else None
            if INCLUDE_BAD_SUPPLIER and bad_idx is None:
                bad_idx = rng.randrange(N_SUPPLIERS_PER_PRODUCT)
                bad_tier = (BAD_QUALITY_TIER, BAD_PRICE_TIER)
                daily_tiers[bad_idx] = bad_tier
                supplier_cycle_tiers[bad_idx][day_cycle_idx[bad_idx]] = bad_tier

            good_idx = good_indices[0] if good_indices else None

            if good_idx is None:
                candidate_indices = [i for i in range(N_SUPPLIERS_PER_PRODUCT) if i != bad_idx]
                if not candidate_indices:
                    raise ValueError("缺少可用的供应商索引来设置高质量供应商")
                good_idx = rng.choice(candidate_indices)
                good_tier = (GOOD_QUALITY_TIER, rng.choice(GOOD_PRICE_TIERS))
                daily_tiers[good_idx] = good_tier
                supplier_cycle_tiers[good_idx][day_cycle_idx[good_idx]] = good_tier

            if bad_idx is None or good_idx is None:
                raise ValueError("缺少可用的供应商索引来设置高质量与低质量供应商")
            
            good_tier = daily_tiers[good_idx]
            bad_tier = daily_tiers[bad_idx]
            
            if good_tier[0] == GOOD_QUALITY_TIER or bad_tier[0] == BAD_QUALITY_TIER:
                pass
            else:
                print(
                    good_tier,
                    bad_tier,
                    "采样失败"
                )


            for s_idx in range(N_SUPPLIERS_PER_PRODUCT):
                cycle_idx = day_cycle_idx[s_idx]
                cycle_days = supplier_cycle_days[s_idx]
                q_tier, p_tier = daily_tiers[s_idx]

                q_score = sample_quality(now_start, q_tier, rng)
                supplier_price = sample_supplier_price(base_price, p_tier, rng)

                record = {
                    "store": store_id,
                    "category": category,
                    "upc": upc,
                    "date": date,
                    "cycle_idx": cycle_idx,
                    "cycle_days": cycle_days,
                    "base_cost_price": base_price,
                    "now_start_point": now_start,
                    "supplier_id": f"supplier_{s_idx + 1}",
                    "quality_tier": q_tier,
                    "quality_score": round(q_score, 4),
                    "price_tier": p_tier,
                    "supplier_price": round(supplier_price, 6),
                    "transport_days_min": supplier_transport_days[s_idx][0],
                    "transport_days_max": supplier_transport_days[s_idx][1],
                }

                supplier_id = f"supplier_{s_idx + 1}"
                output_records[supplier_id].append(record)

                all_prices.append(record["supplier_price"])
                all_qualities.append(record["quality_score"])

            

        # 验证每个日期都存在“最好质量档”与“最差质量且最低价”供应商，否则视为无效采样
        by_date: Dict[str, List[Dict[str, float]]] = {}
        for sid, rows in output_records.items():
            if sid.startswith("_"):
                continue
            for rec in rows:
                by_date.setdefault(rec["date"], []).append(rec)

        count = 0
        for date, rows in by_date.items():
            has_best = any(r.get("quality_tier") == GOOD_QUALITY_TIER for r in rows)
            has_worst = any(
                r.get("quality_tier") == BAD_QUALITY_TIER and r.get("price_tier") == BAD_PRICE_TIER
                for r in rows
            )
            if not has_best or not has_worst:
                count += 1
                if count < 20:
                    continue
                else:
                    print(f"[INFO] 无效采样：缺少必要供应商 date={date} best={has_best} worst={has_worst}")
                    return {}, None


        corr = compute_correlation(all_prices, all_qualities)
        output_records["_corr_price_quality"] = corr
        output_records["_cycle_days_per_supplier"] = supplier_cycle_days
        output_records["_cycle_ranges_per_supplier"] = {
            f"supplier_{idx+1}": {"cycle_days": days} for idx, days in supplier_cycle_days.items()
        }
        output_records["_transport_days_per_supplier"] = {
            f"supplier_{idx+1}": {"min": t[0], "max": t[1]} for idx, t in supplier_transport_days.items()
        }
        return output_records, corr

    # --------- 重采样：保留“最接近目标区间”的那次结果 ---------

    base_seed = hash((store_id, category, upc)) & 0xffffffff

    best_records: Optional[Dict[str, List[Dict[str, float]]]] = None
    best_corr: Optional[float] = None
    best_dist: float = float("inf")

    for attempt in range(MAX_RESAMPLE_ATTEMPTS):
        records, corr = simulate_once(base_seed + attempt)
        if corr is None:
            # 这次采样没法算相关系数，直接跳过
            print(f"[INFO] {daily_path.name} attempt={attempt + 1} corr=None, 跳过")
            continue

        # 计算和目标区间的“距离”：在区间内就是 0，区间外是到最近边界的差值
        if target_range[0] <= corr <= target_range[1]:
            dist = 0.0
        elif corr < target_range[0]:
            dist = target_range[0] - corr
        else:
            dist = corr - target_range[1]

        # 如果更接近目标区间，就更新“最优解”
        if dist < best_dist:
            best_dist = dist
            best_corr = corr
            best_records = records

        hit_str = "HIT" if dist == 0.0 else "MISS"
        print(
            f"[{hit_str}] {daily_path.name} attempt={attempt + 1}, "
            f"corr={corr:.4f}, dist={dist:.4f}, target={target_range}"
        )

        # 提前结束：已经命中区间，没必要继续尝试
        if dist == 0.0:
            break

    # 所有尝试都没拿到任何 corr（极端情况）
    if best_records is None or best_corr is None:
        print(f"[WARN] {daily_path} 所有尝试均无法计算相关系数，已跳过写文件")
        return None

    # 输出到同目录下 upc_suppliers.json，使用“最优”那次记录
    out_path = daily_path.with_name(f"{upc}_suppliers.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(best_records, f, ensure_ascii=False, indent=2)

    # 绘制供应商档位随时间变化图
    tier_plot_path = daily_path.with_name(f"{upc}_tiers.png")
    plot_supplier_tiers(best_records, tier_plot_path)
    tier_fluctuation_base = daily_path.with_name(f"{upc}_tier_fluctuations.png")
    plot_supplier_tier_fluctuations(best_records, tier_fluctuation_base)
    best_worst_tier_path = daily_path.with_name(f"{upc}_best_worst_tiers.png")
    plot_best_worst_tiers(best_records, best_worst_tier_path)

    in_range = target_range[0] <= best_corr <= target_range[1]
    print(
        f"[OK] 生成供给商价格: {out_path} | "
        f"目标区间: {target_range} | "
        f"最佳相关系数: {best_corr:.4f} | "
        f"命中区间: {in_range} | "
        f"最小距离: {best_dist:.4f}"
    )

    return best_corr


def main():
    now_start_points = load_now_start_points(NOW_START_JSON)

    corrs: List[float] = []

    # 遍历所有 *_daily.json 文件，不过滤任何目录
    for daily_file in DATA_ROOT.rglob("*_daily.json"):

        # 解析 store_id / category：假设目录格式中这两层始终存在
        parts = daily_file.parts  # 例: ['data','filtered_post_data','12','Analgesics','filtered_middle_data','xxxx_daily.json']

        try:
            # 自动寻找 store_id（数字）
            store_id = next(p for p in parts if p.isdigit())

            # category = store_id 下一层目录
            store_index = parts.index(store_id)
            category = parts[store_index + 1]
            category = category.replace("_", " ")

        except Exception:
            print(f"[WARN] 路径无法解析 store/category, 已跳过: {daily_file}")
            continue

        corr = process_daily_file(
            daily_path=daily_file,
            store_id=store_id,
            category=category,
            now_start_points=now_start_points,
        )
        if corr is not None:
            corrs.append(corr)

    if corrs:
        plot_corr_distribution(corrs, Path("corr_distribution.png"))
        print(f"[INFO] 共生成 {len(corrs)} 条相关系数，均值 {sum(corrs)/len(corrs):.4f}")


if __name__ == "__main__":
    main()
