#!/usr/bin/env python
"""
统计 news_merged.jsonl 的分布：
1) 按 mode 统计数量
2) 影响方向 / 影响因子 分布
3) 影响强度分布
4) 被影响的品类分布 (target_category)
5) 被影响的 SKU 分布 (target_sku_upc)

运行:
    python script/news_impact_distribution.py \
        --file data/simulate_data/15/news_merged.jsonl \
        --top 20 \
        [--plot-dir plots_news]
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def load_news(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description="News impact distribution summary.")
    parser.add_argument("--file", default="data/simulate_data/15/news_merged.jsonl", help="Path to news JSONL file.")
    parser.add_argument("--top", type=int, default=20, help="Top N categories/SKUs to show.")
    parser.add_argument("--plot-dir", default=None, help="Directory to save plots (optional).")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")

    mode_cnt = Counter()
    dir_factor_cnt = Counter()
    category_cnt = Counter()
    sku_cnt = Counter()
    strength_values = []
    
    # 按 mode 分组的数据
    direction_by_mode: dict[str, list[str]] = {}
    strength_by_mode: dict[str, list[float]] = {}
    direction_cnt_by_mode: dict[str, Counter] = {}

    for obj in load_news(path):
        mode = str(obj.get("mode") or obj.get("IMPACT_SCOPE") or obj.get("MODE") or "unknown").lower()
        direction = str(obj.get("impact_direction") or obj.get("IMPACT_DIRECTION") or "").lower() or "none"
        factor = str(obj.get("impact_factor") or obj.get("IMPACT_FACTOR") or "").lower() or "none"
        cat = obj.get("target_category") or obj.get("CATEGORY") or obj.get("SKU_CATEGORY")
        sku = obj.get("target_sku_upc") or obj.get("SKU_UPC")
        strength = obj.get("impact_strength") or obj.get("IMPACT_STRENGTH")

        mode_cnt[mode] += 1
        dir_factor_cnt[(direction, factor)] += 1
        if cat:
            category_cnt[str(cat)] += 1
        if sku:
            sku_cnt[str(sku)] += 1
        try:
            if strength is not None:
                strength_float = float(strength)
                strength_values.append(strength_float)
                # 按 mode 分组记录 strength
                if mode not in strength_by_mode:
                    strength_by_mode[mode] = []
                strength_by_mode[mode].append(strength_float)
        except Exception:
            pass
        
        # 按 mode 分组记录 direction
        if mode not in direction_by_mode:
            direction_by_mode[mode] = []
            direction_cnt_by_mode[mode] = Counter()
        direction_by_mode[mode].append(direction)
        direction_cnt_by_mode[mode][direction] += 1

    def print_counter(title, counter, top=None):
        print(f"\n{title}:")
        items = counter.most_common(top)
        for key, cnt in items:
            if isinstance(key, tuple):
                print(f"{key[0]:12s} | {key[1]:12s} | {cnt}")
            else:
                print(f"{key:20s} | {cnt}")

    print_counter("Mode 分布", mode_cnt, top=None)
    print_counter("影响方向/因子 分布", dir_factor_cnt, top=None)
    if strength_values:
        strength_values.sort()
        n = len(strength_values)
        min_v = strength_values[0]
        max_v = strength_values[-1]
        p50 = strength_values[int(0.5 * (n - 1))]
        p90 = strength_values[int(0.9 * (n - 1))]
        p99 = strength_values[int(0.99 * (n - 1))]
        print("\n影响强度分布:")
        print(f"count={n}, min={min_v:.4f}, p50={p50:.4f}, p90={p90:.4f}, p99={p99:.4f}, max={max_v:.4f}")
    else:
        print("\n影响强度分布: 无有效数据")
    print_counter(f"品类分布 (Top {args.top})", category_cnt, top=args.top)
    print_counter(f"SKU 分布 (Top {args.top})", sku_cnt, top=args.top)

    # ------------- 绘图 -------------
    if args.plot_dir:
        try:
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
            import matplotlib.pyplot as plt  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("需要 matplotlib 以生成图表，请先 pip install matplotlib") from exc

        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

        # mode 饼图
        if mode_cnt:
            labels, sizes = zip(*mode_cnt.most_common())
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
            ax.set_title("Mode 分布")
            fig.savefig(plot_dir / "mode_distribution.png", dpi=150)
            plt.close(fig)

        # 影响方向/因子 堆叠条形图
        if dir_factor_cnt:
            labels = [f"{d}/{f}" for (d, f), _ in dir_factor_cnt.most_common()]
            values = [v for _, v in dir_factor_cnt.most_common()]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(range(len(values)), values)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_title("影响方向/因子 分布")
            fig.tight_layout()
            fig.savefig(plot_dir / "direction_factor_distribution.png", dpi=150)
            plt.close(fig)

        # 影响强度直方图
        if strength_values:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(strength_values, bins=40, color="steelblue", edgecolor="white")
            ax.set_title("影响强度分布")
            ax.set_xlabel("impact_strength")
            ax.set_ylabel("count")
            fig.tight_layout()
            fig.savefig(plot_dir / "impact_strength_hist.png", dpi=150)
            plt.close(fig)

        # 品类 TopN 条形图
        if category_cnt:
            cat_items = category_cnt.most_common(args.top)
            labels, values = zip(*cat_items)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(range(len(values)), values)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_title(f"品类分布 Top {args.top}")
            fig.tight_layout()
            fig.savefig(plot_dir / "category_top.png", dpi=150)
            plt.close(fig)

        # SKU TopN 条形图
        if sku_cnt:
            sku_items = sku_cnt.most_common(args.top)
            labels, values = zip(*sku_items)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(range(len(values)), values)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_title(f"SKU 分布 Top {args.top}")
            fig.tight_layout()
            fig.savefig(plot_dir / "sku_top.png", dpi=150)
            plt.close(fig)

        # ========== 新增：按 mode 分组的可视化 ==========
        try:
            # 1. 各个 mode 下 impact direction 的分布
            if direction_cnt_by_mode:
                # 找出所有出现过的 direction
                all_directions = set()
                for cnt in direction_cnt_by_mode.values():
                    all_directions.update(cnt.keys())
                all_directions = sorted(list(all_directions))

                # 为每个 mode 创建子图
                modes_list = sorted(direction_cnt_by_mode.keys())
                n_modes = len(modes_list)

                if n_modes > 0:
                    # 计算子图布局
                    cols = min(3, n_modes)
                    rows = (n_modes + cols - 1) // cols

                    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
                    if n_modes == 1:
                        axes = [axes]
                    else:
                        axes = axes.flatten()

                    for idx, mode in enumerate(modes_list):
                        cnt = direction_cnt_by_mode[mode]
                        directions = [d for d in all_directions if cnt[d] > 0]
                        counts = [cnt[d] for d in directions]

                        ax = axes[idx]
                        bars = ax.bar(range(len(directions)), counts, color="steelblue", edgecolor="white")
                        ax.set_xticks(range(len(directions)))
                        ax.set_xticklabels(directions, rotation=45, ha="right")
                        ax.set_title(f"Mode: {mode}\nDirection Distribution")
                        ax.set_ylabel("Count")
                        ax.grid(axis="y", alpha=0.3)

                        # 在柱子上显示数值
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(
                                bar.get_x() + bar.get_width() / 2.0,
                                height,
                                f"{int(height)}",
                                ha="center",
                                va="bottom",
                                fontsize=8,
                            )

                    # 隐藏多余的子图
                    for idx in range(n_modes, len(axes)):
                        axes[idx].axis("off")

                    fig.tight_layout()
                    fig.savefig(plot_dir / "direction_by_mode.png", dpi=150)
                    plt.close(fig)

            # 2. 各个 mode 下 impact strength 的分布
            if strength_by_mode:
                modes_with_strength = {k: v for k, v in strength_by_mode.items() if len(v) > 0}

                if modes_with_strength:
                    n_modes = len(modes_with_strength)
                    cols = min(3, n_modes)
                    rows = (n_modes + cols - 1) // cols

                    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
                    if n_modes == 1:
                        axes = [axes]
                    else:
                        axes = axes.flatten()

                    modes_list = sorted(modes_with_strength.keys())
                    for idx, mode in enumerate(modes_list):
                        strengths = modes_with_strength[mode]

                        ax = axes[idx]
                        ax.hist(strengths, bins=30, color="steelblue", edgecolor="white", alpha=0.7)
                        ax.set_title(f"Mode: {mode}\nStrength Distribution (n={len(strengths)})")
                        ax.set_xlabel("impact_strength")
                        ax.set_ylabel("Count")
                        ax.grid(axis="y", alpha=0.3)

                        # 添加统计信息
                        if len(strengths) > 0:
                            mean_val = sum(strengths) / len(strengths)
                            ax.axvline(
                                mean_val,
                                color="red",
                                linestyle="--",
                                linewidth=1.5,
                                label=f"Mean: {mean_val:.3f}",
                            )
                            ax.legend(fontsize=8)

                    # 隐藏多余的子图
                    for idx in range(n_modes, len(axes)):
                        axes[idx].axis("off")

                    fig.tight_layout()
                    fig.savefig(plot_dir / "strength_by_mode.png", dpi=150)
                    plt.close(fig)

                    # 额外：所有 mode 的 strength 分布对比（箱线图）
                    if n_modes > 1:
                        fig, ax = plt.subplots(figsize=(max(8, n_modes * 1.5), 6))
                        data_to_plot = [modes_with_strength[mode] for mode in modes_list]
                        bp = ax.boxplot(data_to_plot, labels=modes_list, patch_artist=True)

                        # 美化箱线图
                        colors = plt.cm.Set3(range(len(bp["boxes"])))
                        for patch, color in zip(bp["boxes"], colors):
                            patch.set_facecolor(color)
                            patch.set_alpha(0.7)

                        ax.set_title("Impact Strength Distribution by Mode (Boxplot)")
                        ax.set_xlabel("Mode")
                        ax.set_ylabel("impact_strength")
                        ax.grid(axis="y", alpha=0.3)
                        plt.xticks(rotation=45, ha="right")
                        fig.tight_layout()
                        fig.savefig(plot_dir / "strength_by_mode_boxplot.png", dpi=150)
                        plt.close(fig)
        except Exception as e:
            print(f"\n警告：生成按 mode 分组的图表时出错: {e}")
            import traceback
            traceback.print_exc()

        print(f"\n图表已保存到: {plot_dir}")


if __name__ == "__main__":
    main()
