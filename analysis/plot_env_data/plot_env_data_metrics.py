#!/usr/bin/env python3
"""
绘制 env_data 中的四个指标随时间的变化：
1. Net worth over time
2. Money balance over time (funds)
3. Units sold (cumulative)
4. Expired items

风格参考提供的图片，使用2x2布局，带置信区间
"""

import json
import os
from pathlib import Path
from datetime import datetime, date
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import numpy as np
    HAS_MATPLOTLIB = True
    
    # 设置学术论文风格
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman']
    # rcParams['font.size'] = 11
    # rcParams['axes.labelsize'] = 11
    # rcParams['axes.titlesize'] = 12
    # rcParams['xtick.labelsize'] = 10
    # rcParams['ytick.labelsize'] = 10
    # rcParams['legend.fontsize'] = 9
    # rcParams['figure.titlesize'] = 13

    rcParams['font.size'] = 14
    rcParams['axes.labelsize'] = 14
    rcParams['axes.titlesize'] = 14
    rcParams['xtick.labelsize'] = 16
    rcParams['ytick.labelsize'] = 16
    rcParams['legend.fontsize'] = 16
    rcParams['figure.titlesize'] = 20
    rcParams['axes.titlesize']   = 20
    # rcParams['axes.titleweight'] = 'bold'
    rcParams['axes.titlepad']    = 8
    rcParams['axes.linewidth'] = 0.8
    rcParams['grid.linewidth'] = 0.5
    rcParams['lines.linewidth'] = 1.5
    rcParams['patch.linewidth'] = 0.5
    rcParams['xtick.major.width'] = 0.8
    rcParams['ytick.major.width'] = 0.8
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False
    rcParams['text.usetex'] = False
    # PDF 字体（强烈推荐）
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, cannot generate plots")

def parse_date(date_str):
    """解析日期字符串"""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except:
        return None

def extract_metrics_data(tool_calls_path):
    """从 tool_calls.jsonl 文件中提取所有指标数据"""
    daily_data = []
    
    if not os.path.exists(tool_calls_path):
        return None
    
    try:
        with open(tool_calls_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    tool = record.get('tool', '')
                    
                    if tool == 'end_today':
                        result = record.get('result', {})
                        if isinstance(result, dict):
                            result_data = result.get('result', {})
                            if isinstance(result_data, dict):
                                current_date = result_data.get('current_date')
                                funds = result_data.get('funds')
                                net_worth = result_data.get('net_worth')
                                sales_by_sku = result_data.get('sales_by_sku', {})
                                expired_discount_by_sku = result_data.get('expired_discount_by_sku', {})
                                
                                if current_date:
                                    # 计算总销量（累计）
                                    total_sales = sum(sales_by_sku.values()) if isinstance(sales_by_sku, dict) else 0
                                    # 计算过期商品数量
                                    expired_count = sum(expired_discount_by_sku.values()) if isinstance(expired_discount_by_sku, dict) else 0
                                    
                                    daily_data.append({
                                        'date': current_date,
                                        'funds': float(funds) if funds is not None else 0.0,
                                        'net_worth': float(net_worth) if net_worth is not None else 0.0,
                                        'units_sold': int(total_sales),
                                        'expired_items': int(expired_count)
                                    })
                except Exception as e:
                    continue
    except Exception as e:
        print(f"Error reading {tool_calls_path}: {e}")
        return None
    
    if not daily_data:
        return None
    
    # 按日期排序
    daily_data.sort(key=lambda x: x['date'])
    
    # 计算累计销量
    cumulative_sales = 0
    for entry in daily_data:
        cumulative_sales += entry['units_sold']
        entry['cumulative_units_sold'] = cumulative_sales
    
    return daily_data

# def plot_single_metric(ax, scenario_data, scenario_names_sorted, colors, default_colors, max_day, 
#                        metric_key, ylabel, title, get_data_func):
#     """绘制单个指标的通用函数"""
#     for idx, scenario_name in enumerate(scenario_names_sorted):
#         data = scenario_data[scenario_name]
#         dates = []
#         values = []
        
#         for entry in data:
#             date_obj = parse_date(entry['date'])
#             if date_obj:
#                 dates.append(date_obj)
#                 values.append(get_data_func(entry))
        
#         if dates:
#             start_date = dates[0]
#             day_numbers = [(d - start_date).days + 1 for d in dates]
            
#             color = colors.get(scenario_name, default_colors[idx % len(default_colors)])
#             ax.plot(day_numbers, values, label=scenario_name, color=color, linewidth=1.5, alpha=0.8)
#             # 添加置信区间
#             if len(values) > 1 and len(day_numbers) == len(values):
#                 try:
#                     window = min(5, max(1, len(values) // 10 + 1))
#                     if window > 1 and len(values) >= window:
#                         values_array = np.array(values, dtype=float)
#                         kernel = np.ones(window, dtype=float) / window
#                         smoothed = np.convolve(values_array, kernel, mode='same')
#                         std_val = float(np.std(values_array))
#                         std = std_val * 0.3 if std_val > 0 else 1.0
#                         if std > 0:
#                             lower = np.maximum(0, smoothed - std)
#                             upper = smoothed + std
#                             # 对称裁剪，避免边界效应
#                             trim = window // 2
#                             if trim > 0 and len(day_numbers) > 2 * trim:
#                                 day_numbers_ci = day_numbers[trim:-trim]
#                                 lower_ci = lower[trim:-trim]
#                                 upper_ci = upper[trim:-trim]
#                             else:
#                                 day_numbers_ci = day_numbers
#                                 lower_ci = lower
#                                 upper_ci = upper
#                             ax.fill_between(day_numbers_ci, lower_ci, upper_ci, alpha=0.2, color=color)
#                 except Exception:
#                     pass
    
#     ax.set_xlabel('Days')
#     ax.set_ylabel(ylabel)
#     ax.set_title(title, fontweight='bold')
#     ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
#     ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black', framealpha=0.9)
#     if max_day > 0:
#         ax.set_xlim(left=0, right=max_day)

# def plot_metrics(env_data_dir='env_data', output_dir='env_data_plots'):
#     """绘制 env_data 目录下所有场景的四个指标"""
    
#     if not HAS_MATPLOTLIB:
#         print("matplotlib not available, cannot generate plots")
#         return
    
#     env_data_path = Path(env_data_dir)
#     if not env_data_path.exists():
#         print(f"Error: {env_data_dir} directory not found")
#         return
    
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 收集所有场景的数据
#     scenario_data = {}
    
#     for scenario_dir in env_data_path.iterdir():
#         if not scenario_dir.is_dir():
#             continue
        
#         scenario_name = scenario_dir.name
#         tool_calls_path = scenario_dir / 'tool_calls.jsonl'
        
#         if not tool_calls_path.exists():
#             print(f"Warning: {tool_calls_path} not found, skipping")
#             continue
        
#         print(f"Processing {scenario_name}...")
#         metrics_data = extract_metrics_data(str(tool_calls_path))
        
#         if metrics_data:
#             scenario_data[scenario_name] = metrics_data
#             print(f"  Found {len(metrics_data)} data points")
#         else:
#             print(f"  No data found")
    
#     if not scenario_data:
#         print("No data found in env_data directory")
#         return
    
#     # 定义颜色方案（参考图片中的颜色）
#     colors = {
#         'easy': '#1f77b4',      # 深蓝色
#         'middle': '#2ca02c',    # 绿色
#         'hard': '#ff7f0e',      # 橙色
#     }
    
#     # 如果没有预定义颜色，使用默认颜色
#     default_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
#     scenario_names_sorted = sorted(scenario_data.keys())
    
#     # 先收集所有数据，找到最大天数
#     max_day = 0
#     all_day_numbers = {}
    
#     for scenario_name in scenario_names_sorted:
#         data = scenario_data[scenario_name]
#         dates = []
#         for entry in data:
#             date_obj = parse_date(entry['date'])
#             if date_obj:
#                 dates.append(date_obj)
        
#         if dates:
#             start_date = dates[0]
#             day_numbers = [(d - start_date).days + 1 for d in dates]
#             all_day_numbers[scenario_name] = day_numbers
#             if day_numbers:
#                 max_day = max(max_day, max(day_numbers))
    
#     # 准备数据提取函数
#     def get_networth(entry): return entry['net_worth']
#     def get_funds(entry): return entry['funds']
#     def get_cumulative_sales(entry): return entry['cumulative_units_sold']
    
#     # 处理过期商品累计数据
#     scenario_data_with_expired = {}
#     for scenario_name, data in scenario_data.items():
#         expired_items = []
#         cumulative_expired = 0
#         for entry in data:
#                 cumulative_expired += entry['expired_items']
#                 expired_items.append(cumulative_expired)
#         scenario_data_with_expired[scenario_name] = [
#             {**entry, 'cumulative_expired': expired_items[i]} 
#             for i, entry in enumerate(data)
#         ]
#     def get_cumulative_expired(entry): return entry['cumulative_expired']
    
#     # 1. 保存单独的 Net worth 图
#     fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
#     plot_single_metric(ax1, scenario_data, scenario_names_sorted, colors, default_colors, max_day,
#                        'net_worth', 'Net worth', 'Net worth over time', get_networth)
#     plt.tight_layout()
#     output_path1 = os.path.join(output_dir, 'networth_over_time.png')
#     output_path1_pdf = os.path.join(output_dir, 'networth_over_time.pdf')

#     plt.savefig(output_path1, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
#     plt.savefig(output_path1_pdf, bbox_inches='tight')

#     print(f"Net worth plot saved to {output_path1}")
#     print(f"Net worth plot saved to {output_path1_pdf}")
#     plt.close()
    
#     # 2. 保存单独的 Money balance 图
#     fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
#     plot_single_metric(ax2, scenario_data, scenario_names_sorted, colors, default_colors, max_day,
#                        'funds', 'Money balance', 'Money balance over time', get_funds)
#     plt.tight_layout()
#     output_path2 = os.path.join(output_dir, 'money_balance_over_time.png')
#     plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
#     print(f"Money balance plot saved to {output_path2}")
#     plt.close()
    
#     # 3. 保存 Net worth 和 Money balance 合并图（上下排列）
#     fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(8, 12))
#     plot_single_metric(ax3a, scenario_data, scenario_names_sorted, colors, default_colors, max_day,
#                        'net_worth', 'Net worth', 'Net worth over time', get_networth)
#     plot_single_metric(ax3b, scenario_data, scenario_names_sorted, colors, default_colors, max_day,
#                        'funds', 'Money balance', 'Money balance over time', get_funds)
#     plt.tight_layout()
#     output_path3 = os.path.join(output_dir, 'networth_and_funds.png')
#     plt.savefig(output_path3, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
#     print(f"Net worth and Funds combined plot saved to {output_path3}")
#     plt.close()
    
#     # 4. 保存单独的 Units sold 图
#     fig4, ax4 = plt.subplots(1, 1, figsize=(8, 6))
#     plot_single_metric(ax4, scenario_data, scenario_names_sorted, colors, default_colors, max_day,
#                        'cumulative_units_sold', 'Units sold (cumulative)', 'Units sold (cumulative)', get_cumulative_sales)
#     plt.tight_layout()
#     output_path4 = os.path.join(output_dir, 'units_sold_cumulative.png')
#     plt.savefig(output_path4, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
#     print(f"Units sold plot saved to {output_path4}")
#     plt.close()
    
#     # 5. 保存单独的 Expired items 图
#     fig5, ax5 = plt.subplots(1, 1, figsize=(8, 6))
#     plot_single_metric(ax5, scenario_data_with_expired, scenario_names_sorted, colors, default_colors, max_day,
#                        'cumulative_expired', 'Expired items (cumulative)', 'Expired items (cumulative)', get_cumulative_expired)
#     plt.tight_layout()
#     output_path5 = os.path.join(output_dir, 'expired_items_cumulative.png')
#     plt.savefig(output_path5, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
#     print(f"Expired items plot saved to {output_path5}")
#     plt.close()
    
#     # 6. 保存原来的2x2组合图
#     fig6, axes = plt.subplots(2, 2, figsize=(14, 10))
#     plot_single_metric(axes[0, 0], scenario_data, scenario_names_sorted, colors, default_colors, max_day,
#                        'net_worth', 'Net worth', 'Net worth over time', get_networth)
#     plot_single_metric(axes[0, 1], scenario_data, scenario_names_sorted, colors, default_colors, max_day,
#                        'funds', 'Money balance', 'Money balance over time', get_funds)
#     plot_single_metric(axes[1, 0], scenario_data, scenario_names_sorted, colors, default_colors, max_day,
#                        'cumulative_units_sold', 'Units sold (cumulative)', 'Units sold (cumulative)', get_cumulative_sales)
#     plot_single_metric(axes[1, 1], scenario_data_with_expired, scenario_names_sorted, colors, default_colors, max_day,
#                        'cumulative_expired', 'Expired items (cumulative)', 'Expired items (cumulative)', get_cumulative_expired)
#     plt.tight_layout(rect=[0, 0, 1, 0.99])
#     output_path6 = os.path.join(output_dir, 'env_data_metrics_2x2.png')
#     plt.savefig(output_path6, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
#     print(f"Combined 2x2 plot saved to {output_path6}")
#     plt.close()
    
#     # 打印统计信息
#     print("\n" + "="*80)
#     print("统计信息")
#     print("="*80)
    
#     for scenario_name, data in sorted(scenario_data.items()):
#         if not data:
#             continue
        
#         dates = [entry['date'] for entry in data]
#         networths = [entry['net_worth'] for entry in data]
#         funds = [entry['funds'] for entry in data]
#         total_sales = data[-1]['cumulative_units_sold'] if data else 0
#         total_expired = sum(entry['expired_items'] for entry in data)
        
#         initial_networth = networths[0] if networths else 0
#         final_networth = networths[-1] if networths else 0
#         max_funds = max(funds) if funds else 0
#         min_funds = min(funds) if funds else 0
        
#         print(f"\n场景: {scenario_name.upper()}")
#         print(f"  数据点数: {len(data)}")
#         print(f"  日期范围: {dates[0]} ~ {dates[-1]}")
#         print(f"  初始净值: {initial_networth:.2f}")
#         print(f"  最终净值: {final_networth:.2f}")
#         print(f"  最高资金: {max_funds:.2f}")
#         print(f"  最低资金: {min_funds:.2f}")
#         print(f"  累计销量: {total_sales}")
#         print(f"  累计过期: {total_expired}")
    
#     print("\n" + "="*80)

# def main():
#     import argparse
    
#     parser = argparse.ArgumentParser(
#         description="绘制 env_data 中的四个指标随时间的变化",
#         formatter_class=argparse.RawDescriptionHelpFormatter
#     )
    
#     parser.add_argument(
#         '--env-data-dir',
#         type=str,
#         default='env_data',
#         help='env_data 目录路径（默认: env_data）'
#     )
    
#     parser.add_argument(
#         '--output-dir',
#         type=str,
#         default='env_data_plots',
#         help='输出目录路径（默认: env_data_plots）'
#     )
    
#     args = parser.parse_args()
    
#     print("开始分析 env_data 目录...")
#     plot_metrics(args.env_data_dir, args.output_dir)
#     print("\n分析完成！")

# if __name__ == '__main__':
#     main()


def save_png_and_pdf(output_dir: str, basename_no_ext: str, dpi: int = 300):
    """
    保存当前 figure：同时输出 PNG 与 PDF
    - basename_no_ext: 例如 'networth_over_time'（不带扩展名）
    """
    png_path = os.path.join(output_dir, f"{basename_no_ext}.png")
    pdf_path = os.path.join(output_dir, f"{basename_no_ext}.pdf")

    plt.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(pdf_path, bbox_inches='tight')

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def plot_single_metric(
    ax,
    scenario_data,
    scenario_names_sorted,
    colors,
    default_colors,
    max_day,
    ylabel,
    title,
    get_data_func
):
    """绘制单个指标（多场景曲线 + 简易阴影带）"""

    for idx, scenario_name in enumerate(scenario_names_sorted):
        data = scenario_data[scenario_name]
        dates = []
        values = []

        for entry in data:
            date_obj = parse_date(entry['date'])
            if date_obj:
                dates.append(date_obj)
                values.append(get_data_func(entry))

        if not dates:
            continue

        start_date = dates[0]
        day_numbers = [(d - start_date).days + 1 for d in dates]

        color = colors.get(scenario_name, default_colors[idx % len(default_colors)])
        ax.plot(day_numbers, values, label=scenario_name, color=color, linewidth=1.5, alpha=0.85)

        # 简易“置信区间”阴影（基于平滑 + 全局标准差缩放）
        if len(values) > 1 and len(day_numbers) == len(values):
            try:
                window = min(5, max(1, len(values) // 10 + 1))
                if window > 1 and len(values) >= window:
                    values_array = np.array(values, dtype=float)
                    kernel = np.ones(window, dtype=float) / window
                    smoothed = np.convolve(values_array, kernel, mode='same')

                    std_val = float(np.std(values_array))
                    std = std_val * 0.3 if std_val > 0 else 0.0

                    if std > 0:
                        lower = np.maximum(0, smoothed - std)
                        upper = smoothed + std

                        trim = window // 2
                        if trim > 0 and len(day_numbers) > 2 * trim:
                            x_ci = day_numbers[trim:-trim]
                            lower_ci = lower[trim:-trim]
                            upper_ci = upper[trim:-trim]
                        else:
                            x_ci = day_numbers
                            lower_ci = lower
                            upper_ci = upper

                        ax.fill_between(x_ci, lower_ci, upper_ci, alpha=0.18, color=color)
            except Exception:
                pass

    # 注意：这里不再显式传 fontsize，让 rcParams 统一生效
    ax.set_xlabel('Days')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black', framealpha=0.9)

    if max_day > 0:
        ax.set_xlim(left=0, right=max_day)


def plot_metrics(env_data_dir='env_data', output_dir='env_data_plots'):
    """绘制 env_data 目录下所有场景的四个指标"""

    if not HAS_MATPLOTLIB:
        print("matplotlib not available, cannot generate plots")
        return

    env_data_path = Path(env_data_dir)
    if not env_data_path.exists():
        print(f"Error: {env_data_dir} directory not found")
        return

    os.makedirs(output_dir, exist_ok=True)

    scenario_data = {}

    for scenario_dir in env_data_path.iterdir():
        if not scenario_dir.is_dir():
            continue

        scenario_name = scenario_dir.name
        tool_calls_path = scenario_dir / 'tool_calls.jsonl'

        if not tool_calls_path.exists():
            print(f"Warning: {tool_calls_path} not found, skipping")
            continue

        print(f"Processing {scenario_name}...")
        metrics_data = extract_metrics_data(str(tool_calls_path))

        if metrics_data:
            scenario_data[scenario_name] = metrics_data
            print(f"  Found {len(metrics_data)} data points")
        else:
            print("  No data found")

    if not scenario_data:
        print("No data found in env_data directory")
        return

    # 颜色方案
    colors = {
        'easy': '#1f77b4',
        'middle': '#2ca02c',
        'hard': '#ff7f0e',
    }
    default_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
    scenario_names_sorted = sorted(scenario_data.keys())

    # 最大天数
    max_day = 0
    for scenario_name in scenario_names_sorted:
        data = scenario_data[scenario_name]
        dates = []
        for entry in data:
            d = parse_date(entry['date'])
            if d:
                dates.append(d)
        if dates:
            start_date = dates[0]
            day_numbers = [(d - start_date).days + 1 for d in dates]
            if day_numbers:
                max_day = max(max_day, max(day_numbers))

    # 指标提取
    def get_networth(entry): return entry['net_worth']
    def get_funds(entry): return entry['funds']
    def get_cumulative_sales(entry): return entry['cumulative_units_sold']

    # 过期累计
    scenario_data_with_expired = {}
    for scenario_name, data in scenario_data.items():
        cumulative_expired = 0
        new_data = []
        for entry in data:
            cumulative_expired += entry['expired_items']
            new_data.append({**entry, 'cumulative_expired': cumulative_expired})
        scenario_data_with_expired[scenario_name] = new_data

    def get_cumulative_expired(entry): return entry['cumulative_expired']

    # 1) Net worth
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    plot_single_metric(
        ax1, scenario_data, scenario_names_sorted, colors, default_colors, max_day,
        ylabel='Net worth', title='Net worth over time', get_data_func=get_networth
    )
    plt.tight_layout()
    save_png_and_pdf(output_dir, 'networth_over_time')
    plt.close()

    # 2) Money balance
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    plot_single_metric(
        ax2, scenario_data, scenario_names_sorted, colors, default_colors, max_day,
        ylabel='Money balance', title='Money balance over time', get_data_func=get_funds
    )
    plt.tight_layout()
    save_png_and_pdf(output_dir, 'money_balance_over_time')
    plt.close()

    # 3) Net worth + Money balance（上下）
    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(8, 12))
    plot_single_metric(
        ax3a, scenario_data, scenario_names_sorted, colors, default_colors, max_day,
        ylabel='Net worth', title='Net worth over time', get_data_func=get_networth
    )
    plot_single_metric(
        ax3b, scenario_data, scenario_names_sorted, colors, default_colors, max_day,
        ylabel='Money balance', title='Money balance over time', get_data_func=get_funds
    )
    plt.tight_layout()
    save_png_and_pdf(output_dir, 'networth_and_funds')
    plt.close()

    # 4) Units sold (cumulative)
    fig4, ax4 = plt.subplots(1, 1, figsize=(8, 6))
    plot_single_metric(
        ax4, scenario_data, scenario_names_sorted, colors, default_colors, max_day,
        ylabel='Units sold (cumulative)', title='Units sold (cumulative)', get_data_func=get_cumulative_sales
    )
    plt.tight_layout()
    save_png_and_pdf(output_dir, 'units_sold_cumulative')
    plt.close()

    # 5) Expired items (cumulative)
    fig5, ax5 = plt.subplots(1, 1, figsize=(8, 6))
    plot_single_metric(
        ax5, scenario_data_with_expired, scenario_names_sorted, colors, default_colors, max_day,
        ylabel='Expired items (cumulative)', title='Expired items (cumulative)', get_data_func=get_cumulative_expired
    )
    plt.tight_layout()
    save_png_and_pdf(output_dir, 'expired_items_cumulative')
    plt.close()

    # 6) 2x2 总图
    fig6, axes = plt.subplots(2, 2, figsize=(14, 10))

    plot_single_metric(
        axes[0, 0], scenario_data, scenario_names_sorted, colors, default_colors, max_day,
        ylabel='Net worth', title='Net worth over time', get_data_func=get_networth
    )
    plot_single_metric(
        axes[0, 1], scenario_data, scenario_names_sorted, colors, default_colors, max_day,
        ylabel='Money balance', title='Money balance over time', get_data_func=get_funds
    )
    plot_single_metric(
        axes[1, 0], scenario_data, scenario_names_sorted, colors, default_colors, max_day,
        ylabel='Units sold (cumulative)', title='Units sold (cumulative)', get_data_func=get_cumulative_sales
    )
    plot_single_metric(
        axes[1, 1], scenario_data_with_expired, scenario_names_sorted, colors, default_colors, max_day,
        ylabel='Expired items (cumulative)', title='Expired items (cumulative)', get_data_func=get_cumulative_expired
    )

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    save_png_and_pdf(output_dir, 'env_data_metrics_2x2')
    plt.close()

    # 统计信息
    print("\n" + "=" * 80)
    print("统计信息")
    print("=" * 80)

    for scenario_name, data in sorted(scenario_data.items()):
        if not data:
            continue

        dates = [entry['date'] for entry in data]
        networths = [entry['net_worth'] for entry in data]
        funds = [entry['funds'] for entry in data]
        total_sales = data[-1]['cumulative_units_sold']
        total_expired = sum(entry['expired_items'] for entry in data)

        initial_networth = networths[0] if networths else 0
        final_networth = networths[-1] if networths else 0
        max_funds = max(funds) if funds else 0
        min_funds = min(funds) if funds else 0

        print(f"\n场景: {scenario_name.upper()}")
        print(f"  数据点数: {len(data)}")
        print(f"  日期范围: {dates[0]} ~ {dates[-1]}")
        print(f"  初始净值: {initial_networth:.2f}")
        print(f"  最终净值: {final_networth:.2f}")
        print(f"  最高资金: {max_funds:.2f}")
        print(f"  最低资金: {min_funds:.2f}")
        print(f"  累计销量: {total_sales}")
        print(f"  累计过期: {total_expired}")

    print("\n" + "=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="绘制 env_data 中的四个指标随时间的变化（PNG + PDF）",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--env-data-dir', type=str, default='env_data', help='env_data 目录路径（默认: env_data）')
    parser.add_argument('--output-dir', type=str, default='env_data_plots', help='输出目录路径（默认: env_data_plots）')

    args = parser.parse_args()

    print("开始分析 env_data 目录...")
    plot_metrics(args.env_data_dir, args.output_dir)
    print("\n分析完成！")


if __name__ == '__main__':
    main()