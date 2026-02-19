#!/usr/bin/env python3
"""
直接使用focus_sku_tools_daily.json绘制散点图
展示Average Calls per Focus SKU-Day与AVG Profit和AVG Sales的关系
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib numpy")
except Exception as e:
    HAS_MATPLOTLIB = False
    print(f"Warning: matplotlib error: {e}")

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not installed. Correlation will not be calculated. Install with: pip install scipy")


# =========================
# Plot style (reference-aligned)
# =========================
def set_paper_style_like_reference():
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams['font.size'] = 14
    rcParams['axes.labelsize'] = 14
    rcParams['axes.titlesize'] = 14
    rcParams['xtick.labelsize'] = 16
    rcParams['ytick.labelsize'] = 16
    rcParams['legend.fontsize'] = 16
    rcParams['figure.titlesize'] = 18
    rcParams['axes.titlesize']   = 18
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


# 主要工具列表
MAIN_TOOLS = [
    'view_inventory',
    'view_sku_sales_history',
    'view_current_date_supplier_prices',
    'view_sku_prices',
    'view_sku_avg_ratings',
    'view_sku_reviews',
    'view_return_rates',
    'view_supplier_price_history',
]

# 工具显示名称
TOOL_DISPLAY_NAMES = {
    'view_inventory': 'Inventory',
    'view_sku_sales_history': 'Sales History',
    'view_current_date_supplier_prices': 'Supplier Prices',
    'view_sku_prices': 'SKU Prices',
    'view_sku_avg_ratings': 'SKU Ratings',
    'view_sku_reviews': 'SKU Reviews',
    'view_return_rates': 'Return Rates',
    'view_supplier_price_history': 'Price History',
}


def parse_date(date_str: str) -> Optional[datetime]:
    """解析日期字符串"""
    if not date_str:
        return None
    formats = ['%Y-%m-%d', '%m/%d/%Y', '%m/%d/%y']
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except:
            continue
    return None


def extract_performance_from_tool_calls(run_dir: Path) -> Optional[Dict[str, float]]:
    """从tool_calls.jsonl提取avg_daily_sales和avg_daily_profit"""
    tool_calls_path = run_dir / 'tool_calls.jsonl'
    if not tool_calls_path.exists():
        return None
    
    daily_data = []
    
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
                                money_earned = result_data.get('money_earned', 0.0)
                                sales_by_sku = result_data.get('sales_by_sku', {})
                                
                                if current_date:
                                    if isinstance(sales_by_sku, dict):
                                        total_sales = sum(sales_by_sku.values())
                                    else:
                                        total_sales = 0
                                    
                                    daily_data.append({
                                        'date': current_date,
                                        'money_earned': float(money_earned) if money_earned else 0.0,
                                        'total_sales': int(total_sales)
                                    })
                except:
                    continue
    except Exception as e:
        return None
    
    if not daily_data:
        return None
    
    sales_list = [d['total_sales'] for d in daily_data]
    profit_list = [d['money_earned'] for d in daily_data]
    
    avg_daily_sales = sum(sales_list) / len(sales_list) if sales_list else 0.0
    avg_daily_profit = sum(profit_list) / len(profit_list) if profit_list else 0.0
    
    return {
        'avg_daily_sales': avg_daily_sales,
        'avg_daily_profit': avg_daily_profit
    }


def calculate_run_tool_averages(run_data: Dict[str, Any]) -> Dict[str, float]:
    """计算每个run的每个工具的avg_calls_per_sku_day（参考aggregate_focus_sku_tools_by_model.py的逻辑）"""
    
    days_data = run_data.get('days', {})
    if not days_data:
        return {}
    
    run_total_sku_days = 0
    run_tool_stats = defaultdict(lambda: {'total_calls': 0, 'sku_days': set()})
    
    for day, day_data in days_data.items():
        focus_skus = day_data.get('focus_skus', [])
        sku_tool_calls = day_data.get('sku_tool_calls', {})
        
        for sku in focus_skus:
            run_total_sku_days += 1
            
            # 统计该SKU在该天的工具调用
            tools_for_sku = sku_tool_calls.get(sku, {})
            for tool_name, call_count in tools_for_sku.items():
                run_tool_stats[tool_name]['total_calls'] += call_count
                run_tool_stats[tool_name]['sku_days'].add((sku, day))
    
    # 计算每个工具的avg_calls_per_sku_day
    tool_averages = {}
    for tool_name, stats in run_tool_stats.items():
        total_calls = stats['total_calls']
        avg_calls_per_sku_day = total_calls / run_total_sku_days if run_total_sku_days > 0 else 0
        tool_averages[tool_name] = avg_calls_per_sku_day
    
    return tool_averages


def load_and_prepare_data(input_file: str, paper_data_dir: str) -> List[Dict[str, Any]]:
    """加载focus_sku_tools_daily.json并准备绘图数据"""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    runs = data.get('runs', [])
    paper_data_path = Path(paper_data_dir)
    
    prepared_data = []
    
    for run in runs:
        run_id = run.get('run_id', '')
        scenario = run.get('scenario', '')
        model = run.get('model', '')
        
        # 计算该run的每个工具的avg_calls_per_sku_day
        tool_averages = calculate_run_tool_averages(run)
        
        # 从tool_calls.jsonl提取performance数据
        # 构建run目录路径
        run_dir = paper_data_path / scenario / model / run_id
        performance = extract_performance_from_tool_calls(run_dir)
        
        if performance:
            prepared_data.append({
                'run_id': run_id,
                'scenario': scenario,
                'model': model,
                'tool_averages': tool_averages,  # {tool_name: avg_calls_per_sku_day}
                'avg_daily_sales': performance['avg_daily_sales'],
                'avg_daily_profit': performance['avg_daily_profit']
            })
    
    return prepared_data


def plot_tool_averages_vs_performance(data: List[Dict[str, Any]], output_dir: Path):
    """绘制工具平均调用次数与性能的散点图"""
    
    if not data:
        print("No data to plot")
        return
    
    # 设置图表样式
    set_paper_style_like_reference()
    
    # 按场景和模型分组
    scenarios = sorted(set(d['scenario'] for d in data))
    models = sorted(set(d['model'] for d in data))
    
    # 为每个模型分配颜色
    model_colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    model_color_map = {model: model_colors[i] for i, model in enumerate(models)}
    
    # 为每个工具绘制散点图（两个版本：包含所有点和只包含非零点）
    print("\nGenerating plots by tool type...")
    
    for tool in MAIN_TOOLS:
        # 版本1：包含所有点（包括0值）
        _plot_single_tool(data, tool, models, model_color_map, output_dir, include_zeros=True)
        
        # 版本2：只包含非零点（排除0值）
        _plot_single_tool(data, tool, models, model_color_map, output_dir, include_zeros=False)


def _plot_single_tool(data: List[Dict[str, Any]], tool: str, models: List[str], 
                     model_color_map: Dict[str, Any], output_dir: Path, include_zeros: bool = True):
    """绘制单个工具的散点图 - 生成两张独立的图：sales和income"""
    # 根据include_zeros决定标题和文件名后缀
    suffix = "" if include_zeros else "_non_zero_only"
    title_suffix = " (All Runs)" if include_zeros else " (Non-Zero Only)"
    
    # 收集所有数据点
    all_tool_avgs = []
    all_sales = []
    all_profits = []
    
    for model in models:
        model_data = [d for d in data if d['model'] == model]
        tool_avgs = []
        sales = []
        profits = []
        
        for run_data in model_data:
            tool_avg = run_data['tool_averages'].get(tool, 0)
            
            if include_zeros:
                # 显示所有runs，包括没有使用该工具的（tool_avg = 0）
                tool_avgs.append(tool_avg)
                sales.append(run_data['avg_daily_sales'])
                profits.append(run_data['avg_daily_profit'])
            else:
                # 只显示使用了该工具的runs（排除0值）
                if tool_avg > 0:
                    tool_avgs.append(tool_avg)
                    sales.append(run_data['avg_daily_sales'])
                    profits.append(run_data['avg_daily_profit'])
        
        if tool_avgs:
            all_tool_avgs.extend(tool_avgs)
            all_sales.extend(sales)
            all_profits.extend(profits)
    
    # 计算相关性并添加趋势线（只使用非零数据点）
    non_zero_data = [(x, y1, y2) for x, y1, y2 in zip(all_tool_avgs, all_sales, all_profits) if x > 0]
    
    if len(non_zero_data) > 1:
        non_zero_tool_avgs = [x for x, _, _ in non_zero_data]
        non_zero_sales = [y1 for _, y1, _ in non_zero_data]
        non_zero_profits = [y2 for _, _, y2 in non_zero_data]
    else:
        non_zero_tool_avgs = []
        non_zero_sales = []
        non_zero_profits = []
    
    # ========== 图1: Avg Daily Sales ==========
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    fig1.suptitle(f'{TOOL_DISPLAY_NAMES.get(tool, tool)} vs Avg Daily Sales{title_suffix}', fontweight='bold')
    
    # 绘制散点图
    for model in models:
        model_data = [d for d in data if d['model'] == model]
        tool_avgs = []
        sales = []
        
        for run_data in model_data:
            tool_avg = run_data['tool_averages'].get(tool, 0)
            
            if include_zeros:
                tool_avgs.append(tool_avg)
                sales.append(run_data['avg_daily_sales'])
            else:
                if tool_avg > 0:
                    tool_avgs.append(tool_avg)
                    sales.append(run_data['avg_daily_sales'])
        
        if tool_avgs:
            sizes = [100] * len(tool_avgs)
            ax1.scatter(tool_avgs, sales, c=[model_color_map[model]], 
                       label=model, alpha=0.6, s=sizes, edgecolors='black', linewidths=0.5)
    
    # 添加趋势线和相关性（只基于非零数据点）
    if len(non_zero_data) > 1:
        try:
            z1 = np.polyfit(non_zero_tool_avgs, non_zero_sales, 1)
            p1 = np.poly1d(z1)
            x_line = np.linspace(min(non_zero_tool_avgs), max(non_zero_tool_avgs), 100)
            ax1.plot(x_line, p1(x_line), "r--", alpha=0.8, linewidth=2, label='Trend')
        except:
            pass
        
        if HAS_SCIPY:
            try:
                corr_sales, p_sales = stats.pearsonr(non_zero_tool_avgs, non_zero_sales)
                num_runs = len(all_tool_avgs) if include_zeros else len(non_zero_data)
                num_using_tool = len(non_zero_data)
                
                if include_zeros:
                    info_text = f'r = {corr_sales:.3f}\np = {p_sales:.3f}\nRuns: {num_runs} (using: {num_using_tool})'
                else:
                    info_text = f'r = {corr_sales:.3f}\np = {p_sales:.3f}\nRuns: {num_using_tool}'
                
                ax1.text(0.05, 0.95, info_text, 
                        transform=ax1.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            except Exception as e:
                print(f"Warning: Could not calculate correlation: {e}")
    
    ax1.set_xlabel(f'Avg Calls per Focus SKU-Day\n({TOOL_DISPLAY_NAMES.get(tool, tool)})')
    ax1.set_ylabel('Avg Daily Sales')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    output_file1_png = output_dir / f'focus_sku_tool_{tool}_vs_sales{suffix}.png'
    output_file1_pdf = output_dir / f'focus_sku_tool_{tool}_vs_sales{suffix}.pdf'
    plt.savefig(output_file1_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_file1_pdf, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file1_png.name} and {output_file1_pdf.name}")
    
    # ========== 图2: Income (原avg daily profit) ==========
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    fig2.suptitle(f'{TOOL_DISPLAY_NAMES.get(tool, tool)} vs Income{title_suffix}', 
               fontweight='bold')
    
    # 绘制散点图
    for model in models:
        model_data = [d for d in data if d['model'] == model]
        tool_avgs = []
        profits = []
        
        for run_data in model_data:
            tool_avg = run_data['tool_averages'].get(tool, 0)
            
            if include_zeros:
                tool_avgs.append(tool_avg)
                profits.append(run_data['avg_daily_profit'])
            else:
                if tool_avg > 0:
                    tool_avgs.append(tool_avg)
                    profits.append(run_data['avg_daily_profit'])
        
        if tool_avgs:
            sizes = [100] * len(tool_avgs)
            ax2.scatter(tool_avgs, profits, c=[model_color_map[model]], 
                       label=model, alpha=0.6, s=sizes, edgecolors='black', linewidths=0.5)
    
    # 添加趋势线和相关性（只基于非零数据点）
    if len(non_zero_data) > 1:
        try:
            z2 = np.polyfit(non_zero_tool_avgs, non_zero_profits, 1)
            p2 = np.poly1d(z2)
            x_line = np.linspace(min(non_zero_tool_avgs), max(non_zero_tool_avgs), 100)
            ax2.plot(x_line, p2(x_line), "r--", alpha=0.8, linewidth=2, label='Trend')
        except:
            pass
        
        if HAS_SCIPY:
            try:
                corr_profit, p_profit = stats.pearsonr(non_zero_tool_avgs, non_zero_profits)
                num_runs = len(all_tool_avgs) if include_zeros else len(non_zero_data)
                num_using_tool = len(non_zero_data)
                
                if include_zeros:
                    info_text2 = f'r = {corr_profit:.3f}\np = {p_profit:.3f}\nRuns: {num_runs} (using: {num_using_tool})'
                else:
                    info_text2 = f'r = {corr_profit:.3f}\np = {p_profit:.3f}\nRuns: {num_using_tool}'
                
                ax2.text(0.05, 0.95, info_text2, 
                        transform=ax2.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            except Exception as e:
                print(f"Warning: Could not calculate correlation: {e}")
    
    ax2.set_xlabel(f'Avg Calls per Focus SKU-Day\n({TOOL_DISPLAY_NAMES.get(tool, tool)})')
    ax2.set_ylabel('Income')
    ax2.set_title('Tool Calls vs Income')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    output_file2_png = output_dir / f'focus_sku_tool_{tool}_vs_income{suffix}.png'
    output_file2_pdf = output_dir / f'focus_sku_tool_{tool}_vs_income{suffix}.pdf'
    plt.savefig(output_file2_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_file2_pdf, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file2_png.name} and {output_file2_pdf.name}")


def main():
    parser = argparse.ArgumentParser(
        description='绘制Average Calls per Focus SKU-Day与性能的散点图',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='/Users/linghuazhang/Desktop/Project/RetailBench/analysis/analysis_strategy_focus_skus_data/focus_sku_tools_daily.json',
        help='输入JSON文件路径（默认: focus_sku_tools_daily.json）'
    )
    
    parser.add_argument(
        '--paper-data-dir',
        type=str,
        default='paper_data',
        help='paper_data目录路径（用于读取tool_calls.jsonl）'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='focus_sku_tools_performance_plots',
        help='输出图片目录（默认: focus_sku_tools_performance_plots）'
    )
    
    args = parser.parse_args()
    
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required for plotting")
        print("Please install with: pip install matplotlib numpy scipy")
        return
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading data from: {input_path}")
    print(f"Reading performance data from: {args.paper_data_dir}")
    
    data = load_and_prepare_data(str(input_path), args.paper_data_dir)
    
    print(f"\nLoaded {len(data)} runs with performance data")
    
    print(f"Generating plots...")
    plot_tool_averages_vs_performance(data, output_dir)
    
    print(f"\n{'='*60}")
    print(f"All plots saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

