#!/usr/bin/env python3
"""
根据focus_sku_tools_daily.json的统计信息，按模型和环境聚合，
计算每个模型在不同环境下，对于关注到的SKU平均会调用哪些工具
输出JSON统计文件和终端表格
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
import argparse

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    print("Warning: tabulate not installed. Install with: pip install tabulate")
    print("Will use simple text table format instead.")


def aggregate_by_model_scenario(input_file: str) -> Dict[str, Any]:
    """按模型和环境聚合统计信息
    
    统计逻辑：
    1. 先对每个run计算平均值
    2. 然后对同一model-scenario下的所有runs的平均值再求平均
    """
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    runs = data.get('runs', [])
    
    # 存储每个run的统计结果
    # {model: {scenario: [run_stats, ...]}}
    run_stats_by_model_scenario = defaultdict(lambda: defaultdict(list))
    
    # 处理每个run
    for run in runs:
        model = run.get('model', 'unknown')
        scenario = run.get('scenario', 'unknown')
        run_id = run.get('run_id', 'unknown')
        
        days_data = run.get('days', {})
        if not days_data:
            continue
        
        # 计算该run的总体信息
        run_total_days = len(days_data)
        run_focus_skus = set()
        run_total_sku_days = 0
        
        # 统计该run的工具调用
        # {tool_name: {total_calls, num_skus_using}}
        run_tool_stats = defaultdict(lambda: {'total_calls': 0, 'sku_days': set()})
        
        for day, day_data in days_data.items():
            focus_skus = day_data.get('focus_skus', [])
            sku_tool_calls = day_data.get('sku_tool_calls', {})
            
            for sku in focus_skus:
                run_focus_skus.add(sku)
                run_total_sku_days += 1
                
                # 统计该SKU在该天的工具调用
                tools_for_sku = sku_tool_calls.get(sku, {})
                for tool_name, call_count in tools_for_sku.items():
                    run_tool_stats[tool_name]['total_calls'] += call_count
                    run_tool_stats[tool_name]['sku_days'].add((sku, day))  # 使用(sku, day)作为唯一标识
        
        # 计算该run的每个工具的平均值
        run_tool_averages = {}
        for tool_name, stats in run_tool_stats.items():
            num_skus_using = len(stats['sku_days'])
            total_calls = stats['total_calls']
            
            # 该run的平均值
            avg_calls_per_sku_day = total_calls / run_total_sku_days if run_total_sku_days > 0 else 0
            usage_frequency = num_skus_using / run_total_sku_days if run_total_sku_days > 0 else 0
            avg_calls_per_using_sku = total_calls / num_skus_using if num_skus_using > 0 else 0
            
            run_tool_averages[tool_name] = {
                'avg_calls_per_sku_day': avg_calls_per_sku_day,
                'usage_frequency': usage_frequency,
                'avg_calls_per_using_sku': avg_calls_per_using_sku,
                'total_calls': total_calls,  # 保留原始数据用于汇总
                'num_skus_using': num_skus_using,
            }
        
        # 保存该run的统计结果
        run_stat = {
            'run_id': run_id,
            'total_days': run_total_days,
            'total_unique_focus_skus': len(run_focus_skus),
            'total_sku_days': run_total_sku_days,
            'tool_statistics': run_tool_averages
        }
        
        run_stats_by_model_scenario[model][scenario].append(run_stat)
    
    # 对每个model-scenario组合，计算所有runs的平均值
    aggregated_stats = {}
    
    for model, scenarios in run_stats_by_model_scenario.items():
        aggregated_stats[model] = {}
        
        for scenario, run_stats_list in scenarios.items():
            if not run_stats_list:
                continue
            
            num_runs = len(run_stats_list)
            
            # 汇总信息（所有runs的总和）
            total_days = sum(r['total_days'] for r in run_stats_list)
            total_sku_days = sum(r['total_sku_days'] for r in run_stats_list)
            
            # 从原始数据中收集所有唯一SKU
            all_unique_skus = set()
            for run in runs:
                if run.get('model') == model and run.get('scenario') == scenario:
                    days_data = run.get('days', {})
                    for day_data in days_data.values():
                        focus_skus = day_data.get('focus_skus', [])
                        all_unique_skus.update(focus_skus)
            total_unique_focus_skus = len(all_unique_skus)
            
            # 收集所有工具名称
            all_tools = set()
            for run_stat in run_stats_list:
                all_tools.update(run_stat['tool_statistics'].keys())
            
            # 对每个工具，计算所有runs的平均值的平均值
            tool_stats = {}
            for tool_name in all_tools:
                # 收集所有runs中该工具的平均值
                avg_calls_per_sku_day_list = []
                usage_frequency_list = []
                avg_calls_per_using_sku_list = []
                total_calls_list = []
                num_skus_using_list = []
                
                for run_stat in run_stats_list:
                    if tool_name in run_stat['tool_statistics']:
                        tool_data = run_stat['tool_statistics'][tool_name]
                        avg_calls_per_sku_day_list.append(tool_data['avg_calls_per_sku_day'])
                        usage_frequency_list.append(tool_data['usage_frequency'])
                        avg_calls_per_using_sku_list.append(tool_data['avg_calls_per_using_sku'])
                        total_calls_list.append(tool_data['total_calls'])
                        num_skus_using_list.append(tool_data['num_skus_using'])
                
                # 计算所有runs的平均值的平均值
                avg_calls_per_sku_day = sum(avg_calls_per_sku_day_list) / len(avg_calls_per_sku_day_list) if avg_calls_per_sku_day_list else 0
                usage_frequency = sum(usage_frequency_list) / len(usage_frequency_list) if usage_frequency_list else 0
                avg_calls_per_using_sku = sum(avg_calls_per_using_sku_list) / len(avg_calls_per_using_sku_list) if avg_calls_per_using_sku_list else 0
                
                # 汇总数据（所有runs的总和）
                total_calls = sum(total_calls_list)
                num_skus_using = sum(num_skus_using_list)  # 注意：这是所有runs的总和，不是唯一值
                
                tool_stats[tool_name] = {
                    'total_calls': total_calls,
                    'num_skus_using': num_skus_using,
                    'avg_calls_per_sku_day': round(avg_calls_per_sku_day, 3),  # 所有runs平均值的平均
                    'usage_frequency': round(usage_frequency, 3),  # 所有runs平均值的平均
                    'avg_calls_per_using_sku': round(avg_calls_per_using_sku, 3),  # 所有runs平均值的平均
                }
            
            aggregated_stats[model][scenario] = {
                'total_runs': num_runs,
                'total_days': total_days,
                'total_unique_focus_skus': total_unique_focus_skus,  # 注意：这里需要改进，应该合并所有runs的唯一SKU
                'total_sku_days': total_sku_days,
                'tool_statistics': tool_stats
            }
    
    return aggregated_stats


def print_table(aggregated_stats: Dict[str, Any]):
    """在终端打印表格 - 按场景分类，每个场景下按模型分组"""
    
    # 工具名称的简短别名
    tool_aliases = {
        'view_current_date_supplier_prices': 'supplier_prices',
        'view_inventory': 'inventory',
        'view_return_rates': 'return_rates',
        'view_sku_avg_ratings': 'sku_ratings',
        'view_sku_prices': 'sku_prices',
        'view_sku_reviews': 'sku_reviews',
        'view_sku_sales_history': 'sales_history',
        'view_supplier_price_history': 'price_history',
    }
    
    # 收集所有工具名称
    all_tools = set()
    for model_data in aggregated_stats.values():
        for scenario_data in model_data.values():
            all_tools.update(scenario_data.get('tool_statistics', {}).keys())
    
    all_tools = sorted(all_tools)
    
    # 创建工具显示名称映射
    tool_display_names = {tool: tool_aliases.get(tool, tool) for tool in all_tools}
    
    # 按场景分组数据
    scenario_groups = defaultdict(lambda: defaultdict(dict))
    for model, scenarios in aggregated_stats.items():
        for scenario, stats in scenarios.items():
            scenario_groups[scenario][model] = stats
    
    # 按场景顺序输出（still_middle, still_hard, dynamic_hard, baseline等）
    scenario_order = ['still_middle', 'still_hard', 'dynamic_hard', 'baseline']
    other_scenarios = [s for s in sorted(scenario_groups.keys()) if s not in scenario_order]
    sorted_scenarios = [s for s in scenario_order if s in scenario_groups] + other_scenarios
    
    # 为每个场景打印表格
    for scenario in sorted_scenarios:
        models_data = scenario_groups[scenario]
        if not models_data:
            continue
        
        print("\n" + "="*180)
        print(f"Scenario: {scenario.upper()}")
        print("="*180)
        
        # 汇总表格：每个模型的基本信息
        print(f"\n--- Summary for {scenario} ---")
        summary_rows = []
        
        for model in sorted(models_data.keys()):
            stats = models_data[model]
            
            row = [
                model, 
                stats['total_runs'], 
                stats['total_days'], 
                stats['total_unique_focus_skus'], 
                stats['total_sku_days']
            ]
            summary_rows.append(row)
        
        summary_headers = [
            'Model', 
            'Runs', 
            'Days', 
            'Unique SKUs', 
            'SKU-Days'
        ]
        
        if HAS_TABULATE:
            print(tabulate(summary_rows, headers=summary_headers, tablefmt='grid'))
        else:
            # 简单表格格式
            col_widths = [20, 8, 8, 12, 12]
            header_line = " | ".join([h.ljust(col_widths[i]) for i, h in enumerate(summary_headers)])
            print(header_line)
            print("-" * len(header_line))
            for row in summary_rows:
                row_line = " | ".join([str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)])
                print(row_line)
        
        # 详细工具统计表格：每个工具的平均调用次数
        print(f"\n--- Detailed Tool Statistics: Average Calls per Focus SKU-Day ({scenario}) ---")
        detail_rows = []
        
        # 用于计算平均值的累加器
        avg_accumulator = {tool: [] for tool in all_tools}
        
        for model in sorted(models_data.keys()):
            stats = models_data[model]
            tool_stats = stats.get('tool_statistics', {})
            
            row = [model]
            
            # 为每个工具添加统计
            for tool in all_tools:
                if tool in tool_stats:
                    tool_data = tool_stats[tool]
                    value = tool_data['avg_calls_per_sku_day']
                    row.append(f"{value:.3f}")
                    avg_accumulator[tool].append(value)
                else:
                    row.append('0.000')
            
            detail_rows.append(row)
        
        # 计算平均值行
        avg_row = ['Average']
        for tool in all_tools:
            if avg_accumulator[tool]:
                avg_value = sum(avg_accumulator[tool]) / len(avg_accumulator[tool])
                avg_row.append(f"{avg_value:.3f}")
            else:
                avg_row.append('0.000')
        detail_rows.append(avg_row)
        
        # 表头 - 使用简短名称
        detail_headers = ['Model'] + [tool_display_names[tool] for tool in all_tools]
        
        if HAS_TABULATE:
            print(tabulate(detail_rows, headers=detail_headers, tablefmt='grid', maxcolwidths=[20] + [15]*len(all_tools)))
        else:
            # 简单表格格式
            print(" | ".join([h[:20] if i == 0 else h[:15] for i, h in enumerate(detail_headers)]))
            print("-" * 180)
            for row in detail_rows:
                print(" | ".join([str(cell)[:20] if i == 0 else str(cell)[:15] for i, cell in enumerate(row)]))
        
        # 使用频率表格：每个工具的使用频率
        print(f"\n--- Tool Usage Frequency: Percentage of Focus SKU-Days Using Each Tool ({scenario}) ---")
        freq_rows = []
        
        # 用于计算平均值的累加器
        freq_accumulator = {tool: [] for tool in all_tools}
        
        for model in sorted(models_data.keys()):
            stats = models_data[model]
            tool_stats = stats.get('tool_statistics', {})
            
            row = [model]
            
            # 为每个工具添加使用频率
            for tool in all_tools:
                if tool in tool_stats:
                    tool_data = tool_stats[tool]
                    freq_value = tool_data['usage_frequency']
                    row.append(f"{freq_value*100:.1f}%")
                    freq_accumulator[tool].append(freq_value)
                else:
                    row.append('0.0%')
            
            freq_rows.append(row)
        
        # 计算平均值行
        avg_freq_row = ['Average']
        for tool in all_tools:
            if freq_accumulator[tool]:
                avg_freq = sum(freq_accumulator[tool]) / len(freq_accumulator[tool])
                avg_freq_row.append(f"{avg_freq*100:.1f}%")
            else:
                avg_freq_row.append('0.0%')
        freq_rows.append(avg_freq_row)
        
        freq_headers = ['Model'] + [tool_display_names[tool] for tool in all_tools]
        
        if HAS_TABULATE:
            print(tabulate(freq_rows, headers=freq_headers, tablefmt='grid', maxcolwidths=[20] + [15]*len(all_tools)))
        else:
            # 简单表格格式
            print(" | ".join([h[:20] if i == 0 else h[:15] for i, h in enumerate(freq_headers)]))
            print("-" * 180)
            for row in freq_rows:
                print(" | ".join([str(cell)[:20] if i == 0 else str(cell)[:15] for i, cell in enumerate(row)]))
    
    # 打印总体汇总
    print("\n" + "="*180)
    print("Overall Summary Across All Scenarios")
    print("="*180)
    
    overall_summary = []
    for scenario in sorted_scenarios:
        models_data = scenario_groups[scenario]
        total_runs = sum(s['total_runs'] for s in models_data.values())
        total_days = sum(s['total_days'] for s in models_data.values())
        total_skus = sum(s['total_unique_focus_skus'] for s in models_data.values())
        total_sku_days = sum(s['total_sku_days'] for s in models_data.values())
        num_models = len(models_data)
        
        overall_summary.append([
            scenario,
            num_models,
            total_runs,
            total_days,
            total_skus,
            total_sku_days
        ])
    
    overall_headers = ['Scenario', 'Models', 'Total Runs', 'Total Days', 'Total Unique SKUs', 'Total SKU-Days']
    
    if HAS_TABULATE:
        print(tabulate(overall_summary, headers=overall_headers, tablefmt='grid'))
    else:
        print(" | ".join(overall_headers))
        print("-" * 100)
        for row in overall_summary:
            print(" | ".join([str(cell) for cell in row]))


def main():
    parser = argparse.ArgumentParser(
        description='按模型和环境聚合focus_sku工具调用统计',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='focus_sku_tools_daily.json',
        help='输入JSON文件路径（默认: focus_sku_tools_daily.json）'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='focus_sku_tools_aggregated.json',
        help='输出JSON文件路径（默认: focus_sku_tools_aggregated.json）'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        print("Please run analyze_focus_sku_tools_daily.py first to generate the input file.")
        return
    
    print(f"Reading input from: {input_path}")
    aggregated_stats = aggregate_by_model_scenario(str(input_path))
    
    # 保存JSON
    output_path = Path(args.output)
    output_data = {
        'summary': {
            'total_models': len(aggregated_stats),
            'total_scenarios': sum(len(scenarios) for scenarios in aggregated_stats.values()),
        },
        'model_scenario_statistics': aggregated_stats
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")
    
    # 打印表格
    print_table(aggregated_stats)
    
    print("\n" + "="*120)
    print("Analysis complete!")
    print("="*120)


if __name__ == '__main__':
    main()

