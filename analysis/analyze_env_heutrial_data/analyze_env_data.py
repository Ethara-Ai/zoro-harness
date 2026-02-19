#!/usr/bin/env python3
"""
分析 env_data 文件夹下的运行数据，按场景（easy, middle, hard）分别展示
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

def analyze_tool_calls(tool_calls_path):
    """分析单个 tool_calls.jsonl 文件"""
    daily_data = []
    total_expired = 0
    total_ordered = 0
    total_returns = 0
    total_sold = 0
    networth_history = []
    
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
                                money_earned = result_data.get('money_earned', 0.0)
                                sales_by_sku = result_data.get('sales_by_sku', {})
                                expired_discount_by_sku = result_data.get('expired_discount_by_sku', {})
                                returns_by_sku = result_data.get('returns_by_sku', {})
                                net_worth = result_data.get('net_worth', 0.0)
                                
                                total_sales = sum(sales_by_sku.values()) if isinstance(sales_by_sku, dict) else 0
                                expired_count = sum(expired_discount_by_sku.values()) if isinstance(expired_discount_by_sku, dict) else 0
                                return_count = sum(returns_by_sku.values()) if isinstance(returns_by_sku, dict) else 0
                                
                                total_expired += expired_count
                                total_returns += return_count
                                total_sold += total_sales
                                
                                daily_data.append({
                                    'date': current_date,
                                    'money_earned': float(money_earned) if money_earned else 0.0,
                                    'total_sales': int(total_sales),
                                    'expired_count': int(expired_count),
                                    'return_count': int(return_count),
                                    'net_worth': float(net_worth) if net_worth else 0.0
                                })
                                
                                networth_history.append({
                                    'date': current_date,
                                    'net_worth': float(net_worth) if net_worth else 0.0
                                })
                    
                    elif tool == 'place_order':
                        result = record.get('result', {})
                        if isinstance(result, dict):
                            result_data = result.get('result', {})
                            if isinstance(result_data, dict) and 'lines' in result_data:
                                for line_item in result_data['lines']:
                                    if isinstance(line_item, dict):
                                        quantity = line_item.get('quantity', 0)
                                        if isinstance(quantity, (int, float)):
                                            total_ordered += int(quantity)
                except:
                    continue
    except Exception as e:
        return None
    
    if not daily_data:
        return None
    
    dates = [d['date'] for d in daily_data if d['date']]
    if not dates:
        return None
    
    dates.sort()
    run_days = len(set(dates))
    
    avg_daily_sales = sum(d['total_sales'] for d in daily_data) / len(daily_data) if daily_data else 0
    avg_daily_profit = sum(d['money_earned'] for d in daily_data) / len(daily_data) if daily_data else 0
    expired_ratio = total_expired / total_ordered if total_ordered > 0 else 0
    return_ratio = total_returns / total_sold if total_sold > 0 else 0
    
    return {
        'run_days': run_days,
        'avg_daily_sales': avg_daily_sales,
        'avg_daily_profit': avg_daily_profit,
        'expired_ratio': expired_ratio,
        'return_ratio': return_ratio,
        'total_expired': total_expired,
        'total_ordered': total_ordered,
        'total_returns': total_returns,
        'total_sold': total_sold,
        'all_daily_sales': [d['total_sales'] for d in daily_data],
        'all_daily_profit': [d['money_earned'] for d in daily_data],
        'networth_history': networth_history
    }

def analyze_env_data(env_data_dir):
    """分析 env_data 目录下的所有数据"""
    env_data_path = Path(env_data_dir)
    scenario_data = {}
    
    if not env_data_path.exists():
        print(f"Warning: {env_data_dir} directory not found")
        return scenario_data
    
    scenario_order = ['easy', 'middle', 'hard']
    
    for scenario_name in scenario_order:
        scenario_dir = env_data_path / scenario_name
        if not scenario_dir.exists() or not scenario_dir.is_dir():
            continue
        
        print(f"Processing scenario: {scenario_name}")
        
        tool_calls_path = scenario_dir / 'tool_calls.jsonl'
        if tool_calls_path.exists():
            try:
                result = analyze_tool_calls(str(tool_calls_path))
                if result:
                    result['scenario'] = scenario_name
                    result['run_id'] = scenario_name
                    scenario_data[scenario_name] = [result]  # 每个场景只有一个运行
            except Exception as e:
                print(f"  Error processing {scenario_name}: {e}")
                continue
    
    return scenario_data

def calculate_statistics(scenario_data):
    """计算统计指标，按场景分组"""
    all_stats = {}
    
    for scenario, runs in scenario_data.items():
        if not runs:
            continue
        
        # 对于 env_data，每个场景只有一个运行
        run = runs[0]
        
        all_stats[scenario] = {
            'avg_run_days': run['run_days'],
            'avg_daily_sales': run['avg_daily_sales'],
            'avg_daily_profit': run['avg_daily_profit'],
            'expired_ratio': run['expired_ratio'],
            'return_ratio': run['return_ratio'],
            'total_expired': run['total_expired'],
            'total_ordered': run['total_ordered'],
            'total_returns': run['total_returns'],
            'total_sold': run['total_sold'],
            'max_days': run['run_days'],  # 只有一个运行，所以 max = avg
            'all_daily_sales': run['all_daily_sales'],
            'all_daily_profit': run['all_daily_profit'],
            'networth_history': run['networth_history']
        }
    
    return all_stats

def format_table_manually(headers, rows):
    """手动格式化表格（当 tabulate 不可用时）"""
    if not rows:
        return ""
    
    # 计算每列的最大宽度
    col_widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # 创建分隔线
    separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    
    # 格式化表头
    header_row = "|" + "|".join(f" {str(h):<{w}} " for h, w in zip(headers, col_widths)) + "|"
    
    # 格式化数据行
    data_rows = []
    for row in rows:
        data_row = "|" + "|".join(f" {str(cell):<{w}} " for cell, w in zip(row, col_widths)) + "|"
        data_rows.append(data_row)
    
    # 组合表格
    table = [separator, header_row, separator] + data_rows + [separator]
    return "\n".join(table)

def print_statistics(all_stats):
    """打印统计结果，使用表格形式"""
    print("\n" + "="*100)
    print("Env Data 统计结果")
    print("="*100)
    
    scenario_order = ['easy', 'middle', 'hard']
    
    # 准备表格数据
    table_data = []
    for scenario in scenario_order:
        if scenario not in all_stats:
            continue
        
        data = all_stats[scenario]
        row = [
            scenario.capitalize(),
            f"{data['avg_run_days']:.2f}",
            f"{data['avg_daily_sales']:.2f}",
            f"{data['avg_daily_profit']:.2f}",
            f"{data['expired_ratio']:.4f}",
            f"{data['return_ratio']:.4f}",
            f"{data['max_days']}"
        ]
        table_data.append(row)
    
    # 计算平均值
    if table_data:
        avg_run_days = sum(float(row[1]) for row in table_data) / len(table_data)
        avg_daily_sales = sum(float(row[2]) for row in table_data) / len(table_data)
        avg_daily_profit = sum(float(row[3]) for row in table_data) / len(table_data)
        
        # 计算总过期比例
        total_expired = sum(all_stats[s]['total_expired'] for s in scenario_order if s in all_stats)
        total_ordered = sum(all_stats[s]['total_ordered'] for s in scenario_order if s in all_stats)
        avg_expired_ratio = total_expired / total_ordered if total_ordered > 0 else 0
        
        # 计算总退货比例
        total_returns = sum(all_stats[s]['total_returns'] for s in scenario_order if s in all_stats)
        total_sold = sum(all_stats[s]['total_sold'] for s in scenario_order if s in all_stats)
        avg_return_ratio = total_returns / total_sold if total_sold > 0 else 0
        
        # 最大天数
        max_days = max(float(row[6]) for row in table_data) if table_data else 0
        
        avg_row = [
            "Average",
            f"{avg_run_days:.2f}",
            f"{avg_daily_sales:.2f}",
            f"{avg_daily_profit:.2f}",
            f"{avg_expired_ratio:.4f}",
            f"{avg_return_ratio:.4f}",
            f"{max_days:.0f}"
        ]
        table_data.append(avg_row)
    
    # 表头
    headers = [
        "Scenario",
        "Avg. Days",
        "Avg. Daily Sales",
        "Avg. Daily Profit",
        "Expiry Ratio",
        "Return Ratio",
        "Max Days"
    ]
    
    # 打印表格
    if HAS_TABULATE:
        print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="left"))
    else:
        print(format_table_manually(headers, table_data))

def save_statistics(all_stats, output_dir='env_data_analysis'):
    """保存统计结果到 JSON 文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    output_data = {}
    for scenario, data in all_stats.items():
        output_data[scenario] = {
            'avg_run_days': float(data['avg_run_days']),
            'avg_daily_sales': float(data['avg_daily_sales']),
            'avg_daily_profit': float(data['avg_daily_profit']),
            'expired_ratio': float(data['expired_ratio']),
            'return_ratio': float(data['return_ratio']),
            'total_expired': int(data['total_expired']),
            'total_ordered': int(data['total_ordered']),
            'total_returns': int(data['total_returns']),
            'total_sold': int(data['total_sold']),
            'max_days': int(data['max_days']),
            'all_daily_sales': [float(s) for s in data['all_daily_sales']],
            'all_daily_profit': [float(p) for p in data['all_daily_profit']],
        }
    
    output_path = os.path.join(output_dir, 'statistics.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n统计结果已保存到 {output_path}")

def main():
    env_data_dir = 'env_data'
    
    print("开始分析 env_data 文件夹...")
    scenario_data = analyze_env_data(env_data_dir)
    
    if not scenario_data:
        print("未找到任何数据！")
        return
    
    print(f"\n找到 {len(scenario_data)} 个场景的数据")
    for scenario, runs in scenario_data.items():
        print(f"  {scenario}: {len(runs)} 个运行")
    
    print("\n计算统计指标...")
    all_stats = calculate_statistics(scenario_data)
    
    print_statistics(all_stats)
    save_statistics(all_stats)
    
    print("\n分析完成！")

if __name__ == '__main__':
    main()

