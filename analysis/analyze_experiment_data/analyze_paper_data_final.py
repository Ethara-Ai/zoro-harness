#!/usr/bin/env python3
"""
分析 paper_data 文件夹下的运行数据，按场景（still_middle, still_hard, dynamic_hard）分别展示
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

def parse_date(date_str):
    """解析日期字符串"""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except:
        return None

def set_paper_style_like_reference():
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams['font.size'] = 14
    rcParams['axes.labelsize'] = 16
    rcParams['axes.titlesize'] = 14
    rcParams['xtick.labelsize'] = 18
    rcParams['ytick.labelsize'] = 18
    rcParams['legend.fontsize'] = 14
    rcParams['figure.titlesize'] = 18
    rcParams['axes.titlesize']   = 18
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

set_paper_style_like_reference()

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

    if 'still_middle' in tool_calls_path:
        scenario = 'still_middle'
    elif 'still_hard' in tool_calls_path:
        scenario = 'still_hard'
    elif 'dynamic_hard' in tool_calls_path:
        scenario = 'dynamic_hard'
    else:
        scenario = 'unknown'

    category_number = 5 if scenario == 'still_middle' else 20
    
    avg_daily_sales = sum(d['total_sales'] for d in daily_data) / len(daily_data) if daily_data else 0
    avg_daily_profit = sum(d['money_earned'] for d in daily_data) / len(daily_data) if daily_data else 0
    avg_daily_sales_per_category = avg_daily_sales / category_number
    avg_daily_profit_per_category = avg_daily_profit / category_number
    expired_ratio = total_expired / total_ordered if total_ordered > 0 else 0
    return_ratio = total_returns / total_sold if total_sold > 0 else 0
    
    return {
        'run_days': run_days,
        'avg_daily_sales': avg_daily_sales,
        'avg_daily_profit': avg_daily_profit,
        'avg_daily_sales_per_category': avg_daily_sales_per_category,
        'avg_daily_profit_per_category': avg_daily_profit_per_category,
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

def validate_args_json(run_dir, scenario_name):
    """验证 args.json 中的 config_type 是否与文件夹名一致
    对于 baseline 场景，从 db_path 中提取配置类型
    """
    args_path = run_dir / 'args.json'
    
    if not args_path.exists():
        print(f"    Warning: {run_dir.name} missing args.json")
        return False, None
    
    try:
        with open(args_path, 'r', encoding='utf-8') as f:
            args = json.load(f)
        
        # 对于 baseline 场景，从 db_path 中提取配置类型
        if scenario_name == 'baseline':
            db_path = args.get('db_path', '')
            # db_path 格式: "model/baseline/model_name_config_type"
            # 例如: "model/baseline/gemini-3-flash-preview_still_middle"
            config_type = ''
            if db_path:
                # 从 db_path 中提取配置类型
                # 先尝试匹配已知的配置类型
                known_types = ['still_middle', 'still_hard', 'dynamic_hard', 'still_middle2']
                for known_type in known_types:
                    if db_path.endswith('_' + known_type):
                        config_type = known_type
                        break
                
                # 如果没匹配到，尝试从路径中提取最后一部分
                if not config_type:
                    # 获取文件名部分（去掉路径）
                    filename = db_path.split('/')[-1] if '/' in db_path else db_path
                    # 从文件名中提取配置类型（假设格式是 model_name_config_type）
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        # 尝试匹配最后几个部分
                        for i in range(len(parts) - 1, 0, -1):
                            potential_type = '_'.join(parts[i:])
                            if potential_type in known_types:
                                config_type = potential_type
                                break
        else:
            # 对于非 baseline 场景，直接使用 config_type 字段
            config_type = args.get('config_type', '')
        
        if config_type != scenario_name and scenario_name != 'baseline':
            # baseline 场景不需要与文件夹名匹配，因为它可能包含多个配置类型
            print(f"    Warning: {run_dir.name} - args.json config_type '{config_type}' doesn't match folder '{scenario_name}'")
            return False, config_type
        
        # baseline 场景总是返回 True（因为它的配置类型在 db_path 中，不需要与文件夹名匹配）
        if scenario_name == 'baseline':
            return True, config_type
        
        return True, config_type
    except Exception as e:
        print(f"    Warning: {run_dir.name} - Failed to read args.json: {e}")
        return False, None

def analyze_paper_data(paper_data_dir):
    """分析 paper_data 目录下的所有数据"""
    paper_data_path = Path(paper_data_dir)
    scenario_data = defaultdict(lambda: defaultdict(list))
    validation_stats = {
        'total_runs': 0,
        'valid_runs': 0,
        'invalid_runs': 0,
        'missing_args': 0,
        'mismatches': []
    }
    
    try:
        for scenario_dir in paper_data_path.iterdir():
            if not scenario_dir.is_dir():
                continue
            
            scenario_name = scenario_dir.name
            print(f"Processing scenario: {scenario_name}")
            
            for model_dir in scenario_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                
                model_name = model_dir.name
                
                for run_dir in model_dir.iterdir():
                    if not run_dir.is_dir() or not run_dir.name.startswith('run_env_'):
                        continue
                    
                    validation_stats['total_runs'] += 1
                    
                    # 验证 args.json
                    is_valid, config_type = validate_args_json(run_dir, scenario_name)
                    
                    if not is_valid:
                        validation_stats['invalid_runs'] += 1
                        if config_type is None:
                            validation_stats['missing_args'] += 1
                        else:
                            validation_stats['mismatches'].append({
                                'run_id': run_dir.name,
                                'scenario': scenario_name,
                                'model': model_name,
                                'config_type': config_type
                            })
                        # 即使验证失败，仍然处理数据（可选：可以跳过）
                        # continue
                    
                    tool_calls_path = run_dir / 'tool_calls.jsonl'
                    if tool_calls_path.exists():
                        try:
                            result = analyze_tool_calls(str(tool_calls_path))
                            if result:
                                result['scenario'] = scenario_name
                                result['model'] = model_name
                                result['run_id'] = run_dir.name
                                result['config_type'] = config_type if config_type else 'unknown'
                                scenario_data[scenario_name][model_name].append(result)
                                validation_stats['valid_runs'] += 1
                        except:
                            continue
    except Exception as e:
        print(f"Error: {e}")
    
    # 打印验证统计
    print(f"\n验证统计:")
    print(f"  总运行数: {validation_stats['total_runs']}")
    print(f"  有效运行数: {validation_stats['valid_runs']}")
    print(f"  无效运行数: {validation_stats['invalid_runs']}")
    print(f"  缺少 args.json: {validation_stats['missing_args']}")
    print(f"  配置类型不匹配: {len(validation_stats['mismatches'])}")
    
    if validation_stats['mismatches']:
        print(f"\n配置类型不匹配的详细列表:")
        for mismatch in validation_stats['mismatches'][:10]:  # 只显示前10个
            print(f"  {mismatch['scenario']}/{mismatch['model']}/{mismatch['run_id']}: "
                  f"文件夹='{mismatch['scenario']}', args.json='{mismatch['config_type']}'")
        if len(validation_stats['mismatches']) > 10:
            print(f"  ... 还有 {len(validation_stats['mismatches']) - 10} 个不匹配项")
    
    return scenario_data

def calculate_statistics(scenario_data):
    """计算统计指标，按场景分组"""
    all_stats = {}
    
    for scenario, models in scenario_data.items():
        stats = {}
        for model_name, runs in models.items():
            if not runs:
                continue
            
            run_days_list = [r['run_days'] for r in runs]
            avg_run_days = sum(run_days_list) / len(run_days_list) if run_days_list else 0
            
            all_daily_sales = []
            for r in runs:
                all_daily_sales.extend(r['all_daily_sales'])
            avg_daily_sales = sum(all_daily_sales) / len(all_daily_sales) if all_daily_sales else 0
            
            all_daily_profit = []
            for r in runs:
                all_daily_profit.extend(r['all_daily_profit'])
            avg_daily_profit = sum(all_daily_profit) / len(all_daily_profit) if all_daily_profit else 0
            
            # 计算平均每个类别的销量和收入
            # 直接使用合并后的 avg_daily_sales 和 avg_daily_profit 除以类别数
            # 获取该场景的类别数量
            scenario = runs[0].get('scenario', '') if runs else ''
            config_type = runs[0].get('config_type', scenario) if runs else scenario
            if not config_type or config_type == 'unknown':
                config_type = scenario
            
            # 根据场景确定类别数量
            if config_type == 'still_middle' or config_type == 'baseline':
                category_number = 5
            elif config_type in ['still_hard', 'dynamic_hard']:
                category_number = 20
            else:
                # 默认从路径判断
                if 'still_middle' in scenario or 'baseline' in scenario:
                    category_number = 5
                else:
                    category_number = 20
            
            # 直接计算：平均每类销量 = 平均每天销量 / 类别数
            avg_daily_sales_per_category = avg_daily_sales / category_number if category_number > 0 else 0
            # 直接计算：平均每类盈利 = 平均每天盈利 / 类别数
            avg_daily_profit_per_category = avg_daily_profit / category_number if category_number > 0 else 0
            
            total_expired = sum(r['total_expired'] for r in runs)
            total_ordered = sum(r['total_ordered'] for r in runs)
            total_returns = sum(r['total_returns'] for r in runs)
            total_sold = sum(r['total_sold'] for r in runs)
            expired_ratio = total_expired / total_ordered if total_ordered > 0 else 0
            return_ratio = total_returns / total_sold if total_sold > 0 else 0
            
            longest_run = max(runs, key=lambda x: x['run_days']) if runs else None
            
            stats[model_name] = {
                'avg_run_days': avg_run_days,
                'avg_daily_sales': avg_daily_sales,
                'avg_daily_profit': avg_daily_profit,
                'avg_daily_sales_per_category': avg_daily_sales_per_category,
                'avg_daily_profit_per_category': avg_daily_profit_per_category,
                'expired_ratio': expired_ratio,
                'return_ratio': return_ratio,
                'total_expired': total_expired,
                'total_ordered': total_ordered,
                'total_returns': total_returns,
                'total_sold': total_sold,
                'all_run_days': run_days_list,
                'all_daily_sales': all_daily_sales,
                'all_daily_profit': all_daily_profit,
                'longest_run': longest_run
            }
        
        all_stats[scenario] = stats
    
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
    """打印统计结果，按场景分组，使用表格形式"""
    print("\n" + "="*100)
    print("统计结果（按场景分组）")
    print("="*100)
    
    scenario_order = ['still_middle', 'still_hard', 'dynamic_hard', 'baseline']
    
    for scenario in scenario_order:
        if scenario not in all_stats:
            continue
        
        stats = all_stats[scenario]
        if not stats:
            continue
        
        print(f"\n{'='*100}")
        print(f"场景: {scenario.upper()}")
        print(f"{'='*100}\n")
        
        # 准备表格数据
        table_data = []
        for model_name, data in sorted(stats.items()):
            row = [
                model_name,
                f"{data['avg_run_days']:.2f}",
                f"{data['avg_daily_sales']:.2f}",
                f"{data['avg_daily_profit']:.2f}",
                f"{data['avg_daily_sales_per_category']:.2f}",
                f"{data['avg_daily_profit_per_category']:.2f}",
                f"{data['expired_ratio']:.4f}",
                f"{data['return_ratio']:.4f}",
                f"{data['total_expired']}/{data['total_ordered']}",
                f"{data['longest_run']['run_days']}" if data['longest_run'] else "N/A"
            ]
            table_data.append(row)
        
        # 表头
        headers = [
            "模型",
            "平均运行天数",
            "平均每天销量",
            "平均每天盈利",
            "平均每类销量",
            "平均每类盈利",
            "过期比例",
            "退货比例",
            "过期/总进货",
            "最长运行天数"
        ]
        
        # 打印表格
        if HAS_TABULATE:
            print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="left"))
        else:
            print(format_table_manually(headers, table_data))
        
        # 计算七个模型的平均值
        if len(stats) > 0:
            print(f"\n{'='*100}")
            print("七个模型平均值:")
            print(f"{'='*100}\n")
            
            # 计算平均值
            avg_run_days = sum(data['avg_run_days'] for data in stats.values()) / len(stats)
            
            # 合并所有模型的每日数据
            all_avg_daily_sales = []
            all_avg_daily_profit = []
            total_expired_all = 0
            total_ordered_all = 0
            total_returns_all = 0
            total_sold_all = 0
            
            for data in stats.values():
                all_avg_daily_sales.extend(data['all_daily_sales'])
                all_avg_daily_profit.extend(data['all_daily_profit'])
                total_expired_all += data['total_expired']
                total_ordered_all += data['total_ordered']
                total_returns_all += data['total_returns']
                total_sold_all += data['total_sold']
            
            avg_daily_sales = sum(all_avg_daily_sales) / len(all_avg_daily_sales) if all_avg_daily_sales else 0
            avg_daily_profit = sum(all_avg_daily_profit) / len(all_avg_daily_profit) if all_avg_daily_profit else 0
            
            # 计算平均每类销量和盈利
            all_avg_sales_per_category = [data['avg_daily_sales_per_category'] for data in stats.values() if data.get('avg_daily_sales_per_category', 0) > 0]
            all_avg_profit_per_category = [data['avg_daily_profit_per_category'] for data in stats.values() if data.get('avg_daily_profit_per_category', 0) > 0]
            avg_sales_per_category = sum(all_avg_sales_per_category) / len(all_avg_sales_per_category) if all_avg_sales_per_category else 0
            avg_profit_per_category = sum(all_avg_profit_per_category) / len(all_avg_profit_per_category) if all_avg_profit_per_category else 0
            
            avg_expired_ratio = total_expired_all / total_ordered_all if total_ordered_all > 0 else 0
            avg_return_ratio = total_returns_all / total_sold_all if total_sold_all > 0 else 0
            
            # 准备平均值行
            avg_row = [
                "平均值",
                f"{avg_run_days:.2f}",
                f"{avg_daily_sales:.2f}",
                f"{avg_daily_profit:.2f}",
                f"{avg_sales_per_category:.2f}",
                f"{avg_profit_per_category:.2f}",
                f"{avg_expired_ratio:.4f}",
                f"{avg_return_ratio:.4f}",
                f"{total_expired_all}/{total_ordered_all}",
                "N/A"
            ]
            
            # 打印平均值表格
            if HAS_TABULATE:
                print(tabulate([avg_row], headers=headers, tablefmt="grid", stralign="left"))
            else:
                print(format_table_manually(headers, [avg_row]))
        
        # 打印详细信息（运行天数列表）
        print(f"\n详细运行天数:")
        for model_name, data in sorted(stats.items()):
            print(f"  {model_name}: {data['all_run_days']}")
        
        # 打印最长运行的 run_id
        print(f"\n最长运行详情:")
        for model_name, data in sorted(stats.items()):
            if data['longest_run']:
                print(f"  {model_name}: {data['longest_run']['run_id']} ({data['longest_run']['run_days']} 天)")

def save_statistics(all_stats, output_dir='paper_data_analysis'):
    """保存统计结果到 JSON 文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    output_data = {}
    for scenario, stats in all_stats.items():
        output_data[scenario] = {}
        for model_name, data in stats.items():
            output_data[scenario][model_name] = {
                'avg_run_days': float(data['avg_run_days']),
                'avg_daily_sales': float(data['avg_daily_sales']),
                'avg_daily_profit': float(data['avg_daily_profit']),
                'avg_daily_sales_per_category': float(data.get('avg_daily_sales_per_category', 0.0)),
                'avg_daily_profit_per_category': float(data.get('avg_daily_profit_per_category', 0.0)),
                'expired_ratio': float(data['expired_ratio']),
                'return_ratio': float(data['return_ratio']),
                'total_expired': int(data['total_expired']),
                'total_ordered': int(data['total_ordered']),
                'total_returns': int(data['total_returns']),
                'total_sold': int(data['total_sold']),
                'all_run_days': [int(d) for d in data['all_run_days']],
                'all_daily_sales': [float(s) for s in data['all_daily_sales']],
                'all_daily_profit': [float(p) for p in data['all_daily_profit']],
            }
            if data['longest_run']:
                output_data[scenario][model_name]['longest_run_id'] = data['longest_run']['run_id']
                output_data[scenario][model_name]['longest_run_days'] = data['longest_run']['run_days']
    
    output_path = os.path.join(output_dir, 'statistics.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n统计结果已保存到 {output_path}")

def plot_networth_trajectory(all_stats, output_dir='paper_data_analysis'):
    """绘制 networth 变化轨迹"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except:
        print("matplotlib not available, skipping plots")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    scenario_order = ['still_middle', 'still_hard', 'dynamic_hard', 'baseline']
    
    # 为每个场景创建单独的图
    for scenario in scenario_order:
        if scenario not in all_stats:
            continue
        
        stats = all_stats[scenario]
        if not stats:
            continue
        
        plt.figure(figsize=(14, 8))
        
        for model_name, data in sorted(stats.items()):
            if not data['longest_run']:
                continue
            
            networth_history = data['longest_run']['networth_history']
            if not networth_history:
                continue
            
            networth_history.sort(key=lambda x: x['date'] if x['date'] else '')
            
            dates = []
            networths = []
            for entry in networth_history:
                if entry['date']:
                    dates.append(entry['date'])
                    networths.append(entry['net_worth'])
            
            if dates:
                start_date = parse_date(dates[0])
                if start_date:
                    day_numbers = [(parse_date(d) - start_date).days for d in dates if parse_date(d)]
                    plt.plot(day_numbers, networths, label=model_name, marker='o', markersize=2, linewidth=1.5)
        
        plt.xlabel('Day')
        plt.ylabel('Net Worth')
        plt.title(f'Net Worth Trajectory - {scenario.upper()}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'networth_trajectory_{scenario}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Networth trajectory for {scenario} saved to {output_path}")
        
        # 同时保存PDF格式
        pdf_path = os.path.join(output_dir, f'networth_trajectory_{scenario}.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Networth trajectory for {scenario} saved to {pdf_path}")
        
        plt.close()

def plot_category_metrics(all_stats, output_dir='paper_data_analysis'):
    """绘制 Sales Per Category 和 Income Per Category 柱状图
    将两个指标放在一个图中，所有场景（still_middle, still_hard, dynamic_hard）放在一起
    包含启发式策略（Heuristic Policy）作为对比
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except:
        print("matplotlib not available, skipping category plots")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 只处理这三个场景
    scenarios_to_plot = ['still_middle', 'still_hard', 'dynamic_hard']
    
    # 启发式策略数据（Sales By Category, Income By Category）
    heuristic_data = {
        'still_middle': {
            'sales_per_category': 674.18 / 5,  # 134.84
            'income_per_category': 729.46 / 5   # 145.89
        },
        'still_hard': {
            'sales_per_category': 1870.21 / 20,  # 93.51
            'income_per_category': 2809.39 / 20  # 140.47
        },
        'dynamic_hard': {
            'sales_per_category': 1667.84 / 20,  # 83.39
            'income_per_category': 2748.94 / 20  # 137.45
        }
    }
    
    # 模型名称映射：原始名称 -> 显示名称
    model_name_mapping = {
        'deepseekv3_2': 'DeepSeek-V3.2 (Exp.)',
        'gemini3_fast': 'Gemini-3 (Fast)',
        'glm4_6': 'GLM-4.6',
        'grok4_fast': 'Grok-4.1 Fast',
        'kimi_thinking': 'Kimi-K2 (Thinking)',
        'gpt5_1mini': 'OpenAI-5.1 Mini',
        'qwen_235b': 'Qwen-235B'
    }
    
    # 收集所有模型名称（跨所有场景）
    all_model_names = set()
    for scenario in scenarios_to_plot:
        if scenario in all_stats:
            all_model_names.update(all_stats[scenario].keys())
    all_model_names = sorted(all_model_names)
    
    if not all_model_names:
        print("No models found for plotting")
        return
    
    # 将原始模型名称映射为显示名称
    display_model_names = [model_name_mapping.get(name, name) for name in all_model_names]
    
    # 准备数据：每个场景每个模型的两个指标
    sales_data = {scenario: [] for scenario in scenarios_to_plot}
    profit_data = {scenario: [] for scenario in scenarios_to_plot}
    
    for scenario in scenarios_to_plot:
        if scenario not in all_stats:
            continue
        stats = all_stats[scenario]
        for model_name in all_model_names:
            if model_name in stats:
                data = stats[model_name]
                sales_data[scenario].append(data.get('avg_daily_sales_per_category', 0.0))
                profit_data[scenario].append(data.get('avg_daily_profit_per_category', 0.0))
            else:
                sales_data[scenario].append(0.0)
                profit_data[scenario].append(0.0)
    
    # 创建子图：上下两个图，上面是 Sales，下面是 Income
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(14, len(all_model_names) * 1.2), 12))
    
    x = np.arange(len(all_model_names))
    width = 0.2  # 每个场景一个柱子，三个场景 + 启发式策略共占宽度
    
    # 场景颜色映射（使用更专业、更协调的配色方案）
    scenario_colors = {
        'still_middle': '#3989BD',  # 深蓝色 - 专业且易读
        'still_hard': '#FF8F2B',    # 深紫红色 - 与蓝色形成良好对比
        'dynamic_hard': '#50B150'   # 深橙色 - 温暖且醒目
    }
    
    # 绘制 Sales Per Category
    for i, scenario in enumerate(scenarios_to_plot):
        if scenario not in all_stats:
            continue
        offset = (i - 1.5) * width  # -1.5, -0.5, 0.5
        color = scenario_colors.get(scenario, 'gray')
        scenerio_text = "Easy" if scenario == 'still_middle' else "Middle" if scenario == 'still_hard' else "Hard"
        bars = ax1.bar(x + offset, sales_data[scenario], width, label=scenerio_text, 
                      alpha=0.8, color=color)
        
        # 添加数值标签
        for bar, val in zip(bars, sales_data[scenario]):
            if val > 0:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}',
                       ha='center', va='bottom')
    
    # 添加启发式策略（用水平线表示，每个场景一条线，使用与柱状图相同的颜色）
    for i, scenario in enumerate(scenarios_to_plot):
        if scenario in heuristic_data:
            heuristic_sales = heuristic_data[scenario]['sales_per_category']
            color = scenario_colors.get(scenario, 'red')
            # 绘制贯穿整个图表的水平参考线
            scenerio_text = "Easy" if scenario == 'still_middle' else "Middle" if scenario == 'still_hard' else "Hard"
            ax1.axhline(y=heuristic_sales, color=color, linestyle='--', linewidth=2, 
                       alpha=0.7, label=f'Heuristic ({scenerio_text})')
    
    ax1.set_ylabel('Sales Per Category')
    ax1.set_title('Sales Per Category Comparison Across Scenarios (with Heuristic Policy)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_model_names, rotation=0, ha='center')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 绘制 Income Per Category
    for i, scenario in enumerate(scenarios_to_plot):
        if scenario not in all_stats:
            continue
        offset = (i - 1.5) * width
        color = scenario_colors.get(scenario, 'gray')
        scenerio_text = "Easy" if scenario == 'still_middle' else "Middle" if scenario == 'still_hard' else "Hard"
        bars = ax2.bar(x + offset, profit_data[scenario], width, label=scenerio_text, 
                      alpha=0.8, color=color)
        
        # 添加数值标签
        for bar, val in zip(bars, profit_data[scenario]):
            if val > 0:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}',
                       ha='center', va='bottom')
    
    # 添加启发式策略（用水平线表示，每个场景一条线，使用与柱状图相同的颜色）
    for i, scenario in enumerate(scenarios_to_plot):
        if scenario in heuristic_data:
            heuristic_income = heuristic_data[scenario]['income_per_category']
            color = scenario_colors.get(scenario, 'red')
            # 绘制贯穿整个图表的水平参考线
            scenerio_text = "Easy" if scenario == 'still_middle' else "Middle" if scenario == 'still_hard' else "Hard"
            ax2.axhline(y=heuristic_income, color=color, linestyle='--', linewidth=2, 
                       alpha=0.7, label=f'Heuristic ({scenerio_text})')
    
    ax2.set_ylabel('Income Per Category')
    ax2.set_title('Income Per Category Comparison Across Scenarios (with Heuristic Policy)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(display_model_names, rotation=0, ha='center')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'category_metrics_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Category metrics comparison saved to {output_path}")
    
    # 同时保存PDF格式
    pdf_path = os.path.join(output_dir, 'category_metrics_comparison.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Category metrics comparison saved to {pdf_path}")
    
    plt.close()

def plot_selected_models_metrics(all_stats, output_dir='paper_data_analysis'):
    """绘制选定模型的 Sales Per Category 和 Income Per Category 柱状图
    只显示三个模型：Grok-4.1 Fast, Gemini-3 (Fast), DeepSeek-V3.2 (Exp.)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except:
        print("matplotlib not available, skipping selected models plot")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 只处理这三个场景
    scenarios_to_plot = ['still_middle', 'still_hard', 'dynamic_hard']
    
    # 场景显示名称映射
    scenario_display_names = {
        'still_middle': 'Easy',
        'still_hard': 'Middle',
        'dynamic_hard': 'Hard'
    }
    
    # 模型名称映射
    model_name_mapping = {
        'deepseekv3_2': 'DeepSeek-V3.2 (Exp.)',
        'gemini3_fast': 'Gemini-3 (Fast)',
        'glm4_6': 'GLM-4.6',
        'grok4_fast': 'Grok-4.1 Fast',
        'kimi_thinking': 'Kimi-K2 (Thinking)',
        'gpt5_1mini': 'OpenAI-5.1 Mini',
        'qwen_235b': 'Qwen-235B'
    }
    
    # 反向映射：显示名称 -> 原始名称
    reverse_mapping = {v: k for k, v in model_name_mapping.items()}
    
    # 只选择这三个模型
    selected_display_names = ['Grok-4.1 Fast', 'Gemini-3 (Fast)', 'DeepSeek-V3.2 (Exp.)']
    selected_model_names = [reverse_mapping[name] for name in selected_display_names if name in reverse_mapping]
    
    if not selected_model_names:
        print("Selected models not found")
        return
    
    # 场景颜色映射
    scenario_colors = {
        'still_middle': '#3989BD',
        'still_hard': '#FF8F2B',
        'dynamic_hard': '#50B150'
    }
    
    # 准备数据：每个场景每个模型的两个指标
    sales_data = {scenario: [] for scenario in scenarios_to_plot}
    profit_data = {scenario: [] for scenario in scenarios_to_plot}
    
    for scenario in scenarios_to_plot:
        if scenario not in all_stats:
            continue
        stats = all_stats[scenario]
        for model_name in selected_model_names:
            if model_name in stats:
                data = stats[model_name]
                sales_data[scenario].append(data.get('avg_daily_sales_per_category', 0.0))
                profit_data[scenario].append(data.get('avg_daily_profit_per_category', 0.0))
            else:
                sales_data[scenario].append(0.0)
                profit_data[scenario].append(0.0)
    
    # 创建单个图，将两个指标放在一起
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    x = np.arange(len(selected_model_names))
    width = 0.13  # 每个柱子宽度
    
    # 将原始模型名称映射为显示名称
    display_model_names = [model_name_mapping.get(name, name) for name in selected_model_names]
    
    # 为每个模型绘制柱状图
    # 每个模型位置：3个场景，每个场景有2个柱子（Sales和Income）
    for model_idx in range(len(selected_model_names)):
        base_x = model_idx
        
        # 为每个场景绘制Sales和Income
        for i, scenario in enumerate(scenarios_to_plot):
            if scenario not in all_stats:
                continue
            scenario_offset = (i - 1) * width * 2.5  # 场景之间的间距
            
            # 绘制Sales数据（实心柱，alpha=0.8）
            color = scenario_colors.get(scenario, 'gray')
            sales_val = sales_data[scenario][model_idx]
            scenario_display = scenario_display_names.get(scenario, scenario.upper())
            if sales_val > 0:
                # 只在第一个模型时添加图例标签
                label = f'{scenario_display} (Sales)' if model_idx == 0 else ''
                ax.bar(base_x + scenario_offset, sales_val, width, color=color, alpha=0.8, 
                      label=label)
                # 添加数值标签
                ax.text(base_x + scenario_offset, sales_val, f'{sales_val:.1f}',
                       ha='center', va='bottom')
            
            # 绘制Income数据（带边框的柱，alpha=0.5）
            income_val = profit_data[scenario][model_idx]
            if income_val > 0:
                # 只在第一个模型时添加图例标签
                label = f'{scenario_display} (Income)' if model_idx == 0 else ''
                ax.bar(base_x + scenario_offset + width, income_val, width, 
                      color=color, alpha=0.5, edgecolor=color, linewidth=2,
                      label=label)
                # 添加数值标签
                ax.text(base_x + scenario_offset + width, income_val, f'{income_val:.1f}',
                       ha='center', va='bottom')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Value Per Category')
    ax.set_title('Sales and Income Per Category Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(display_model_names, rotation=0, ha='center')
    ax.legend(loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'selected_models_metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Selected models metrics saved to {output_path}")
    
    # 同时保存PDF格式
    pdf_path = os.path.join(output_dir, 'selected_models_metrics.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Selected models metrics saved to {pdf_path}")
    
    plt.close()

def main():
    paper_data_dir = 'paper_data'
    
    print("开始分析 paper_data 文件夹...")
    scenario_data = analyze_paper_data(paper_data_dir)
    
    if not scenario_data:
        print("未找到任何数据！")
        return
    
    print(f"\n找到 {len(scenario_data)} 个场景的数据")
    for scenario, models in scenario_data.items():
        print(f"  {scenario}: {len(models)} 个模型")
    
    print("\n计算统计指标...")
    all_stats = calculate_statistics(scenario_data)
    
    print_statistics(all_stats)
    save_statistics(all_stats)
    
    print("\n绘制 networth 变化轨迹...")
    plot_networth_trajectory(all_stats)
    
    print("\n绘制类别指标柱状图...")
    plot_category_metrics(all_stats)
    
    print("\n绘制选定模型的类别指标柱状图...")
    plot_selected_models_metrics(all_stats)
    
    print("\n分析完成！")

if __name__ == '__main__':
    main()

