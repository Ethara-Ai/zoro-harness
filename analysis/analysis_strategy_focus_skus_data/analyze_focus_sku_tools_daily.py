#!/usr/bin/env python3
"""
统计每个run的strategy阶段，每天关注到的focus_skus以及这些SKU调用了哪些工具
输出格式：每个run按天统计focus_skus和对应的工具调用
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Any
import argparse

# 定义可能查询SKU数据的工具（用于识别哪些工具可能包含SKU参数）
SKU_QUERY_TOOLS = {
    'view_sku_sales_history',
    'view_current_date_supplier_prices',
    'view_supplier_price_history',
    'view_sku_prices',
    'view_sku_reviews',
    'view_sku_avg_ratings',
    'view_return_rates',
}

# 其他可能相关的工具（不直接查询SKU，但可能间接相关）
OTHER_TOOLS = {
    'view_inventory',
    'view_funds_and_date',
    'view_current_orders',
    'view_notes',
    'view_today_news',
    'view_news_detail',
    'view_news_history',
    'set_macro_strategy',
    'set_execute_strategy',
    'set_action',
}


def extract_sku_from_tool_call(tool_call: Dict[str, Any]) -> Set[str]:
    """从工具调用中提取SKU ID列表"""
    skus = set()
    tool_name = tool_call.get('name', '')
    arguments = tool_call.get('arguments', {})
    
    # 检查是否包含sku_ids参数
    if 'sku_ids' in arguments:
        sku_ids = arguments['sku_ids']
        if isinstance(sku_ids, list):
            skus.update(sku_ids)
        elif isinstance(sku_ids, str):
            skus.add(sku_ids)
    
    # 检查是否包含sku_id参数（单数形式）
    if 'sku_id' in arguments:
        sku_id = arguments['sku_id']
        if isinstance(sku_id, str):
            skus.add(sku_id)
    
    return skus


def analyze_strategy_phase(run_dir: Path, scenario: str = None, model: str = None) -> Dict[str, Any]:
    """分析单个run的strategy阶段，按天统计focus_skus和工具调用"""
    result = {
        'run_id': run_dir.name,
        'scenario': scenario,
        'model': model,
        'days': {}
    }
    
    # 获取所有天的目录
    day_dirs = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.isdigit()], 
                      key=lambda x: int(x.name))
    
    for day_dir in day_dirs:
        day = int(day_dir.name)
        
        # 读取最终策略文件，获取focus_skus
        final_strategy_file = run_dir / f'day_{day}_final_strategy.json'
        if not final_strategy_file.exists():
            continue
        
        try:
            with open(final_strategy_file, 'r', encoding='utf-8') as f:
                final_strategy = json.load(f)
            
            focus_skus = final_strategy.get('strategy', {}).get('execute_strategy', {}).get('focus_skus', [])
            if not focus_skus:
                continue
        except Exception as e:
            print(f"Error reading {final_strategy_file}: {e}")
            continue
        
        # 统计该天strategy阶段的所有工具调用
        # {sku: {tool_name: count}}
        day_queries = defaultdict(lambda: defaultdict(int))
        
        # 统计所有工具调用（不区分SKU）
        all_tool_calls = defaultdict(int)  # {tool_name: count}
        
        # 遍历该天的所有strategy文件
        strategy_files = sorted(day_dir.glob('strategy_*.json'), 
                               key=lambda x: int(x.stem.split('_')[-1]))
        
        for strategy_file in strategy_files:
            try:
                with open(strategy_file, 'r', encoding='utf-8') as f:
                    strategy_data = json.load(f)
                
                tool_calls = strategy_data.get('tool_calls', [])
                
                for tool_call in tool_calls:
                    tool_name = tool_call.get('name', '')
                    if not tool_name:
                        continue
                    
                    # 统计所有工具调用
                    all_tool_calls[tool_name] += 1
                    
                    # 检查是否是查询SKU数据的工具
                    if tool_name in SKU_QUERY_TOOLS:
                        skus = extract_sku_from_tool_call(tool_call)
                        # 对于每个查询到的SKU，记录工具调用（只记录focus_skus中的）
                        for sku in skus:
                            if sku in focus_skus:
                                day_queries[sku][tool_name] += 1
                    
                    # 对于view_inventory，可能返回所有SKU
                    elif tool_name == 'view_inventory':
                        # 对于focus_skus，标记为查询了库存
                        for sku in focus_skus:
                            day_queries[sku][tool_name] += 1
                    
                    # 对于其他工具，如果它们被调用，也记录（但不关联到特定SKU）
                    # 这里我们只记录与focus_skus相关的工具调用
                    elif tool_name in OTHER_TOOLS:
                        # 这些工具可能间接影响所有SKU，但不直接查询特定SKU
                        # 可以选择记录或不记录，这里先不记录到特定SKU
                        pass
            
            except Exception as e:
                print(f"Error reading {strategy_file}: {e}")
                continue
        
        # 只保留focus_skus的查询信息
        focus_sku_queries = {}
        for sku in focus_skus:
            if sku in day_queries:
                # 直接使用工具名称作为key，值为调用次数
                focus_sku_queries[sku] = dict(day_queries[sku])
            else:
                # 即使没有查询记录，也记录下来
                focus_sku_queries[sku] = {}
        
        # 添加该天所有工具调用的统计
        day_tool_stats = dict(all_tool_calls)
        
        if focus_sku_queries:
            result['days'][day] = {
                'focus_skus': focus_skus,
                'sku_tool_calls': focus_sku_queries,  # {sku: {tool: count}}
                'all_tool_calls': day_tool_stats  # 该天所有工具调用统计
            }
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='统计每个run每天focus_skus的工具调用情况',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--paper-data-dir',
        type=str,
        default='paper_data',
        help='paper_data 目录路径（默认: paper_data）'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='focus_sku_tools_daily.json',
        help='输出JSON文件路径（默认: focus_sku_tools_daily.json）'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='只分析特定模型（例如: gpt5_1mini）'
    )
    
    parser.add_argument(
        '--scenario',
        type=str,
        help='只分析特定场景（例如: still_middle）'
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.paper_data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        return
    
    # 查找所有run目录
    run_dirs = []
    
    # 根据参数确定搜索路径
    if args.scenario and args.model:
        search_path = data_dir / args.scenario / args.model
    elif args.scenario:
        search_path = data_dir / args.scenario
    elif args.model:
        # 在所有场景下搜索该模型
        for scenario_dir in data_dir.iterdir():
            if scenario_dir.is_dir():
                model_dir = scenario_dir / args.model
                if model_dir.exists():
                    run_dirs.extend([(d, scenario_dir.name, args.model) for d in model_dir.iterdir() 
                                   if d.is_dir() and d.name.startswith('run_env_')])
        search_path = None
    else:
        search_path = data_dir
    
    if search_path and search_path.exists():
        # 从路径中提取scenario和model
        for run_dir in search_path.rglob('run_env_*'):
            if run_dir.is_dir():
                # 提取scenario和model
                parts = run_dir.parts
                try:
                    # paper_data/{scenario}/{model}/{run_env_xxx}
                    paper_data_idx = parts.index('paper_data') if 'paper_data' in parts else -1
                    if paper_data_idx >= 0 and len(parts) > paper_data_idx + 3:
                        scenario = parts[paper_data_idx + 1]
                        model = parts[paper_data_idx + 2]
                        run_dirs.append((run_dir, scenario, model))
                except:
                    # 如果无法提取，使用None
                    run_dirs.append((run_dir, None, None))
    
    if not run_dirs:
        print(f"No run directories found in {data_dir}")
        return
    
    print(f"Found {len(run_dirs)} run directories")
    
    # 分析每个run
    all_results = []
    for i, (run_dir, scenario, model) in enumerate(run_dirs, 1):
        print(f"Processing {i}/{len(run_dirs)}: {scenario}/{model}/{run_dir.name}")
        try:
            result = analyze_strategy_phase(run_dir, scenario, model)
            if result['days']:
                all_results.append(result)
        except Exception as e:
            print(f"Error processing {run_dir}: {e}")
            continue
    
    print(f"\nSuccessfully analyzed {len(all_results)} runs")
    
    # 保存结果
    output_path = Path(args.output)
    output_data = {
        'summary': {
            'total_runs': len(all_results),
            'total_days': sum(len(r['days']) for r in all_results),
        },
        'runs': all_results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_path}")
    
    # 打印摘要
    print("\n=== Summary ===")
    print(f"Total runs analyzed: {len(all_results)}")
    print(f"Total days with focus_skus: {output_data['summary']['total_days']}")
    
    # 统计每个模型-场景的组合
    model_scenario_counts = defaultdict(int)
    for result in all_results:
        key = f"{result['model']}/{result['scenario']}"
        model_scenario_counts[key] += 1
    
    print("\nRuns by Model/Scenario:")
    for key, count in sorted(model_scenario_counts.items()):
        print(f"  {key}: {count} runs")


if __name__ == '__main__':
    main()

