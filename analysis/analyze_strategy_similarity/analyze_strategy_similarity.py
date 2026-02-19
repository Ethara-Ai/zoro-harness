#!/usr/bin/env python3
"""
分析策略相似度

功能：
计算每天的策略与之前策略的相似度（macro_strategy 和 exec_strategy）
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional, Tuple
import multiprocessing as mp
from functools import partial
from openai import OpenAI

DEFAULT_API_KEY = ''
DEFAULT_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'

def create_openai_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
    """
    Create OpenAI client with configurable API key and base URL.
    
    Args:
        api_key: OpenAI API key. If None, uses DEFAULT_API_KEY.
        base_url: Base URL for API. If None, uses DEFAULT_BASE_URL.
    
    Returns:
        Configured OpenAI client instance.
    """
    # Use provided values or defaults (no environment variable fallback)
    if api_key is None:
        api_key = DEFAULT_API_KEY
    if base_url is None:
        base_url = DEFAULT_BASE_URL
    
    return OpenAI(api_key=api_key, base_url=base_url)

def load_final_strategies(run_dir: Path) -> Dict[int, Dict]:
    """
    加载所有天的 final_strategy.json 文件
    
    返回: {day: strategy_data}
    """
    strategies = {}
    
    # 查找所有 day_X_final_strategy.json 文件
    for strategy_file in run_dir.glob('day_*_final_strategy.json'):
        match = re.search(r'day_(\d+)_final_strategy\.json', strategy_file.name)
        if match:
            day = int(match.group(1))
            try:
                with open(strategy_file, 'r', encoding='utf-8') as f:
                    strategies[day] = json.load(f)
            except Exception as e:
                print(f"Error reading {strategy_file}: {e}")
                continue
    
    return strategies

def calculate_macro_strategy_similarity_llm(
    strategy1: Dict,
    strategy2: Dict
) -> Optional[float]:
    """
    使用大模型计算两个宏观策略的相似度
    
    Args:
        strategy1: 第一个策略（包含 macro_strategy）
        strategy2: 第二个策略
    
    Returns:
        相似度分数 (0-1)，如果失败返回 None
    """
    try:
        client = create_openai_client()
        
        # 格式化 macro_strategy
        macro1 = strategy1.get('strategy', {}).get('macro_strategy', [])
        macro2 = strategy2.get('strategy', {}).get('macro_strategy', [])
        
        # 将列表格式化为易读的字符串
        def format_macro_strategy(macro_list):
            if not macro_list:
                return "Empty"
            formatted = []
            for i, item in enumerate(macro_list, 1):
                formatted.append(f"{i}. {item}")
            return "\n".join(formatted)
        
        macro1_formatted = format_macro_strategy(macro1)
        macro2_formatted = format_macro_strategy(macro2)
        
        prompt = f"""Please compare the similarity of the following two macro strategies.

Strategy 1's macro_strategy:
{macro1_formatted}

Strategy 2's macro_strategy:
{macro2_formatted}

Please evaluate the similarity between these two strategies and provide a score between 0 and 1, where:
- 1.0 means identical or almost identical
- 0.8-0.9 means very similar with only minor differences
- 0.6-0.7 means somewhat similar with some common points
- 0.4-0.5 means somewhat similar but with significant differences
- 0.2-0.3 means not very similar
- 0.0-0.1 means completely different

Please return only a floating-point number between 0 and 1, without any additional text or explanation."""
        
        response = client.chat.completions.create(
            model='qwen3-235b-a22b-instruct-2507',
            messages=[
                {"role": "system", "content": "You are a strategy analysis expert skilled at evaluating strategy similarity. Please return only a floating-point number between 0 and 1."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=50
        )
        
        result = response.choices[0].message.content.strip()
        # 尝试提取数字
        try:
            similarity = float(result)
            # 确保在 0-1 范围内
            similarity = max(0.0, min(1.0, similarity))
            return similarity
        except ValueError:
            # 如果返回的不是纯数字，尝试提取数字
            numbers = re.findall(r'\d+\.?\d*', result)
            if numbers:
                similarity = float(numbers[0])
                similarity = max(0.0, min(1.0, similarity))
                return similarity
            return None
        
    except Exception as e:
        print(f"Error calculating similarity with LLM: {e}")
        return None

def calculate_execute_strategy_similarity(
    strategy1: Dict,
    strategy2: Dict
) -> float:
    """
    使用集合交集/并集计算两个执行策略的相似度
    
    针对以下字段计算总交集/总并集：
    - focus_skus
    - sku_supplier_mapping
    - news_to_monitor
    - sku_to_monitor
    
    相似度 = 总交集数量 / 总并集数量
    
    Args:
        strategy1: 第一个策略（包含 execute_strategy）
        strategy2: 第二个策略
    
    Returns:
        相似度分数 (0-1)
    """
    exec1 = strategy1.get('strategy', {}).get('execute_strategy', {})
    exec2 = strategy2.get('strategy', {}).get('execute_strategy', {})
    
    # 将所有四个字段的元素合并到一个集合中
    # 使用字符串前缀来区分不同类型的元素
    set1 = set()
    set2 = set()
    
    # 1. focus_skus: 字符串列表
    for sku in exec1.get('focus_skus', []):
        set1.add(f"focus_sku:{sku}")
    for sku in exec2.get('focus_skus', []):
        set2.add(f"focus_sku:{sku}")
    
    # 2. sku_supplier_mapping: 对象数组，转换为字符串形式
    for m in exec1.get('sku_supplier_mapping', []):
        if isinstance(m, dict):
            sku_id = m.get('sku_id', '')
            supplier_id = m.get('supplier_id', '')
            if sku_id and supplier_id:
                set1.add(f"mapping:{sku_id}:{supplier_id}")
    
    for m in exec2.get('sku_supplier_mapping', []):
        if isinstance(m, dict):
            sku_id = m.get('sku_id', '')
            supplier_id = m.get('supplier_id', '')
            if sku_id and supplier_id:
                set2.add(f"mapping:{sku_id}:{supplier_id}")
    
    # 3. news_to_monitor: 字符串列表
    for news in exec1.get('news_to_monitor', []):
        set1.add(f"news:{news}")
    for news in exec2.get('news_to_monitor', []):
        set2.add(f"news:{news}")
    
    # 4. sku_to_monitor: 字符串列表
    for sku in exec1.get('sku_to_monitor', []):
        set1.add(f"monitor_sku:{sku}")
    for sku in exec2.get('sku_to_monitor', []):
        set2.add(f"monitor_sku:{sku}")
    
    # 计算总交集和总并集
    if set1 or set2:
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        similarity = intersection / union if union > 0 else 0.0
        return similarity
    else:
        # 如果所有字段都为空，返回 1.0（认为完全相同）
        return 1.0

def analyze_run(run_dir: Path) -> Dict:
    """
    分析单个 run 目录
    
    返回分析结果
    """
    result = {
        'run_id': run_dir.name,
        'daily_similarities': {
            'macro': {},
            'exec': {},
            'both': {}
        }
    }
    
    # 加载所有策略
    strategies = load_final_strategies(run_dir)
    
    if not strategies:
        return result
    
    # 计算每天与之前策略的相似度
    sorted_days = sorted(strategies.keys())
    
    for i, day in enumerate(sorted_days):
        current_strategy = strategies[day]
        
        # 与前一天比较
        if i > 0:
            prev_day = sorted_days[i-1]
            prev_strategy = strategies[prev_day]
            
            # 提取策略内容用于保存
            prev_macro = prev_strategy.get('strategy', {}).get('macro_strategy', [])
            prev_exec = prev_strategy.get('strategy', {}).get('execute_strategy', {})
            current_macro = current_strategy.get('strategy', {}).get('macro_strategy', [])
            current_exec = current_strategy.get('strategy', {}).get('execute_strategy', {})
            
            # 计算宏观策略相似度（使用 LLM）
            macro_sim = calculate_macro_strategy_similarity_llm(
                prev_strategy, current_strategy
            )
            if macro_sim is not None:
                result['daily_similarities']['macro'][day] = {
                    'compared_with_day': prev_day,
                    'similarity': macro_sim,
                    'prev_strategy': {
                        'macro_strategy': prev_macro
                    },
                    'current_strategy': {
                        'macro_strategy': current_macro
                    }
                }
            
            # 计算微观策略相似度（使用集合交集/并集）
            exec_sim = calculate_execute_strategy_similarity(
                prev_strategy, current_strategy
            )
            result['daily_similarities']['exec'][day] = {
                'compared_with_day': prev_day,
                'similarity': exec_sim,
                'prev_strategy': {
                    'execute_strategy': prev_exec
                },
                'current_strategy': {
                    'execute_strategy': current_exec
                }
            }
            
            # 计算最终相似度 = (宏观策略相似度 + 微观策略相似度) / 2
            if macro_sim is not None:
                final_sim = (macro_sim + exec_sim) / 2.0
            else:
                # 如果 LLM 计算失败，只使用微观策略相似度
                final_sim = exec_sim
            
            result['daily_similarities']['both'][day] = {
                'compared_with_day': prev_day,
                'similarity': final_sim,
                'macro_similarity': macro_sim,  # 记录宏观分数
                'exec_similarity': exec_sim,    # 记录微观分数
                'prev_strategy': {
                    'macro_strategy': prev_macro,
                    'execute_strategy': prev_exec
                },
                'current_strategy': {
                    'macro_strategy': current_macro,
                    'execute_strategy': current_exec
                }
            }
    
    return result

def analyze_run_wrapper(args: Tuple) -> Tuple[str, str, str, Dict]:
    """
    包装函数，用于多进程处理
    """
    scenario_name, model_name, run_dir = args
    try:
        result = analyze_run(run_dir)
        return (scenario_name, model_name, run_dir.name, result)
    except Exception as e:
        print(f"    错误处理 {run_dir.name}: {e}")
        return (scenario_name, model_name, run_dir.name, None)

def analyze_paper_data(
    paper_data_dir: str = 'paper_data',
    num_processes: int = None,
    scenario: Optional[str] = None,
    model_name: Optional[str] = None,
    run_id: Optional[str] = None,
    run_path: Optional[str] = None
):
    """
    分析 paper_data 目录下的所有数据（支持多进程）
    
    Args:
        paper_data_dir: paper_data 目录路径
        num_processes: 进程数，None 表示使用 CPU 核心数
        scenario: 可选，指定场景名称（如 'still_middle'）
        model_name: 可选，指定模型名称（如 'gpt-4o-mini'）
        run_id: 可选，指定 run_id（如 'run_env_2025-12-27_02-02-30'）
        run_path: 可选，直接指定 run 目录的完整路径
    """
    all_results = defaultdict(lambda: defaultdict(dict))
    
    # 如果指定了 run_path，直接分析该路径
    if run_path:
        run_dir = Path(run_path)
        if not run_dir.exists() or not run_dir.is_dir():
            print(f"错误: 指定的 run_path 不存在或不是目录: {run_path}")
            return dict(all_results)
        
        # 尝试从路径推断 scenario 和 model_name
        scenario_name = "unknown"
        model_name_val = "unknown"
        
        # 尝试从路径中提取：paper_data/{scenario}/{model}/{run_id}
        parts = run_dir.parts
        if 'paper_data' in parts:
            idx = parts.index('paper_data')
            if idx + 1 < len(parts):
                scenario_name = parts[idx + 1]
            if idx + 2 < len(parts):
                model_name_val = parts[idx + 2]
        
        print(f"分析指定的 run: {run_path}")
        print(f"  场景: {scenario_name}, 模型: {model_name_val}, run_id: {run_dir.name}")
        
        result = analyze_run(run_dir)
        all_results[scenario_name][model_name_val][run_dir.name] = result
        
        return dict(all_results)
    
    paper_data_path = Path(paper_data_dir)
    
    # 收集所有需要处理的 run 目录
    tasks = []
    for scenario_dir in paper_data_path.iterdir():
        if not scenario_dir.is_dir():
            continue
        
        scenario_name = scenario_dir.name
        
        # 如果指定了 scenario，只处理匹配的场景
        if scenario and scenario_name != scenario:
            continue
        
        for model_dir in scenario_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_name_val = model_dir.name
            
            # 如果指定了 model_name，只处理匹配的模型
            if model_name and model_name_val != model_name:
                continue
            
            for run_dir in model_dir.iterdir():
                if not run_dir.is_dir() or not run_dir.name.startswith('run_env_'):
                    continue
                
                # 如果指定了 run_id，只处理匹配的 run
                if run_id and run_dir.name != run_id:
                    continue
                
                tasks.append((scenario_name, model_name_val, run_dir))
    
    if not tasks:
        print("未找到匹配的运行目录！")
        return dict(all_results)
    
    print(f"开始分析策略相似度...")
    print(f"找到 {len(tasks)} 个运行目录")
    
    # 如果只有一个任务，不使用多进程
    if len(tasks) == 1:
        print("单任务模式，不使用多进程")
        scenario_name, model_name_val, run_dir = tasks[0]
        result = analyze_run(run_dir)
        all_results[scenario_name][model_name_val][run_dir.name] = result
    else:
        if num_processes is None:
            num_processes = mp.cpu_count()
        
        print(f"使用 {num_processes} 个进程进行处理")
        
        # 使用多进程处理
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(analyze_run_wrapper, tasks)
        
        # 整理结果
        for scenario_name, model_name_val, run_id_val, result in results:
            if result is not None:
                all_results[scenario_name][model_name_val][run_id_val] = result
    
    return dict(all_results)

def save_results(results: Dict, output_dir: str = 'strategy_analysis'):
    """
    保存分析结果
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'strategy_similarity_analysis.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n分析结果已保存到 {output_path}")

def print_summary(results: Dict):
    """
    打印统计摘要
    """
    print("\n" + "="*100)
    print("策略相似度分析摘要")
    print("="*100)
    
    # 计算总运行数
    total_runs = sum(len(runs) for models in results.values() for runs in models.values())
    
    for scenario, models in results.items():
        print(f"\n场景: {scenario}")
        print("-" * 100)
        
        for model_name, runs in models.items():
            print(f"\n  模型: {model_name}")
            
            run_count = len(runs)
            similarity_scores_macro = []
            similarity_scores_exec = []
            similarity_scores_both = []
            
            for run_id, run_data in runs.items():
                # 收集相似度分数
                similarities = run_data.get('daily_similarities', {})
                for day, sim_data in similarities.get('macro', {}).items():
                    similarity_scores_macro.append(sim_data.get('similarity', 0))
                for day, sim_data in similarities.get('exec', {}).items():
                    similarity_scores_exec.append(sim_data.get('similarity', 0))
                for day, sim_data in similarities.get('both', {}).items():
                    similarity_scores_both.append(sim_data.get('similarity', 0))
                
                # 如果是单个 run，显示详细信息
                if total_runs == 1:
                    print(f"\n    Run ID: {run_id}")
                    macro_sims = similarities.get('macro', {})
                    exec_sims = similarities.get('exec', {})
                    both_sims = similarities.get('both', {})
                    
                    if macro_sims or exec_sims or both_sims:
                        print(f"    每日相似度详情:")
                        days = sorted(set(list(macro_sims.keys()) + list(exec_sims.keys()) + list(both_sims.keys())))
                        for day in days:
                            both_data = both_sims.get(day, {})
                            macro_sim = both_data.get('macro_similarity')
                            exec_sim = both_data.get('exec_similarity')
                            final_sim = both_data.get('similarity')
                            
                            if macro_sim is not None:
                                print(f"      Day {day}: 宏观={macro_sim:.3f}, 微观={exec_sim:.3f}, 整体={final_sim:.3f}")
                            else:
                                print(f"      Day {day}: 微观={exec_sim:.3f}, 整体={final_sim:.3f}")
            
            print(f"    运行数: {run_count}")
            
            if similarity_scores_macro:
                print(f"    策略相似度（与前一天比较）:")
                print(f"      - macro_strategy 平均相似度: {sum(similarity_scores_macro) / len(similarity_scores_macro):.3f}")
                print(f"      - execute_strategy 平均相似度: {sum(similarity_scores_exec) / len(similarity_scores_exec):.3f}")
                print(f"      - 整体策略平均相似度: {sum(similarity_scores_both) / len(similarity_scores_both):.3f}")
            elif similarity_scores_exec:
                print(f"    策略相似度（与前一天比较）:")
                print(f"      - execute_strategy 平均相似度: {sum(similarity_scores_exec) / len(similarity_scores_exec):.3f}")
                print(f"      - 整体策略平均相似度: {sum(similarity_scores_both) / len(similarity_scores_both):.3f}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='分析策略相似度')
    parser.add_argument('--paper-data-dir', type=str, default='paper_data',
                        help='paper_data 目录路径')
    parser.add_argument('--output-dir', type=str, default='strategy_analysis',
                        help='输出目录')
    parser.add_argument('--num-processes', type=int, default=None,
                        help='进程数（默认使用 CPU 核心数）')
    parser.add_argument('--scenario', type=str, default=None,
                        help='可选，指定场景名称（如 still_middle）')
    parser.add_argument('--model-name', type=str, default=None,
                        help='可选，指定模型名称（如 gpt-4o-mini）')
    parser.add_argument('--run-id', type=str, default=None,
                        help='可选，指定 run_id（如 run_env_2025-12-27_02-02-30）')
    parser.add_argument('--run-path', type=str, default=None,
                        help='可选，直接指定 run 目录的完整路径（优先级最高）')
    
    args = parser.parse_args()
    
    results = analyze_paper_data(
        paper_data_dir=args.paper_data_dir,
        num_processes=args.num_processes,
        scenario=args.scenario,
        model_name=args.model_name,
        run_id=args.run_id,
        run_path=args.run_path
    )
    
    save_results(results, output_dir=args.output_dir)
    print_summary(results)
    
    print("\n分析完成！")

if __name__ == '__main__':
    main()
