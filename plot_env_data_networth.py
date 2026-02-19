#!/usr/bin/env python3
"""
绘制 env_data 中 networth 随着 date 的变化
学术论文风格（ACL/ICML/NIPS style）
"""

import json
import os
from pathlib import Path
from datetime import datetime, date

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
    rcParams['font.size'] = 11
    rcParams['axes.labelsize'] = 11
    rcParams['axes.titlesize'] = 12
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10
    rcParams['legend.fontsize'] = 10
    rcParams['figure.titlesize'] = 12
    rcParams['axes.linewidth'] = 0.8
    rcParams['grid.linewidth'] = 0.5
    rcParams['lines.linewidth'] = 1.5
    rcParams['patch.linewidth'] = 0.5
    rcParams['xtick.major.width'] = 0.8
    rcParams['ytick.major.width'] = 0.8
    rcParams['xtick.minor.width'] = 0.6
    rcParams['ytick.minor.width'] = 0.6
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False
    rcParams['text.usetex'] = False  # 不使用 LaTeX，避免依赖
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, cannot generate plots")

def parse_date(date_str):
    """解析日期字符串"""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except:
        return None

def extract_networth_data(tool_calls_path):
    """从 tool_calls.jsonl 文件中提取 networth 和 date 数据"""
    networth_data = []
    
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
                                net_worth = result_data.get('net_worth')
                                
                                if current_date and net_worth is not None:
                                    networth_data.append({
                                        'date': current_date,
                                        'net_worth': float(net_worth)
                                    })
                except:
                    continue
    except Exception as e:
        print(f"Error reading {tool_calls_path}: {e}")
        return None
    
    if not networth_data:
        return None
    
    # 按日期排序
    networth_data.sort(key=lambda x: x['date'])
    
    return networth_data

def plot_networth_trajectory(env_data_dir='env_data', output_dir='env_data_plots'):
    """绘制 env_data 目录下所有数据的 networth 变化轨迹"""
    
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, cannot generate plots")
        return
    
    env_data_path = Path(env_data_dir)
    if not env_data_path.exists():
        print(f"Error: {env_data_dir} directory not found")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集所有场景的数据
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
        networth_data = extract_networth_data(str(tool_calls_path))
        
        if networth_data:
            scenario_data[scenario_name] = networth_data
            print(f"  Found {len(networth_data)} data points")
        else:
            print(f"  No data found")
    
    if not scenario_data:
        print("No data found in env_data directory")
        return
    
    # 学术风格的配色方案（适合黑白打印）
    colors = ['#000000', '#666666', '#999999', '#CCCCCC']  # 黑色到灰色渐变
    linestyles = ['-', '--', '-.', ':']  # 实线、虚线、点划线、点线
    
    # 创建单个图表，包含所有场景
    fig, ax = plt.subplots(figsize=(6, 4))
    
    scenario_names_sorted = sorted(scenario_data.keys())
    for idx, scenario_name in enumerate(scenario_names_sorted):
        networth_data = scenario_data[scenario_name]
        dates = []
        networths = []
        
        for entry in networth_data:
            date_obj = parse_date(entry['date'])
            if date_obj:
                dates.append(date_obj)
                networths.append(entry['net_worth'])
        
        if dates:
            # 转换为天数（从第一天开始）
            start_date = dates[0]
            day_numbers = [(d - start_date).days for d in dates]
            
            color = colors[idx % len(colors)]
            linestyle = linestyles[idx % len(linestyles)]
            ax.plot(day_numbers, networths, label=scenario_name, 
                   color=color, linestyle=linestyle, linewidth=1.5)
    
    ax.set_xlabel('Day', fontsize=11)
    ax.set_ylabel('Net Worth', fontsize=11)
    ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black', framealpha=1.0)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'networth_trajectory_all.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"\nCombined plot saved to {output_path}")
    plt.close()
    
    # 为每个场景创建单独的图表
    for scenario_name, networth_data in sorted(scenario_data.items()):
        dates = []
        networths = []
        
        for entry in networth_data:
            date_obj = parse_date(entry['date'])
            if date_obj:
                dates.append(date_obj)
                networths.append(entry['net_worth'])
        
        if not dates:
            continue
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # 转换为天数（从第一天开始）
        start_date = dates[0]
        day_numbers = [(d - start_date).days for d in dates]
        
        # 使用黑色实线，学术风格
        ax.plot(day_numbers, networths, linewidth=1.5, color='#000000')
        
        ax.set_xlabel('Day', fontsize=11)
        ax.set_ylabel('Net Worth', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'networth_trajectory_{scenario_name}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Plot for {scenario_name} saved to {output_path}")
        plt.close()
    
    # 打印统计信息
    print("\n" + "="*80)
    print("统计信息")
    print("="*80)
    
    for scenario_name, networth_data in sorted(scenario_data.items()):
        if not networth_data:
            continue
        
        networths = [entry['net_worth'] for entry in networth_data]
        dates = [entry['date'] for entry in networth_data]
        
        initial_networth = networths[0] if networths else 0
        final_networth = networths[-1] if networths else 0
        max_networth = max(networths) if networths else 0
        min_networth = min(networths) if networths else 0
        avg_networth = sum(networths) / len(networths) if networths else 0
        change = final_networth - initial_networth
        change_percent = (change / initial_networth * 100) if initial_networth > 0 else 0
        
        print(f"\n场景: {scenario_name.upper()}")
        print(f"  数据点数: {len(networth_data)}")
        print(f"  日期范围: {dates[0]} ~ {dates[-1]}")
        print(f"  初始净值: {initial_networth:.2f}")
        print(f"  最终净值: {final_networth:.2f}")
        print(f"  最高净值: {max_networth:.2f}")
        print(f"  最低净值: {min_networth:.2f}")
        print(f"  平均净值: {avg_networth:.2f}")
        print(f"  净值变化: {change:+.2f} ({change_percent:+.2f}%)")
    
    print("\n" + "="*80)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="绘制 env_data 中 networth 随着 date 的变化（学术论文风格）",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--env-data-dir',
        type=str,
        default='env_data',
        help='env_data 目录路径（默认: env_data）'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='env_data_plots',
        help='输出目录路径（默认: env_data_plots）'
    )
    
    args = parser.parse_args()
    
    print("开始分析 env_data 目录...")
    plot_networth_trajectory(args.env_data_dir, args.output_dir)
    print("\n分析完成！")

if __name__ == '__main__':
    main()






