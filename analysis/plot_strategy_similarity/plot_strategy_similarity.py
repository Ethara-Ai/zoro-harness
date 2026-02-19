#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot strategy similarity curves (paper style like reference), NO confidence interval,
and ADD weighted score plotting.

Weighted score per day per run:
    weighted = w_macro * macro + w_exec * exec
Only computed when both macro & exec exist for that day in that run.

Exports:
- PNG (300 dpi) + PDF (vector)
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, DefaultDict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from scipy.interpolate import UnivariateSpline
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

    
# =========================
# Plot style (reference-aligned)
# =========================
def set_paper_style_like_reference():
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




# =========================
# Data loading
# =========================
def load_similarity_data(json_path: str) -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# Aggregation
# =========================
AggType = DefaultDict[str, DefaultDict[str, DefaultDict[str, DefaultDict[int, List[float]]]]]


def aggregate_similarities_by_model_scenario_with_weighted(
    data: Dict,
    target_scenarios: List[str],
    w_macro: float,
    w_exec: float
) -> AggType:
    """
    aggregated[scenario][model][sim_type][day] = [values...]

    sim_type includes:
      - macro
      - exec
      - both  (if present in input)
      - weighted (computed from macro & exec per run/day)
    """
    aggregated: AggType = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    for scenario_name, scenario_data in data.items():
        if scenario_name not in target_scenarios:
            continue

        for model_name, model_data in scenario_data.items():
            for _, run_data in model_data.items():
                daily_similarities = run_data.get("daily_similarities", {})

                # 1) 原始类型：macro/exec/both
                for sim_type in ["macro", "exec", "both"]:
                    sim_block = daily_similarities.get(sim_type, {})
                    for day_str, day_data in sim_block.items():
                        if isinstance(day_data, dict) and "similarity" in day_data:
                            aggregated[scenario_name][model_name][sim_type][int(day_str)].append(
                                float(day_data["similarity"])
                            )

                # 2) weighted：以 run/day 为粒度先算，再聚合
                macro_block = daily_similarities.get("macro", {})
                exec_block = daily_similarities.get("exec", {})

                # 取交集 day（同一天同时有 macro 和 exec 才能算 weighted）
                common_days = set(macro_block.keys()) & set(exec_block.keys())
                for day_str in common_days:
                    m = macro_block.get(day_str, {})
                    e = exec_block.get(day_str, {})
                    if not (isinstance(m, dict) and "similarity" in m and isinstance(e, dict) and "similarity" in e):
                        continue
                    weighted_val = w_macro * float(m["similarity"]) + w_exec * float(e["similarity"])
                    aggregated[scenario_name][model_name]["weighted"][int(day_str)].append(weighted_val)

    return aggregated


def calculate_average_curves(
    aggregated_data: AggType
) -> Dict[str, Dict[str, Dict[str, Tuple[List[int], List[float]]]]]:
    """
    avg[scenario][model][sim_type] = (days, means)
    """
    result: Dict[str, Dict[str, Dict[str, Tuple[List[int], List[float]]]]] = defaultdict(lambda: defaultdict(dict))

    for scenario_name, scenario_data in aggregated_data.items():
        for model_name, model_data in scenario_data.items():
            for sim_type, day_data in model_data.items():
                days = sorted(day_data.keys())
                means: List[float] = []
                for day in days:
                    vals = day_data[day]
                    means.append(float(np.mean(vals)) if len(vals) > 0 else np.nan)
                result[scenario_name][model_name][sim_type] = (days, means)

    return result


# =========================
# Smoothing
# =========================
def smooth_curve(days: List[int], values: List[float], smoothing_factor: float = 0.5) -> Tuple[List[int], List[float]]:
    """
    平滑曲线数据
    
    Args:
        days: 天数列表
        values: 对应的值列表
        smoothing_factor: 平滑因子，0-1之间，越大越平滑
    
    Returns:
        平滑后的 (days, smoothed_values)
    """
    if len(days) < 3:
        return days, values
    
    days_arr = np.array(days, dtype=float)
    values_arr = np.array(values, dtype=float)
    
    # 移除 NaN 值
    valid_mask = ~np.isnan(values_arr)
    if valid_mask.sum() < 3:
        return days, values
    
    days_valid = days_arr[valid_mask]
    values_valid = values_arr[valid_mask]
    
    if HAS_SCIPY:
        # 使用 UnivariateSpline 进行平滑
        try:
            # s 参数控制平滑程度，值越大越平滑
            # 根据数据点数量和范围调整 s
            s = len(values_valid) * (1 - smoothing_factor) * np.var(values_valid)
            spline = UnivariateSpline(days_valid, values_valid, s=s)
            days_smooth = days_valid
            values_smooth = spline(days_smooth)
        except:
            # 如果插值失败，使用 Savitzky-Golay 滤波器
            try:
                from scipy.signal import savgol_filter
                window_size = max(3, min(int(len(values_valid) * smoothing_factor * 0.5), len(values_valid)))
                if window_size % 2 == 0:
                    window_size += 1
                if window_size > len(values_valid):
                    window_size = len(values_valid) if len(values_valid) % 2 == 1 else len(values_valid) - 1
                poly_order = min(3, window_size - 1)
                if window_size >= 3 and poly_order >= 1:
                    values_smooth = savgol_filter(values_valid, window_size, poly_order)
                else:
                    values_smooth = values_valid
                days_smooth = days_valid
            except:
                # 如果都失败，使用简单移动平均
                window_size = max(3, min(int(len(values_valid) * smoothing_factor * 0.3), len(values_valid)))
                if window_size % 2 == 0:
                    window_size += 1
                values_smooth = np.convolve(values_valid, np.ones(window_size)/window_size, mode='same')
                days_smooth = days_valid
    else:
        # 使用简单的移动平均
        window_size = max(3, int(len(values_valid) * smoothing_factor * 0.3))
        if window_size % 2 == 0:
            window_size += 1
        
        # 计算移动平均
        values_smooth = np.convolve(values_valid, np.ones(window_size)/window_size, mode='same')
        days_smooth = days_valid
    
    # 重建完整数组（包含 NaN 的位置）
    result_values = np.full_like(values_arr, np.nan)
    result_values[valid_mask] = values_smooth
    
    return days, result_values.tolist()


# =========================
# Plotting (NO CI)
# =========================
def plot_scenario_similarity(
    scenario_name: str,
    scenario_data: Dict[str, Dict[str, Tuple[List[int], List[float]]]],
    output_dir: Path,
    similarity_type: str,
    *,
    fig_width_in: float = 8.0,
    fig_height_in: float = 6.0,
    y_lim: Tuple[float, float] = (0.0, 1.05),
    smooth: bool = True,
    smoothing_factor: float = 0.5,
):
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib not installed")

    # 获取模型数量，动态生成颜色
    # 只包含有数据的模型
    model_names = []
    for name, data in scenario_data.items():
        if similarity_type in data:
            days, means = data[similarity_type]
            if days:  # 确保有数据点
                model_names.append(name)
    model_names = sorted(model_names)
    num_models = len(model_names)
    
    # 使用更易区分的颜色方案
    # 如果模型数量较少，使用预定义的易区分颜色
    # 如果模型数量较多，使用 colormap 生成均匀分布的颜色
    if num_models <= 10:
        # 使用精心挑选的易区分颜色（基于 ColorBrewer 和 perceptually uniform）
        predefined_colors = [
            '#1f77b4',  # 蓝色
            '#ff7f0e',  # 橙色
            '#2ca02c',  # 绿色
            '#d62728',  # 红色
            '#9467bd',  # 紫色
            '#8c564b',  # 棕色
            '#e377c2',  # 粉色
            '#7f7f7f',  # 灰色
            '#bcbd22',  # 黄绿色
            '#17becf',  # 青色
        ]
        colors = predefined_colors[:num_models]
    else:
        # 使用 tab20 colormap 生成更多颜色（最多20种易区分颜色）
        # 如果超过20个，使用 hsv colormap 循环
        if num_models <= 20:
            try:
                cmap = plt.cm.get_cmap('tab20')
            except AttributeError:
                # 新版本 matplotlib 使用直接访问
                cmap = plt.cm.tab20
            colors = [cmap(i / max(num_models - 1, 1)) for i in range(num_models)]
        else:
            # 超过20个模型时，使用 hsv 色相均匀分布
            try:
                cmap = plt.cm.get_cmap('hsv')
            except AttributeError:
                cmap = plt.cm.hsv
            colors = [cmap(i / num_models) for i in range(num_models)]

    label_map = {
        "macro": "Macro Strategy Similarity",
        "exec": "Execute Strategy Similarity",
        "both": "Overall Strategy Similarity",
        "weighted": "Weighted Score",
    }
    
    # 标题类型映射
    title_type_map = {
        "macro": "Macro Strategy Score",
        "exec": "Execution Strategy Score",
        "both": "Overall Strategy Score",
        "weighted": "Weighted Strategy Score",
    }

    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))

    for idx, model_name in enumerate(model_names):
        model_block = scenario_data[model_name]
        if similarity_type not in model_block:
            continue

        days, means = model_block[similarity_type]
        if not days:
            continue

        # 平滑处理
        if smooth:
            days, means = smooth_curve(days, means, smoothing_factor)

        ax.plot(
            days,
            means,
            label=model_name,
            color=colors[idx],
            linewidth=2.0,  # 稍微加粗线条，更易区分
            alpha=0.85,     # 稍微提高不透明度
        )

    ax.set_xlabel("Days")
    ax.set_ylabel(label_map.get(similarity_type, similarity_type))

    scene_map = {
        'still_middle': "Easy Environment",
        'still_hard': "Hard Environment",
        'dynamic_hard': "Dynamic Hard Environment",
    }

    # 标题中明确显示是 macro 还是 execution 的分数
    title_type = title_type_map.get(similarity_type, "Strategy Score")
    ax.set_title(
        f"{scene_map.get(scenario_name, scenario_name)} - {title_type}",
        # fontweight="bold",
    )

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
    ax.legend(loc="best", frameon=True, fancybox=False, edgecolor="black", framealpha=0.9)

    ax.set_xlim(left=0)
    ax.set_ylim(*y_lim)

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / f"strategy_similarity_{scenario_name}_{similarity_type}"

    plt.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")

    plt.close(fig)
    print(f"Saved: {base}.png / {base}.pdf")


# =========================
# Statistics: Mean and Std Dev per Run
# =========================
def calculate_temporal_volatility(values: List[float]) -> Dict[str, float]:
    """
    计算时间序列的波动性指标
    
    Args:
        values: 按时间顺序排列的数值列表（已排序）
    
    Returns:
        包含三个波动性指标的字典:
        - std_diff: 一阶差分的标准差 std(x_t - x_{t-1})
        - mac: 平均绝对变化 mean(|x_t - x_{t-1}|)
        - tv: 总变差 sum(|x_t - x_{t-1}|)
    """
    if len(values) < 2:
        return {
            'std_diff': np.nan,
            'mac': np.nan,
            'tv': np.nan
        }
    
    values_arr = np.array(values, dtype=float)
    
    # 计算一阶差分
    first_order_diffs = np.diff(values_arr)
    
    # 1. 一阶差分的标准差
    std_diff = float(np.std(first_order_diffs))
    
    # 2. 平均绝对变化 (Mean Absolute Change)
    mac = float(np.mean(np.abs(first_order_diffs)))
    
    # 3. 总变差 (Total Variation)
    tv = float(np.sum(np.abs(first_order_diffs)))
    
    return {
        'std_diff': std_diff,
        'mac': mac,
        'tv': tv
    }


def calculate_run_statistics(
    data: Dict,
    target_scenarios: List[str] = None
) -> Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]]:
    """
    计算每个run中macro和exec相似度的平均值和标准差
    
    返回结构:
    {
        scenario: {
            model: {
                run_id: {
                    'macro': {'mean': float, 'std': float, 'count': int},
                    'exec': {'mean': float, 'std': float, 'count': int}
                }
            }
        }
    }
    
    Args:
        data: 策略相似度分析数据
        target_scenarios: 目标场景列表，None表示所有场景
    """
    if target_scenarios is None:
        target_scenarios = ['still_middle', 'still_hard', 'dynamic_hard']
    
    result = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for scenario_name, scenario_data in data.items():
        if scenario_name not in target_scenarios:
            continue
        
        for model_name, model_data in scenario_data.items():
            for run_id, run_data in model_data.items():
                daily_similarities = run_data.get('daily_similarities', {})
                
                # 收集macro相似度值（保留时间顺序）
                macro_day_values = []
                macro_block = daily_similarities.get('macro', {})
                for day_str, day_data in macro_block.items():
                    if isinstance(day_data, dict) and 'similarity' in day_data:
                        day_int = int(day_str)
                        macro_day_values.append((day_int, float(day_data['similarity'])))
                
                # 按天数排序
                macro_day_values.sort(key=lambda x: x[0])
                macro_values = [v for _, v in macro_day_values]
                
                # 收集exec相似度值（保留时间顺序）
                exec_day_values = []
                exec_block = daily_similarities.get('exec', {})
                for day_str, day_data in exec_block.items():
                    if isinstance(day_data, dict) and 'similarity' in day_data:
                        day_int = int(day_str)
                        exec_day_values.append((day_int, float(day_data['similarity'])))
                
                # 按天数排序
                exec_day_values.sort(key=lambda x: x[0])
                exec_values = [v for _, v in exec_day_values]
                
                # 计算macro统计量
                macro_stats = {}
                if macro_values:
                    macro_volatility = calculate_temporal_volatility(macro_values)
                    macro_stats = {
                        'mean': float(np.mean(macro_values)),
                        'std': float(np.std(macro_values)),
                        'count': len(macro_values),
                        'min': float(np.min(macro_values)),
                        'max': float(np.max(macro_values)),
                        'std_diff': macro_volatility['std_diff'],
                        'mac': macro_volatility['mac'],
                        'tv': macro_volatility['tv']
                    }
                else:
                    macro_stats = {
                        'mean': np.nan,
                        'std': np.nan,
                        'count': 0,
                        'min': np.nan,
                        'max': np.nan,
                        'std_diff': np.nan,
                        'mac': np.nan,
                        'tv': np.nan
                    }
                
                # 计算exec统计量
                exec_stats = {}
                if exec_values:
                    exec_volatility = calculate_temporal_volatility(exec_values)
                    exec_stats = {
                        'mean': float(np.mean(exec_values)),
                        'std': float(np.std(exec_values)),
                        'count': len(exec_values),
                        'min': float(np.min(exec_values)),
                        'max': float(np.max(exec_values)),
                        'std_diff': exec_volatility['std_diff'],
                        'mac': exec_volatility['mac'],
                        'tv': exec_volatility['tv']
                    }
                else:
                    exec_stats = {
                        'mean': np.nan,
                        'std': np.nan,
                        'count': 0,
                        'min': np.nan,
                        'max': np.nan,
                        'std_diff': np.nan,
                        'mac': np.nan,
                        'tv': np.nan
                    }
                
                result[scenario_name][model_name][run_id] = {
                    'macro': macro_stats,
                    'exec': exec_stats
                }
    
    return dict(result)


def aggregate_statistics_by_model(
    run_stats: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]]
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """
    按模型聚合统计信息（计算所有run的平均值）
    
    返回结构:
    {
        scenario: {
            model: {
                'macro': {'mean': float, 'std': float, 'std_of_means': float},
                'exec': {'mean': float, 'std': float, 'std_of_means': float}
            }
        }
    }
    
    Args:
        run_stats: calculate_run_statistics 的输出
    """
    result = defaultdict(lambda: defaultdict(dict))
    
    for scenario_name, scenario_data in run_stats.items():
        for model_name, model_data in scenario_data.items():
            macro_means = []
            macro_stds = []
            macro_std_diffs = []
            macro_macs = []
            macro_tvs = []
            exec_means = []
            exec_stds = []
            exec_std_diffs = []
            exec_macs = []
            exec_tvs = []
            
            for run_id, run_stat in model_data.items():
                macro_stat = run_stat.get('macro', {})
                exec_stat = run_stat.get('exec', {})
                
                if macro_stat.get('count', 0) > 0:
                    macro_mean = macro_stat.get('mean')
                    macro_std = macro_stat.get('std')
                    macro_std_diff = macro_stat.get('std_diff')
                    macro_mac = macro_stat.get('mac')
                    macro_tv = macro_stat.get('tv')
                    if not np.isnan(macro_mean):
                        macro_means.append(macro_mean)
                    if not np.isnan(macro_std):
                        macro_stds.append(macro_std)
                    if not np.isnan(macro_std_diff):
                        macro_std_diffs.append(macro_std_diff)
                    if not np.isnan(macro_mac):
                        macro_macs.append(macro_mac)
                    if not np.isnan(macro_tv):
                        macro_tvs.append(macro_tv)
                
                if exec_stat.get('count', 0) > 0:
                    exec_mean = exec_stat.get('mean')
                    exec_std = exec_stat.get('std')
                    exec_std_diff = exec_stat.get('std_diff')
                    exec_mac = exec_stat.get('mac')
                    exec_tv = exec_stat.get('tv')
                    if not np.isnan(exec_mean):
                        exec_means.append(exec_mean)
                    if not np.isnan(exec_std):
                        exec_stds.append(exec_std)
                    if not np.isnan(exec_std_diff):
                        exec_std_diffs.append(exec_std_diff)
                    if not np.isnan(exec_mac):
                        exec_macs.append(exec_mac)
                    if not np.isnan(exec_tv):
                        exec_tvs.append(exec_tv)
            
            # 计算macro聚合统计
            if macro_means:
                result[scenario_name][model_name]['macro'] = {
                    'mean': float(np.mean(macro_means)),  # 所有run的平均值的平均值
                    'std': float(np.mean(macro_stds)),    # 所有run的标准差的平均值
                    'std_of_means': float(np.std(macro_means)),  # run间平均值的标准差
                    'run_count': len(macro_means),
                    'std_diff': float(np.mean(macro_std_diffs)) if macro_std_diffs else np.nan,
                    'mac': float(np.mean(macro_macs)) if macro_macs else np.nan,
                    'tv': float(np.mean(macro_tvs)) if macro_tvs else np.nan
                }
            else:
                result[scenario_name][model_name]['macro'] = {
                    'mean': np.nan,
                    'std': np.nan,
                    'std_of_means': np.nan,
                    'run_count': 0,
                    'std_diff': np.nan,
                    'mac': np.nan,
                    'tv': np.nan
                }
            
            # 计算exec聚合统计
            if exec_means:
                result[scenario_name][model_name]['exec'] = {
                    'mean': float(np.mean(exec_means)),
                    'std': float(np.mean(exec_stds)),
                    'std_of_means': float(np.std(exec_means)),
                    'run_count': len(exec_means),
                    'std_diff': float(np.mean(exec_std_diffs)) if exec_std_diffs else np.nan,
                    'mac': float(np.mean(exec_macs)) if exec_macs else np.nan,
                    'tv': float(np.mean(exec_tvs)) if exec_tvs else np.nan
                }
            else:
                result[scenario_name][model_name]['exec'] = {
                    'mean': np.nan,
                    'std': np.nan,
                    'std_of_means': np.nan,
                    'run_count': 0,
                    'std_diff': np.nan,
                    'mac': np.nan,
                    'tv': np.nan
                }
    
    return dict(result)


def format_table_row(values: List[str], widths: List[int]) -> str:
    """格式化表格行"""
    return "| " + " | ".join(f"{v:<{w}}" for v, w in zip(values, widths)) + " |"


def print_statistics_summary(
    run_stats: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]],
    aggregated_stats: Dict[str, Dict[str, Dict[str, Dict[str, float]]]]
):
    """
    打印统计摘要（表格格式）
    
    Args:
        run_stats: 每个run的统计信息
        aggregated_stats: 按模型聚合的统计信息
    """
    print("\n" + "="*120)
    print("策略相似度统计摘要（按模型聚合）")
    print("="*120)
    
    scene_map = {
        'still_middle': "Easy Environment",
        'still_hard': "Hard Environment",
        'dynamic_hard': "Dynamic Hard Environment",
    }
    
    for scenario_name in sorted(aggregated_stats.keys()):
        scenario_display = scene_map.get(scenario_name, scenario_name.upper())
        print(f"\n{'='*120}")
        print(f"场景: {scenario_display}")
        print('='*120)
        
        scenario_data = aggregated_stats[scenario_name]
        model_names = sorted(scenario_data.keys())
        
        if not model_names:
            print("  无数据")
            continue
        
        # ===== Macro Strategy Similarity 表格 =====
        print(f"\n【Macro Strategy Similarity - 时间波动性指标】")
        print("-" * 120)
        
        # 表头
        headers = ["模型", "Mean", "Std", "Std_diff", "MAC", "TV", "稳定性评估"]
        widths = [25, 10, 10, 12, 12, 12, 15]
        print(format_table_row(headers, widths))
        print("-" * 120)
        
        # 数据行
        for model_name in model_names:
            model_stats = scenario_data[model_name]
            macro_stats = model_stats.get('macro', {})
            
            if np.isnan(macro_stats.get('mean', np.nan)):
                row = [model_name, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"]
            else:
                mean_val = macro_stats.get('mean', np.nan)
                std_val = macro_stats.get('std', np.nan)
                std_diff = macro_stats.get('std_diff', np.nan)
                mac = macro_stats.get('mac', np.nan)
                tv = macro_stats.get('tv', np.nan)
                
                # 稳定性评估
                if not np.isnan(std_diff):
                    if std_diff < 0.05:
                        stability = "稳定"
                    elif std_diff < 0.1:
                        stability = "中等波动"
                    else:
                        stability = "高波动"
                else:
                    stability = "N/A"
                
                row = [
                    model_name,
                    f"{mean_val:.4f}" if not np.isnan(mean_val) else "N/A",
                    f"{std_val:.4f}" if not np.isnan(std_val) else "N/A",
                    f"{std_diff:.4f}" if not np.isnan(std_diff) else "N/A",
                    f"{mac:.4f}" if not np.isnan(mac) else "N/A",
                    f"{tv:.4f}" if not np.isnan(tv) else "N/A",
                    stability
                ]
            
            print(format_table_row(row, widths))
        
        print("-" * 120)
        print("说明: Std_diff = 一阶差分标准差, MAC = 平均绝对变化, TV = 总变差")
        
        # ===== Execute Strategy Similarity 表格 =====
        print(f"\n【Execute Strategy Similarity - 时间波动性指标】")
        print("-" * 120)
        
        # 表头
        headers = ["模型", "Mean", "Std", "Std_diff", "MAC", "TV", "稳定性评估"]
        widths = [25, 10, 10, 12, 12, 12, 15]
        print(format_table_row(headers, widths))
        print("-" * 120)
        
        # 数据行
        for model_name in model_names:
            model_stats = scenario_data[model_name]
            exec_stats = model_stats.get('exec', {})
            
            if np.isnan(exec_stats.get('mean', np.nan)):
                row = [model_name, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"]
            else:
                mean_val = exec_stats.get('mean', np.nan)
                std_val = exec_stats.get('std', np.nan)
                std_diff = exec_stats.get('std_diff', np.nan)
                mac = exec_stats.get('mac', np.nan)
                tv = exec_stats.get('tv', np.nan)
                
                # 稳定性评估
                if not np.isnan(std_diff):
                    if std_diff < 0.05:
                        stability = "稳定"
                    elif std_diff < 0.1:
                        stability = "中等波动"
                    else:
                        stability = "高波动"
                else:
                    stability = "N/A"
                
                row = [
                    model_name,
                    f"{mean_val:.4f}" if not np.isnan(mean_val) else "N/A",
                    f"{std_val:.4f}" if not np.isnan(std_val) else "N/A",
                    f"{std_diff:.4f}" if not np.isnan(std_diff) else "N/A",
                    f"{mac:.4f}" if not np.isnan(mac) else "N/A",
                    f"{tv:.4f}" if not np.isnan(tv) else "N/A",
                    stability
                ]
            
            print(format_table_row(row, widths))
        
        print("-" * 120)
        print("说明: Std_diff = 一阶差分标准差, MAC = 平均绝对变化, TV = 总变差")


def save_statistics(
    run_stats: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]],
    aggregated_stats: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    output_path: str = 'strategy_similarity_statistics.json'
):
    """
    保存统计结果到JSON文件
    
    Args:
        run_stats: 每个run的统计信息
        aggregated_stats: 按模型聚合的统计信息
        output_path: 输出文件路径
    """
    output_data = {
        'run_level_statistics': run_stats,
        'model_level_statistics': aggregated_stats
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n统计结果已保存到: {output_path}")


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="strategy_analysis/strategy_similarity_analysis.json")
    parser.add_argument("--output-dir", type=str, default="strategy_similarity_plots")
    parser.add_argument("--scenarios", type=str, nargs="+", default=["still_middle", "still_hard", "dynamic_hard"])


    # 可选绘制类型：新增 weighted
    parser.add_argument("--types", nargs="+", default=["macro", "exec", "weighted"],
                        help="Types to plot: macro exec both weighted")

    # 加权参数
    parser.add_argument("--w-macro", type=float, default=0.5, help="Weight for macro in weighted score")
    parser.add_argument("--w-exec", type=float, default=0.5, help="Weight for exec in weighted score")

    # figure size（默认保持参考脚本 8x6）
    parser.add_argument("--fig-width", type=float, default=8.0)
    parser.add_argument("--fig-height", type=float, default=6.0)

    # y-axis range
    parser.add_argument("--ymin", type=float, default=0.0)
    parser.add_argument("--ymax", type=float, default=1.05)
    
    # 平滑选项
    parser.add_argument("--no-smooth", action="store_true",
                        help="禁用折线平滑")
    parser.add_argument("--smoothing-factor", type=float, default=0.5,
                        help="平滑因子 (0-1)，越大越平滑，默认 0.5")
    
    # 统计选项
    parser.add_argument("--calculate-stats", action="store_true",
                        help="计算并保存统计信息（平均值和标准差）")
    parser.add_argument("--stats-output", type=str, default="strategy_similarity_statistics.json",
                        help="统计结果输出文件路径")
    
    args = parser.parse_args()
    
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib 未安装：pip install matplotlib numpy")

    # 权重基本校验：允许不归一，但给出合理行为（不强制报错）
    if abs(args.w_macro) + abs(args.w_exec) == 0:
        raise ValueError("w_macro 和 w_exec 不能同时为 0")

    set_paper_style_like_reference()

    data = load_similarity_data(args.input)

    # 如果需要计算统计信息
    if args.calculate_stats:
        print("\n计算统计信息（平均值和标准差）...")
        run_stats = calculate_run_statistics(data, args.scenarios)
        aggregated_stats = aggregate_statistics_by_model(run_stats)
        print_statistics_summary(run_stats, aggregated_stats)
        save_statistics(run_stats, aggregated_stats, args.stats_output)

    aggregated = aggregate_similarities_by_model_scenario_with_weighted(
        data=data,
        target_scenarios=args.scenarios,
        w_macro=args.w_macro,
        w_exec=args.w_exec,
    )
    avg_curves = calculate_average_curves(aggregated)

    out_dir = Path(args.output_dir)
    y_lim = (args.ymin, args.ymax)

    allowed = {"macro", "exec", "both", "weighted"}
    plot_types = [t for t in args.types if t in allowed]
    if len(plot_types) == 0:
        raise ValueError(f"--types 至少包含一种支持的类型：{sorted(list(allowed))}")

    print("\n生成图表...")
    for scenario in args.scenarios:
        if scenario not in avg_curves:
            print(f"Warning: No data for scenario: {scenario}")
            continue
        for sim_type in plot_types:
            plot_scenario_similarity(
                scenario_name=scenario,
                scenario_data=avg_curves[scenario],
                output_dir=out_dir,
                similarity_type=sim_type,
                fig_width_in=args.fig_width,
                fig_height_in=args.fig_height,
                y_lim=y_lim,
                smooth=not args.no_smooth,
                smoothing_factor=args.smoothing_factor,
            )


if __name__ == "__main__":
    main()
