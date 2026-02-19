#!/usr/bin/env python3
"""
检查 paper_data 下所有 tool_calls.jsonl 文件中的工具调用问题：
1. modify_sku_price: 新价格为 0、负数或大于 50
2. place_order: 某个 SKU 的下单数量超过 2000
"""

import json
import os
from pathlib import Path
from collections import defaultdict

def check_tool_calls(file_path):
    """检查单个文件中的工具调用问题"""
    issues = {
        'modify_sku_price': [],
        'place_order': []
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse JSON at line {line_num} in {file_path}: {e}")
                    continue
                
                tool = record.get('tool', '')
                args = record.get('args', {})
                
                # 检查 modify_sku_price
                if tool == 'modify_sku_price':
                    new_price = args.get('new_price')
                    sku_id = args.get('sku_id')
                    
                    if new_price is not None:
                        if new_price == 0:
                            issues['modify_sku_price'].append({
                                'file': str(file_path),
                                'line': line_num,
                                'sku_id': sku_id,
                                'new_price': new_price,
                                'issue': '价格为 0'
                            })
                        elif new_price < 0:
                            issues['modify_sku_price'].append({
                                'file': str(file_path),
                                'line': line_num,
                                'sku_id': sku_id,
                                'new_price': new_price,
                                'issue': '价格为负数'
                            })
                        elif new_price > 50:
                            issues['modify_sku_price'].append({
                                'file': str(file_path),
                                'line': line_num,
                                'sku_id': sku_id,
                                'new_price': new_price,
                                'issue': '价格大于 50'
                            })
                
                # 检查 place_order
                elif tool == 'place_order':
                    items = args.get('items', [])
                    
                    for item in items:
                        sku_id = item.get('sku_id')
                        quantity = item.get('quantity')
                        
                        if quantity is not None and quantity > 2000:
                            issues['place_order'].append({
                                'file': str(file_path),
                                'line': line_num,
                                'sku_id': sku_id,
                                'quantity': quantity,
                                'issue': f'下单数量 {quantity} 超过 2000'
                            })
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return issues

def main():
    # 查找所有 tool_calls.jsonl 文件
    paper_data_dir = Path('paper_data')
    
    if not paper_data_dir.exists():
        print(f"Error: {paper_data_dir} 目录不存在")
        return
    
    tool_calls_files = list(paper_data_dir.rglob('tool_calls.jsonl'))
    
    if not tool_calls_files:
        print(f"未找到任何 tool_calls.jsonl 文件")
        return
    
    print(f"找到 {len(tool_calls_files)} 个 tool_calls.jsonl 文件\n")
    
    all_issues = {
        'modify_sku_price': [],
        'place_order': []
    }
    
    # 检查每个文件
    for file_path in tool_calls_files:
        issues = check_tool_calls(file_path)
        all_issues['modify_sku_price'].extend(issues['modify_sku_price'])
        all_issues['place_order'].extend(issues['place_order'])
    
    # 输出结果
    print("=" * 80)
    print("检查结果汇总")
    print("=" * 80)
    
    # modify_sku_price 问题
    print(f"\n【modify_sku_price 问题】共 {len(all_issues['modify_sku_price'])} 个")
    print("-" * 80)
    
    if all_issues['modify_sku_price']:
        # 按问题类型分组
        by_issue_type = defaultdict(list)
        for issue in all_issues['modify_sku_price']:
            by_issue_type[issue['issue']].append(issue)
        
        for issue_type, issues_list in sorted(by_issue_type.items()):
            print(f"\n{issue_type}: {len(issues_list)} 个")
            for issue in issues_list[:10]:  # 只显示前10个
                print(f"  - 文件: {issue['file']}")
                print(f"    行号: {issue['line']}, SKU: {issue['sku_id']}, 价格: {issue['new_price']}")
            if len(issues_list) > 10:
                print(f"  ... 还有 {len(issues_list) - 10} 个类似问题")
    else:
        print("  未发现问题")
    
    # place_order 问题
    print(f"\n【place_order 问题】共 {len(all_issues['place_order'])} 个")
    print("-" * 80)
    
    if all_issues['place_order']:
        # 按 SKU 分组统计
        by_sku = defaultdict(list)
        for issue in all_issues['place_order']:
            by_sku[issue['sku_id']].append(issue)
        
        print(f"\n下单数量超过 2000 的情况:")
        for sku_id, issues_list in sorted(by_sku.items()):
            print(f"\n  SKU {sku_id}: {len(issues_list)} 次")
            for issue in issues_list[:5]:  # 只显示前5个
                print(f"    - 文件: {issue['file']}")
                print(f"      行号: {issue['line']}, 数量: {issue['quantity']}")
            if len(issues_list) > 5:
                print(f"    ... 还有 {len(issues_list) - 5} 次类似问题")
    else:
        print("  未发现问题")
    
    # 统计信息
    print("\n" + "=" * 80)
    print("统计信息")
    print("=" * 80)
    print(f"检查的文件数: {len(tool_calls_files)}")
    print(f"modify_sku_price 问题总数: {len(all_issues['modify_sku_price'])}")
    print(f"place_order 问题总数: {len(all_issues['place_order'])}")
    
    # 保存详细结果到文件
    output_file = 'tool_calls_issues.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_issues, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到: {output_file}")

if __name__ == '__main__':
    main()

