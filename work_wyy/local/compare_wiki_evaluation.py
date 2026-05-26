#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比添加wiki内容前后的文本检索测评结果

功能：
1. 读取两个测评报告JSON文件（添加wiki前和添加wiki后）
2. 对比各项指标（MRR、Hit@1、Hit@5、Hit@10）
3. 计算提升幅度
4. 生成对比报告
"""

import os
import sys
import json
import argparse
from datetime import datetime


def load_evaluation_report(file_path):
    """加载测评报告JSON文件"""
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在: {file_path}")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        return report
    except Exception as e:
        print(f"错误: 读取文件失败 {file_path}: {e}")
        return None


def compare_reports(before_report, after_report):
    """对比两个测评报告"""
    if not before_report or not after_report:
        return None
    
    before_metrics = before_report.get('metrics', {})
    after_metrics = after_report.get('metrics', {})
    
    if not before_metrics or not after_metrics:
        print("错误: 报告文件中缺少metrics字段")
        return None
    
    comparison = {
        'before': {
            'index_name': before_report.get('index_name', 'N/A'),
            'timestamp': before_report.get('timestamp', 'N/A'),
            'total_queries': before_report.get('total_queries', 0),
            'metrics': before_metrics
        },
        'after': {
            'index_name': after_report.get('index_name', 'N/A'),
            'timestamp': after_report.get('timestamp', 'N/A'),
            'total_queries': after_report.get('total_queries', 0),
            'metrics': after_metrics
        },
        'improvement': {}
    }
    
    # 计算各项指标的提升
    metrics_to_compare = ['mrr', 'hit@1', 'hit@5', 'hit@10']
    
    for metric in metrics_to_compare:
        before_value = before_metrics.get(metric, 0)
        after_value = after_metrics.get(metric, 0)
        
        if before_value > 0:
            improvement_abs = after_value - before_value
            improvement_rel = (improvement_abs / before_value) * 100
        else:
            improvement_abs = after_value
            improvement_rel = 100.0 if after_value > 0 else 0.0
        
        comparison['improvement'][metric] = {
            'before': before_value,
            'after': after_value,
            'absolute': improvement_abs,
            'relative_percent': improvement_rel
        }
    
    return comparison


def print_comparison(comparison):
    """打印对比结果"""
    if not comparison:
        return
    
    print("\n" + "=" * 70)
    print("添加Wiki内容前后文本检索效果对比")
    print("=" * 70)
    
    print(f"\n【测评信息】")
    print(f"  添加前索引: {comparison['before']['index_name']}")
    print(f"  添加前时间: {comparison['before']['timestamp']}")
    print(f"  添加后索引: {comparison['after']['index_name']}")
    print(f"  添加后时间: {comparison['after']['timestamp']}")
    print(f"  查询总数: {comparison['before']['total_queries']}")
    
    print(f"\n【指标对比】")
    print(f"  {'指标':<15} {'添加前':<12} {'添加后':<12} {'绝对提升':<12} {'相对提升':<12}")
    print(f"  {'-'*70}")
    
    metrics_display = {
        'mrr': 'MRR',
        'hit@1': 'Hit@1',
        'hit@5': 'Hit@5',
        'hit@10': 'Hit@10'
    }
    
    for metric, display_name in metrics_display.items():
        imp = comparison['improvement'][metric]
        before = imp['before']
        after = imp['after']
        abs_imp = imp['absolute']
        rel_imp = imp['relative_percent']
        
        # 格式化显示
        before_str = f"{before:.4f}"
        after_str = f"{after:.4f}"
        abs_str = f"{abs_imp:+.4f}" if abs_imp != 0 else "0.0000"
        rel_str = f"{rel_imp:+.2f}%" if rel_imp != 0 else "0.00%"
        
        # 根据提升情况添加颜色标记（在终端中）
        if abs_imp > 0:
            marker = "↑"
        elif abs_imp < 0:
            marker = "↓"
        else:
            marker = "="
        
        print(f"  {display_name:<15} {before_str:<12} {after_str:<12} {abs_str:<12} {rel_str:<12} {marker}")
    
    print(f"\n  {'-'*70}")
    
    # 总结
    total_improvements = sum(1 for imp in comparison['improvement'].values() if imp['absolute'] > 0)
    total_metrics = len(comparison['improvement'])
    
    print(f"\n【总结】")
    print(f"  提升的指标数: {total_improvements}/{total_metrics}")
    
    avg_mrr_imp = comparison['improvement']['mrr']['relative_percent']
    avg_hit1_imp = comparison['improvement']['hit@1']['relative_percent']
    
    print(f"  MRR相对提升: {avg_mrr_imp:+.2f}%")
    print(f"  Hit@1相对提升: {avg_hit1_imp:+.2f}%")
    
    if avg_mrr_imp > 0 or avg_hit1_imp > 0:
        print(f"  ✓ 添加wiki内容后，文本检索效果有所提升")
    elif avg_mrr_imp < 0 or avg_hit1_imp < 0:
        print(f"  ⚠ 添加wiki内容后，部分指标有所下降，可能需要调整搜索权重")
    else:
        print(f"  = 添加wiki内容后，文本检索效果基本不变")
    
    print("=" * 70)


def save_comparison_report(comparison, output_file):
    """保存对比报告到JSON文件"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'comparison': comparison
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n对比报告已保存到: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='对比添加wiki内容前后的文本检索测评结果',
        epilog='示例: python compare_wiki_evaluation.py --before before.json --after after.json'
    )
    parser.add_argument('--before', type=str, required=True,
                       help='添加wiki内容前的测评报告JSON文件路径')
    parser.add_argument('--after', type=str, required=True,
                       help='添加wiki内容后的测评报告JSON文件路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出对比报告文件路径（默认: trainlog/comparison_wiki_*.json）')
    
    args = parser.parse_args()
    
    # 加载两个报告
    print("=" * 70)
    print("加载测评报告")
    print("=" * 70)
    print(f"添加前报告: {args.before}")
    print(f"添加后报告: {args.after}")
    
    before_report = load_evaluation_report(args.before)
    after_report = load_evaluation_report(args.after)
    
    if not before_report or not after_report:
        print("错误: 无法加载测评报告")
        sys.exit(1)
    
    # 对比报告
    comparison = compare_reports(before_report, after_report)
    
    if not comparison:
        print("错误: 对比失败")
        sys.exit(1)
    
    # 打印对比结果
    print_comparison(comparison)
    
    # 保存对比报告
    if args.output:
        output_file = args.output
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        work_wyy_dir = os.path.dirname(script_dir)
        trainlog_dir = os.path.join(work_wyy_dir, 'trainlog')
        os.makedirs(trainlog_dir, exist_ok=True)
        output_file = os.path.join(
            trainlog_dir,
            f'comparison_wiki_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
    
    save_comparison_report(comparison, output_file)
    print("=" * 70)


if __name__ == "__main__":
    main()
