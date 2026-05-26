#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用模式2（es_text_only）对包含wiki文本内容的ES索引进行文本检索测评

功能：
1. 读取find.xlsx获取查询和正确链接
2. 使用模式2（es_text_only）在ES索引中进行纯文本检索
3. 计算MRR、Hit@1、Hit@5、Hit@10等指标
4. 生成详细测评报告

注意：本脚本用于测评文本检索效果，不涉及向量检索和LLM重排序
"""

import os
import sys
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# 导入配置和ES客户端
try:
    from config import WORK_DIR, ES_INDEX_NAME
    from es_client import es
except ImportError:
    WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ES_INDEX_NAME = 'data2'
    from es_client import es

# 导入search_vllm中的测评函数和模块
try:
    import search_vllm
    from search_vllm import calculate_metrics
except ImportError:
    print("错误: 无法导入search_vllm模块")
    sys.exit(1)


def read_find_excel(file_path):
    """读取find.xlsx文件"""
    df = pd.read_excel(file_path, header=None)
    queries = df[0].tolist()
    correct_links = df[1].tolist()
    
    # 清理数据，移除NaN
    queries = [str(q).strip() if pd.notna(q) else "" for q in queries]
    correct_links = [str(link).strip() if pd.notna(link) else "" for link in correct_links]
    
    # 过滤空值
    valid_pairs = [(q, link) for q, link in zip(queries, correct_links) if q and link]
    
    print(f"从 {file_path} 读取了 {len(valid_pairs)} 个有效查询-链接对")
    return valid_pairs


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='使用模式2（es_text_only）对包含wiki文本内容的ES索引进行文本检索测评',
        epilog='注意：本脚本使用纯文本检索模式，不涉及向量检索和LLM重排序'
    )
    parser.add_argument('--find-file', type=str, default=None,
                       help='find.xlsx文件路径（默认: work_wyy/data/find.xlsx）')
    parser.add_argument('--index', type=str, default=None,
                       help='ES索引名称（默认: 使用config.py中的ES_INDEX_NAME）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出报告文件路径（默认: trainlog/evaluation_wiki_content_*.json）')
    
    args = parser.parse_args()
    
    # 获取work_wyy目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    work_wyy_dir = os.path.dirname(script_dir)
    
    # 确定find文件路径
    if args.find_file:
        find_file = args.find_file
        if not os.path.isabs(find_file):
            find_file = os.path.abspath(find_file)
    else:
        find_file = os.path.join(work_wyy_dir, 'data', 'find.xlsx')
    
    if not os.path.exists(find_file):
        print(f"错误: 找不到find.xlsx文件: {find_file}")
        sys.exit(1)
    
    # 确定索引名称
    index_name = args.index if args.index else ES_INDEX_NAME
    
    # 检查索引是否存在
    if not es.indices.exists(index=index_name):
        print(f"错误: 索引 {index_name} 不存在")
        print(f"  请先运行: python store_wiki_content_to_es.py")
        sys.exit(1)
    
    print("=" * 70)
    print("使用模式2（es_text_only）测评包含wiki文本内容的ES索引")
    print("=" * 70)
    print(f"find.xlsx文件: {find_file}")
    print(f"ES索引: {index_name}")
    print(f"检索模式: es_text_only (模式2 - 纯文本检索)")
    print("=" * 70)
    
    # 读取查询数据
    find_pairs = read_find_excel(find_file)
    if not find_pairs:
        print("错误: find.xlsx中没有有效数据")
        sys.exit(1)
    
    queries = [q for q, _ in find_pairs]
    correct_links = [link for _, link in find_pairs]
    
    # 临时修改ES_INDEX_NAME（如果指定了不同的索引）
    original_index = None
    if args.index and args.index != ES_INDEX_NAME:
        # 保存原始索引名称
        original_index = search_vllm.ES_INDEX_NAME
        # 临时修改为指定索引
        search_vllm.ES_INDEX_NAME = index_name
        print(f"注意: 临时使用索引 {index_name} 进行测评（原索引: {original_index}）")
    
    try:
        # 执行测评（使用模式2：纯文本检索）
        print(f"\n开始测评，共 {len(queries)} 个查询...")
        print(f"注意: 使用纯文本检索模式，不涉及向量检索和LLM重排序")
        mrr, hit_at_1, hit_at_5, hit_at_10 = calculate_metrics(
            queries, correct_links, search_mode="es_text_only"
        )
    finally:
        # 恢复原始索引名称
        if original_index is not None:
            search_vllm.ES_INDEX_NAME = original_index
            print(f"已恢复原始索引: {original_index}")
    
    # 打印结果
    print("\n" + "=" * 70)
    print("测评结果")
    print("=" * 70)
    print(f"检索模式: es_text_only (模式2 - 纯文本检索)")
    print(f"ES索引: {index_name}")
    print(f"总查询数: {len(queries)}")
    print(f"MRR: {mrr:.4f}")
    print(f"Hit@1: {hit_at_1:.4f} ({hit_at_1*100:.2f}%)")
    print(f"Hit@5: {hit_at_5:.4f} ({hit_at_5*100:.2f}%)")
    print(f"Hit@10: {hit_at_10:.4f} ({hit_at_10*100:.2f}%)")
    print("=" * 70)
    
    # 保存报告
    if args.output:
        output_file = args.output
    else:
        trainlog_dir = os.path.join(work_wyy_dir, 'trainlog')
        os.makedirs(trainlog_dir, exist_ok=True)
        output_file = os.path.join(
            trainlog_dir,
            f'evaluation_wiki_content_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'index_name': index_name,
        'search_mode': 'es_text_only',
        'search_mode_description': '纯文本检索（不使用向量，不使用LLM）',
        'total_queries': len(queries),
        'metrics': {
            'mrr': mrr,
            'hit@1': hit_at_1,
            'hit@5': hit_at_5,
            'hit@10': hit_at_10
        }
    }
    
    import json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n测评报告已保存到: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
