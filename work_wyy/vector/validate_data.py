#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据文件格式校验脚本
检查待处理的 JSONL 文件是否符合向量生成的要求
"""

import json
import os
from collections import defaultdict

def validate_jsonl_file(file_path, sample_size=100):
    """
    校验 JSONL 文件格式
    
    Args:
        file_path: JSONL 文件路径
        sample_size: 抽样检查的文档数量
    """
    print("=" * 70)
    print(f"开始校验数据文件: {file_path}")
    print("=" * 70)
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return False
    
    # 统计信息
    total_lines = 0
    valid_lines = 0
    invalid_lines = 0
    
    # 字段统计
    field_stats = defaultdict(int)
    missing_fields = defaultdict(int)
    
    # 向量生成所需字段
    required_fields_for_vector = {
        'zh': ['label', 'zh_description', 'zh_aliases'],
        'en': ['label', 'en_description', 'en_aliases']
    }
    
    # 错误详情
    errors = []
    warnings = []
    
    # 样本文档
    sample_docs = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            
            # 跳过空行
            if not line.strip():
                continue
            
            try:
                # 解析 JSON
                data = json.loads(line.strip())
                valid_lines += 1
                
                # 收集字段
                for key in data.keys():
                    field_stats[key] += 1
                
                # 检查必要字段
                label = data.get("label", "")
                descriptions_zh = data.get("zh_description") or data.get("descriptions_zh", "")
                descriptions_en = data.get("en_description") or data.get("descriptions_en", "")
                aliases_zh = data.get("zh_aliases") or data.get("aliases_zh", [])
                aliases_en = data.get("en_aliases") or data.get("aliases_en", [])
                
                # 检查中文向量生成条件
                can_generate_zh_vector = False
                if descriptions_zh and len(descriptions_zh.strip()) > 10:
                    can_generate_zh_vector = True
                elif label:
                    can_generate_zh_vector = True
                
                # 检查英文向量生成条件
                can_generate_en_vector = False
                if descriptions_en and len(descriptions_en.strip()) > 10:
                    can_generate_en_vector = True
                elif label:
                    can_generate_en_vector = True
                
                # 检查 label_vector 生成条件
                can_generate_label_zh_vector = bool(label)
                can_generate_label_en_vector = bool(label)
                
                # 记录缺失字段
                if not label:
                    missing_fields['label'] += 1
                if not descriptions_zh and not descriptions_zh:
                    missing_fields['descriptions_zh'] += 1
                if not descriptions_en and not descriptions_en:
                    missing_fields['descriptions_en'] += 1
                
                # 收集样本文档（前 sample_size 个）
                if len(sample_docs) < sample_size:
                    sample_docs.append({
                        'line_num': line_num,
                        'label': label,
                        'has_descriptions_zh': bool(descriptions_zh and len(descriptions_zh.strip()) > 10),
                        'has_descriptions_en': bool(descriptions_en and len(descriptions_en.strip()) > 10),
                        'has_aliases_zh': bool(aliases_zh and isinstance(aliases_zh, list) and len(aliases_zh) > 0),
                        'has_aliases_en': bool(aliases_en and isinstance(aliases_en, list) and len(aliases_en) > 0),
                        'can_generate_zh_vector': can_generate_zh_vector,
                        'can_generate_en_vector': can_generate_en_vector,
                        'can_generate_label_zh_vector': can_generate_label_zh_vector,
                        'can_generate_label_en_vector': can_generate_label_en_vector,
                        'descriptions_zh_len': len(descriptions_zh) if descriptions_zh else 0,
                        'descriptions_en_len': len(descriptions_en) if descriptions_en else 0,
                    })
                
                # 检查是否有问题
                if not label:
                    errors.append(f"第 {line_num} 行: 缺少 label 字段")
                if not can_generate_zh_vector and not can_generate_en_vector:
                    warnings.append(f"第 {line_num} 行: 无法生成任何向量（缺少描述和标签）")
                
            except json.JSONDecodeError as e:
                invalid_lines += 1
                errors.append(f"第 {line_num} 行: JSON 解析失败 - {str(e)[:100]}")
            except Exception as e:
                invalid_lines += 1
                errors.append(f"第 {line_num} 行: 处理失败 - {str(e)[:100]}")
    
    # 输出统计结果
    print(f"\n📊 文件统计:")
    print(f"  总行数: {total_lines}")
    print(f"  有效行数: {valid_lines}")
    print(f"  无效行数: {invalid_lines}")
    
    print(f"\n📋 字段统计（出现次数）:")
    for field, count in sorted(field_stats.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / valid_lines * 100) if valid_lines > 0 else 0
        print(f"  {field}: {count} ({percentage:.1f}%)")
    
    if missing_fields:
        print(f"\n⚠️  缺失字段统计:")
        for field, count in sorted(missing_fields.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / valid_lines * 100) if valid_lines > 0 else 0
            print(f"  {field}: {count} ({percentage:.1f}%)")
    
    # 分析样本文档
    if sample_docs:
        print(f"\n📝 样本文档分析（前 {len(sample_docs)} 个）:")
        
        zh_vector_count = sum(1 for d in sample_docs if d['can_generate_zh_vector'])
        en_vector_count = sum(1 for d in sample_docs if d['can_generate_en_vector'])
        label_zh_vector_count = sum(1 for d in sample_docs if d['can_generate_label_zh_vector'])
        label_en_vector_count = sum(1 for d in sample_docs if d['can_generate_label_en_vector'])
        
        print(f"  可生成 descriptions_zh_vector: {zh_vector_count}/{len(sample_docs)} ({zh_vector_count/len(sample_docs)*100:.1f}%)")
        print(f"  可生成 descriptions_en_vector: {en_vector_count}/{len(sample_docs)} ({en_vector_count/len(sample_docs)*100:.1f}%)")
        print(f"  可生成 label_zh_vector: {label_zh_vector_count}/{len(sample_docs)} ({label_zh_vector_count/len(sample_docs)*100:.1f}%)")
        print(f"  可生成 label_en_vector: {label_en_vector_count}/{len(sample_docs)} ({label_en_vector_count/len(sample_docs)*100:.1f}%)")
        
        # 显示前5个样本文档的详细信息
        print(f"\n  前5个样本文档详情:")
        for i, doc in enumerate(sample_docs[:5], 1):
            print(f"    {i}. 行 {doc['line_num']}: {doc['label'][:30] if doc['label'] else 'N/A'}")
            print(f"       描述(zh): {'✓' if doc['has_descriptions_zh'] else '✗'} ({doc['descriptions_zh_len']}字)")
            print(f"       描述(en): {'✓' if doc['has_descriptions_en'] else '✗'} ({doc['descriptions_en_len']}字)")
            print(f"       别名(zh): {'✓' if doc['has_aliases_zh'] else '✗'}")
            print(f"       别名(en): {'✓' if doc['has_aliases_en'] else '✗'}")
            print(f"       可生成向量: zh_desc={doc['can_generate_zh_vector']}, en_desc={doc['can_generate_en_vector']}, zh_label={doc['can_generate_label_zh_vector']}, en_label={doc['can_generate_label_en_vector']}")
    
    # 输出错误和警告
    if errors:
        print(f"\n❌ 错误 ({len(errors)} 个，显示前10个）:")
        for error in errors[:10]:
            print(f"  {error}")
        if len(errors) > 10:
            print(f"  ... 还有 {len(errors) - 10} 个错误")
    
    if warnings:
        print(f"\n⚠️  警告 ({len(warnings)} 个，显示前10个）:")
        for warning in warnings[:10]:
            print(f"  {warning}")
        if len(warnings) > 10:
            print(f"  ... 还有 {len(warnings) - 10} 个警告")
    
    # 总结
    print(f"\n{'=' * 70}")
    if invalid_lines == 0 and len(errors) == 0:
        print("✅ 数据文件格式校验通过！")
        print(f"   可以生成向量的文档比例:")
        print(f"   - descriptions_zh_vector: {zh_vector_count/len(sample_docs)*100:.1f}%")
        print(f"   - descriptions_en_vector: {en_vector_count/len(sample_docs)*100:.1f}%")
        print(f"   - label_zh_vector: {label_zh_vector_count/len(sample_docs)*100:.1f}%")
        print(f"   - label_en_vector: {label_en_vector_count/len(sample_docs)*100:.1f}%")
        return True
    else:
        print("❌ 数据文件格式有问题，请先修复后再导入")
        return False
    print(f"{'=' * 70}\n")


def main():
    """主函数"""
    import sys
    
    # 默认检查的数据文件
    data_files = [
        "zh_wiki_v2.jsonl",
        "en_wiki_v3.jsonl"
    ]
    
    # 如果提供了命令行参数，使用参数指定的文件
    if len(sys.argv) > 1:
        data_files = sys.argv[1:]
    
    print("=" * 70)
    print("数据文件格式校验工具")
    print("=" * 70)
    print()
    
    all_valid = True
    for file_path in data_files:
        if os.path.exists(file_path):
            valid = validate_jsonl_file(file_path)
            if not valid:
                all_valid = False
            print()
        else:
            print(f"⚠️  文件不存在: {file_path}\n")
    
    if all_valid:
        print("✅ 所有数据文件校验通过，可以开始导入！")
        return 0
    else:
        print("❌ 部分数据文件有问题，请先修复后再导入")
        return 1


if __name__ == "__main__":
    exit(main())
