#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查向量数据质量的脚本
"""

import json
from ..es_client import es

INDEX_NAME = "data2"

def check_vector_data_quality(file_path, sample_size=100):
    """检查向量数据文件的质量"""
    print(f"检查向量数据文件: {file_path}")
    
    vector_count = 0
    total_count = 0
    dimension_issues = 0
    format_issues = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                    
                try:
                    data = json.loads(line.strip())
                    total_count += 1
                    
                    # 检查中文向量
                    zh_vector = data.get("descriptions_zh_vector")
                    if zh_vector:
                        if isinstance(zh_vector, list):
                            if len(zh_vector) == 1024:
                                vector_count += 1
                            else:
                                dimension_issues += 1
                                print(f"第{i+1}行中文向量维度错误: {len(zh_vector)}")
                        else:
                            format_issues += 1
                            print(f"第{i+1}行中文向量格式错误: {type(zh_vector)}")
                    
                    # 检查英文向量
                    en_vector = data.get("descriptions_en_vector")
                    if en_vector:
                        if isinstance(en_vector, list):
                            if len(en_vector) == 1024:
                                vector_count += 1
                            else:
                                dimension_issues += 1
                                print(f"第{i+1}行英文向量维度错误: {len(en_vector)}")
                        else:
                            format_issues += 1
                            print(f"第{i+1}行英文向量格式错误: {type(en_vector)}")
                            
                except json.JSONDecodeError as e:
                    print(f"第{i+1}行JSON解析错误: {e}")
                    
        print(f"\n检查结果:")
        print(f"  总检查条数: {total_count}")
        print(f"  有效向量数: {vector_count}")
        print(f"  维度错误数: {dimension_issues}")
        print(f"  格式错误数: {format_issues}")
        
    except FileNotFoundError:
        print(f"文件不存在: {file_path}")
    except Exception as e:
        print(f"检查过程中发生错误: {e}")

def check_es_vector_quality(sample_size=100):
    """检查ES中向量数据的质量"""
    print(f"检查ES索引 {INDEX_NAME} 中的向量数据质量")
    
    try:
        # 获取文档总数
        total_docs = es.count(index=INDEX_NAME)['count']
        print(f"索引中文档总数: {total_docs}")
        
        # 随机采样检查
        query = {
            "size": sample_size,
            "query": {
                "function_score": {
                    "query": {"match_all": {}},
                    "random_score": {}
                }
            },
            "_source": ["label", "descriptions_zh_vector", "descriptions_en_vector"]
        }
        
        response = es.search(index=INDEX_NAME, body=query)
        hits = response['hits']['hits']
        
        vector_count = 0
        dimension_issues = 0
        missing_issues = 0
        
        print(f"\n检查 {len(hits)} 个样本文档:")
        
        for hit in hits:
            source = hit['_source']
            label = source.get('label', 'N/A')
            
            # 检查中文向量
            if 'descriptions_zh_vector' in source:
                zh_vector = source['descriptions_zh_vector']
                if isinstance(zh_vector, list) and len(zh_vector) == 1024:
                    vector_count += 1
                else:
                    dimension_issues += 1
                    print(f"  {label}: 中文向量维度错误 ({len(zh_vector) if isinstance(zh_vector, list) else '非列表'})")
            else:
                missing_issues += 1
                # print(f"  {label}: 缺少中文向量")
            
            # 检查英文向量
            if 'descriptions_en_vector' in source:
                en_vector = source['descriptions_en_vector']
                if isinstance(en_vector, list) and len(en_vector) == 1024:
                    vector_count += 1
                else:
                    dimension_issues += 1
                    print(f"  {label}: 英文向量维度错误 ({len(en_vector) if isinstance(en_vector, list) else '非列表'})")
            else:
                missing_issues += 1
                # print(f"  {label}: 缺少英文向量")
        
        print(f"\n检查结果:")
        print(f"  样本数: {len(hits)}")
        print(f"  有效向量数: {vector_count}")
        print(f"  维度错误数: {dimension_issues}")
        print(f"  缺失向量数: {missing_issues}")
        print(f"  向量存在率: {((len(hits)*2 - missing_issues) / (len(hits)*2) * 100):.1f}%")
        
        # 统计向量字段的总体情况
        zh_vector_count = es.count(
            index=INDEX_NAME,
            body={"query": {"exists": {"field": "descriptions_zh_vector"}}}
        )['count']
        
        en_vector_count = es.count(
            index=INDEX_NAME,
            body={"query": {"exists": {"field": "descriptions_en_vector"}}}
        )['count']
        
        print(f"\n总体统计:")
        print(f"  包含中文向量的文档: {zh_vector_count} ({zh_vector_count/total_docs*100:.1f}%)")
        print(f"  包含英文向量的文档: {en_vector_count} ({en_vector_count/total_docs*100:.1f}%)")
        
    except Exception as e:
        print(f"检查ES向量数据时发生错误: {e}")

if __name__ == "__main__":
    print("向量数据质量检查工具")
    print("="*50)
    
    # 检查ES中的向量数据质量
    check_es_vector_quality()