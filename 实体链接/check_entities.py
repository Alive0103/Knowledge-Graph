#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查ES中是否存在特定实体的工具
"""

from es_client import es

def check_entity_exists(entity_name, index_name="data2"):
    """检查特定实体是否存在于ES中"""
    try:
        # 使用match查询检查实体是否存在
        query = {
            "query": {
                "multi_match": {
                    "query": entity_name,
                    "fields": ["label^2", "aliases_zh", "aliases_en"]
                }
            },
            "size": 10
        }
        
        response = es.search(index=index_name, body=query)
        hits = response['hits']['hits']
        
        if hits:
            print(f"✓ 在ES中找到 '{entity_name}' 的相关实体:")
            for i, hit in enumerate(hits, 1):
                source = hit['_source']
                score = hit['_score']
                label = source.get('label', 'N/A')
                print(f"  {i}. 标题: {label} (得分: {score:.4f})")
                if source.get('aliases_zh'):
                    print(f"     中文别名: {', '.join(source.get('aliases_zh', []))}")
                if source.get('aliases_en'):
                    print(f"     英文别名: {', '.join(source.get('aliases_en', []))}")
                print(f"     链接: {source.get('link', 'N/A')}")
                print()
        else:
            print(f"✗ 在ES中未找到与 '{entity_name}' 相关的实体")
            
        return len(hits) > 0
        
    except Exception as e:
        print(f"✗ 查询过程中发生错误: {e}")
        return False

def check_vector_fields(entity_name, index_name="data2"):
    """检查特定实体的向量字段是否存在"""
    try:
        # 先查找实体
        query = {
            "query": {
                "multi_match": {
                    "query": entity_name,
                    "fields": ["label^2", "aliases_zh", "aliases_en"]
                }
            },
            "_source": ["label", "descriptions_zh_vector", "descriptions_en_vector", "content_vector"],
            "size": 5
        }
        
        response = es.search(index=index_name, body=query)
        hits = response['hits']['hits']
        
        if hits:
            print(f"✓ '{entity_name}' 的向量字段检查:")
            for i, hit in enumerate(hits, 1):
                source = hit['_source']
                label = source.get('label', 'N/A')
                print(f"  {i}. 实体: {label}")
                
                # 检查各种向量字段
                vec_fields = ['descriptions_zh_vector', 'descriptions_en_vector', 'content_vector']
                for field in vec_fields:
                    if field in source and source[field]:
                        vec_len = len(source[field]) if isinstance(source[field], list) else "非列表格式"
                        print(f"     {field}: 存在 (长度: {vec_len})")
                    else:
                        print(f"     {field}: 不存在或为空")
                print()
        else:
            print(f"✗ 未找到实体 '{entity_name}'")
            
    except Exception as e:
        print(f"✗ 检查向量字段时发生错误: {e}")

def main():
    """主函数"""
    print("ES实体检查工具")
    print("="*50)
    
    # 测试实体列表
    test_entities = [
        "AK-47",
        "F-16",
        "步枪",
        "坦克",
        "战斗机"
    ]
    
    for entity in test_entities:
        print(f"\n检查实体: '{entity}'")
        print("-"*30)
        check_entity_exists(entity)
        check_vector_fields(entity)
        print("-"*50)

if __name__ == "__main__":
    main()