"""
检查ES中实体的实际数据
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from es_client import es
from urllib.parse import unquote

def check_entity_by_link(link):
    """通过链接检查ES中的实体数据"""
    print(f"检查链接: {link}")
    
    # URL解码
    decoded_link = unquote(link)
    print(f"解码后: {decoded_link}")
    
    # 提取wiki标题
    if "/wiki/" in decoded_link:
        title = decoded_link.split("/wiki/")[1]
        print(f"Wiki标题: {title}")
    
    index_names = ["data2", "data1"]
    
    for index_name in index_names:
        try:
            if not es.indices.exists(index=index_name):
                continue
            
            print(f"\n在索引 '{index_name}' 中搜索...")
            
            # 方法1: 精确匹配link字段（使用match_phrase，因为link是text类型）
            query1 = {
                "query": {
                    "match_phrase": {
                        "link": link
                    }
                }
            }
            
            try:
                response = es.search(index=index_name, body=query1)
                if response["hits"]["total"]["value"] > 0:
                    print(f"✓ 找到实体（精确匹配link）")
                    hit = response["hits"]["hits"][0]
                    source = hit["_source"]
                    print(f"  Label: {source.get('label', 'N/A')}")
                    print(f"  Aliases_zh: {source.get('aliases_zh', [])}")
                    print(f"  Aliases_en: {source.get('aliases_en', [])}")
                    print(f"  Descriptions_zh: {source.get('descriptions_zh', '')[:200]}...")
                    print(f"  Link: {source.get('link', 'N/A')}")
                    # 验证是否真的是精确匹配
                    if source.get('link', '') == link:
                        return source
                    else:
                        print(f"  ⚠ 注意：返回的link不完全匹配")
                        print(f"     期望: {link}")
                        print(f"     实际: {source.get('link', 'N/A')}")
            except Exception as e:
                print(f"  精确匹配失败: {e}")
            
            # 方法1.5: 直接遍历所有文档查找（如果数据量不大）
            # 注意：这个方法只适用于小数据集，大数据集会很慢
            try:
                query_all = {
                    "query": {"match_all": {}},
                    "size": 10000  # 限制最多检查10000条
                }
                response = es.search(index=index_name, body=query_all, scroll='1m')
                total = response["hits"]["total"]["value"]
                print(f"\n  索引中共有 {total} 个文档，检查前10000个...")
                
                hits = response["hits"]["hits"]
                for hit in hits:
                    source = hit["_source"]
                    if source.get('link', '') == link:
                        print(f"  ✓ 找到实体（遍历匹配）")
                        print(f"  Label: {source.get('label', 'N/A')}")
                        print(f"  Aliases_zh: {source.get('aliases_zh', [])}")
                        print(f"  Link: {source.get('link', 'N/A')}")
                        return source
                
                if total > 10000:
                    print(f"  ⚠ 只检查了前10000条，还有 {total - 10000} 条未检查")
            except Exception as e:
                print(f"  遍历检查失败: {e}")
            
            # 方法2: 使用match查询link字段（包含关键词）
            if title:
                query2 = {
                    "query": {
                        "match": {
                            "link": title
                        }
                    },
                    "size": 50
                }
                
                try:
                    response = es.search(index=index_name, body=query2)
                    if response["hits"]["total"]["value"] > 0:
                        print(f"✓ 找到 {response['hits']['total']['value']} 个相关实体（link包含'{title}'）")
                        for i, hit in enumerate(response["hits"]["hits"][:10], 1):
                            source = hit["_source"]
                            link_match = "✓" if source.get('link', '') == link else " "
                            print(f"  {link_match} {i}. Label: {source.get('label', 'N/A')}")
                            print(f"      Link: {source.get('link', 'N/A')[:80]}...")
                            if source.get('link', '') == link:
                                print(f"      ✓ 这是目标实体！")
                                print(f"      Aliases_zh: {source.get('aliases_zh', [])}")
                                print(f"      Descriptions_zh: {source.get('descriptions_zh', '')[:200]}...")
                                return source
                except Exception as e:
                    print(f"  link匹配失败: {e}")
            
            # 方法3: 通过label搜索（使用"堪培拉"和"两栖"）
            query3 = {
                "query": {
                    "bool": {
                        "should": [
                            {"match": {"label": "堪培拉"}},
                            {"match": {"label": "两栖"}},
                            {"match": {"label": "攻击舰"}},
                            {"match": {"aliases_zh": "堪培拉"}},
                            {"match": {"aliases_zh": "两栖"}},
                            {"match": {"aliases_zh": "攻击舰"}},
                            {"match": {"descriptions_zh": "堪培拉"}},
                            {"match": {"descriptions_zh": "两栖"}},
                        ]
                    }
                },
                "size": 50
            }
            
            try:
                response = es.search(index=index_name, body=query3)
                if response["hits"]["total"]["value"] > 0:
                    print(f"\n搜索包含'堪培拉'、'两栖'或'攻击舰'的实体:")
                    print(f"找到 {response['hits']['total']['value']} 个结果")
                    for i, hit in enumerate(response["hits"]["hits"][:20], 1):
                        source = hit["_source"]
                        link_match = "✓" if source.get('link', '') == link else " "
                        print(f"  {link_match} {i}. {source.get('label', 'N/A')}")
                        print(f"      Link: {source.get('link', 'N/A')[:80]}...")
                        if source.get('link', '') == link:
                            print(f"      ✓ 这是目标实体！")
                            print(f"      Aliases_zh: {source.get('aliases_zh', [])}")
                            print(f"      Descriptions_zh: {source.get('descriptions_zh', '')[:200]}...")
                            return source
            except Exception as e:
                print(f"  通过label搜索失败: {e}")
            
            # 方法4: 检查是否有类似的实体（使用"堪培拉级"）
            query4 = {
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"label": "堪培拉级"}},
                        ]
                    }
                },
                "size": 10
            }
            
            try:
                response = es.search(index=index_name, body=query4)
                if response["hits"]["total"]["value"] > 0:
                    print(f"\n搜索包含'堪培拉级'的实体:")
                    print(f"找到 {response['hits']['total']['value']} 个结果")
                    for i, hit in enumerate(response["hits"]["hits"], 1):
                        source = hit["_source"]
                        link_match = "✓" if source.get('link', '') == link else " "
                        print(f"  {link_match} {i}. {source.get('label', 'N/A')}")
                        print(f"      Link: {source.get('link', 'N/A')[:80]}...")
                        print(f"      Aliases_zh: {source.get('aliases_zh', [])}")
                        if source.get('link', '') == link:
                            print(f"      ✓ 这是目标实体！")
                            return source
            except Exception as e:
                print(f"  搜索'堪培拉级'失败: {e}")
            
        except Exception as e:
            print(f"索引 '{index_name}' 查询失败: {e}")
            continue
    
    print("\n❌ 未找到该实体")
    return None

def test_query(query_text):
    """测试查询，看看实际搜索到了什么"""
    print(f"\n{'='*80}")
    print(f"测试查询: '{query_text}'")
    print(f"{'='*80}")
    
    from search_withllm import hybrid_search
    
    try:
        results = hybrid_search(query_text, top_k=20)
        print(f"\n找到 {len(results)} 个结果:")
        for i, result in enumerate(results[:10], 1):
            print(f"  {i}. {result.get('label', 'N/A')}")
            print(f"     链接: {result.get('link', 'N/A')[:80]}...")
            print(f"     别名: {result.get('aliases_zh', [])}")
    except Exception as e:
        print(f"查询失败: {e}")

def main():
    # 检查目标实体
    target_link = "https://zh.wikipedia.org/wiki/%E5%A0%AA%E5%9F%B9%E6%8B%89%E7%BA%A7%E4%B8%A4%E6%A3%B2%E6%94%BB%E5%87%BB%E8%88%B0"
    
    print("="*80)
    print("检查ES中的实体数据")
    print("="*80)
    
    entity = check_entity_by_link(target_link)
    
    if entity:
        print(f"\n{'='*80}")
        print("实体数据摘要")
        print(f"{'='*80}")
        print(f"Label: {entity.get('label', 'N/A')}")
        print(f"Aliases_zh: {entity.get('aliases_zh', [])}")
        print(f"Link: {entity.get('link', 'N/A')}")
        print(f"\n检查该实体是否包含查询关键词:")
        query_keywords = ["堪培拉", "两栖", "攻击舰"]
        for keyword in query_keywords:
            label_contains = keyword in entity.get('label', '')
            aliases_contains = any(keyword in alias for alias in entity.get('aliases_zh', []))
            desc_contains = keyword in entity.get('descriptions_zh', '')
            print(f"  '{keyword}': label={label_contains}, aliases={aliases_contains}, desc={desc_contains}")
    else:
        print("\n❌ 该实体可能不在ES中，或者链接格式不匹配")
    
    # 测试实际查询
    test_query("澳大利亚海军印度太平洋努力2021特遣部队\n堪培拉号两栖编")

if __name__ == "__main__":
    main()

