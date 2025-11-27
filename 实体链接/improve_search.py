"""
改进搜索策略 - 针对失败案例的优化
"""
from search_withllm import hybrid_search, generate_vector, get_alias_and_definition
import re

def expand_short_query(query_text):
    """
    扩展短查询，尝试从查询文本中提取更多信息
    """
    # 移除常见的前缀/后缀
    query = query_text.strip()
    
    # 如果是纯字母数字组合（如"FARA", "Mk41"），尝试添加上下文
    if re.match(r'^[A-Za-z0-9\-]+$', query) and len(query) <= 10:
        # 尝试使用LLM获取更多信息
        try:
            response = get_alias_and_definition(query_text)
            # 提取定义中的关键词
            if "定义：" in response:
                definition = response.split("定义：")[1].strip()
                # 提取可能的实体类型关键词
                keywords = []
                for word in ["导弹", "战斗机", "直升机", "驱逐舰", "护卫舰", "坦克", "步枪", "手枪", "系统", "武器"]:
                    if word in definition:
                        keywords.append(word)
                if keywords:
                    return f"{query_text} {keywords[0]}"
        except:
            pass
    
    return query_text

def improved_hybrid_search(query_text, top_k=20, text_boost=1.0, vector_boost=0.8, use_vector=True):
    """
    改进的混合搜索，针对短查询和模糊查询进行优化
    """
    # 1. 查询扩展
    expanded_query = expand_short_query(query_text)
    
    # 2. 如果查询太短，使用更宽松的搜索策略
    if len(query_text.strip()) <= 5:
        # 使用通配符查询
        text_query = {
            "bool": {
                "should": [
                    {
                        "wildcard": {
                            "label": {
                                "value": f"*{query_text}*",
                                "boost": text_boost
                            }
                        }
                    },
                    {
                        "wildcard": {
                            "aliases_zh": {
                                "value": f"*{query_text}*",
                                "boost": text_boost
                            }
                        }
                    },
                    {
                        "match": {
                            "label": {
                                "query": query_text,
                                "boost": text_boost * 0.5,
                                "fuzziness": "AUTO"
                            }
                        }
                    },
                    {
                        "match": {
                            "aliases_zh": {
                                "query": query_text,
                                "boost": text_boost * 0.5,
                                "fuzziness": "AUTO"
                            }
                        }
                    }
                ]
            }
        }
    else:
        # 使用标准查询
        text_query = {
            "bool": {
                "should": [
                    {
                        "match": {
                            "label": {
                                "query": query_text,
                                "boost": text_boost
                            }
                        }
                    },
                    {
                        "match": {
                            "aliases_zh": {
                                "query": query_text,
                                "boost": text_boost
                            }
                        }
                    },
                    {
                        "match_phrase": {
                            "label": {
                                "query": query_text,
                                "boost": text_boost * 1.5
                            }
                        }
                    },
                    {
                        "match_phrase": {
                            "aliases_zh": {
                                "query": query_text,
                                "boost": text_boost * 1.5
                            }
                        }
                    }
                ]
            }
        }
    
    # 3. 向量检索部分（与原版相同）
    query_vector = None
    knn_query = None
    
    if use_vector:
        try:
            response_content = get_alias_and_definition(query_text)
            if "定义：" in response_content:
                input_definition = response_content.split("定义：")[1].strip()
                query_vector = generate_vector(input_definition, use_cache=True)
        except:
            try:
                query_vector = generate_vector(query_text, use_cache=True)
            except:
                query_vector = None
        
        if query_vector is not None:
            knn_query = {
                "field": "content_vector",
                "query_vector": query_vector,
                "k": 20,  # 增加候选数量
                "num_candidates": 50,  # 增加候选数量
                "boost": vector_boost
            }
    
    # 4. 构建查询
    if knn_query:
        search_query = {
            "query": text_query,
            "knn": knn_query,
            "size": top_k
        }
    else:
        search_query = {
            "query": text_query,
            "size": top_k
        }
    
    # 5. 执行搜索（与原版相同的索引查找逻辑）
    from 实体链接.es_client import es
    index_names = ["data2", "data1"]
    response = None
    
    for index_name in index_names:
        try:
            if es.indices.exists(index=index_name):
                if knn_query and knn_query.get("field") == "content_vector":
                    try:
                        response = es.search(index=index_name, body=search_query)
                    except Exception as e:
                        if "content_vector" in str(e).lower() or "field" in str(e).lower():
                            search_query["knn"]["field"] = "descriptions_zh_vector"
                            try:
                                response = es.search(index=index_name, body=search_query)
                            except Exception as e2:
                                search_query = {"query": text_query, "size": top_k}
                                response = es.search(index=index_name, body=search_query)
                        else:
                            raise e
                else:
                    response = es.search(index=index_name, body=search_query)
                break
        except Exception as e:
            continue
    
    if response is None:
        raise Exception(f"未找到可用的索引，尝试过的索引: {index_names}")
    
    hits = response["hits"]["hits"]
    results = [
        {
            "label": hit["_source"].get("label", ""),
            "aliases_zh": hit["_source"].get("aliases_zh", []),
            "aliases_en": hit["_source"].get("aliases_en", []),
            "descriptions_zh": hit["_source"].get("descriptions_zh", ""),
            "link": hit["_source"].get("link", "")
        } 
        for hit in hits
    ]
    return results

def test_improved_search():
    """测试改进的搜索"""
    test_queries = [
        "FARA",
        "Mk41",
        "LRASM",
        "Block1B",
        "PKS"
    ]
    
    print("="*80)
    print("测试改进的搜索策略")
    print("="*80)
    
    for query in test_queries:
        print(f"\n查询: '{query}'")
        try:
            results = improved_hybrid_search(query, top_k=10)
            print(f"  找到 {len(results)} 个结果")
            if results:
                print("  前3个结果:")
                for i, result in enumerate(results[:3], 1):
                    print(f"    {i}. {result.get('label', 'N/A')}")
                    print(f"       链接: {result.get('link', 'N/A')[:60]}...")
            else:
                print("  ❌ 仍然没有结果")
        except Exception as e:
            print(f"  ❌ 搜索失败: {e}")

if __name__ == "__main__":
    test_improved_search()

