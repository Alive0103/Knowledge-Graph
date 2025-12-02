import json
import torch
from transformers import BertTokenizer, BertModel
from elasticsearch import Elasticsearch
from es_client import es  # 导入您的ES客户端

# 配置
MODEL_PATH = './model/chinese-roberta-wwm-ext-large'
INDEX_NAME = "data2"
VECTOR_DIMS = 1024

# 初始化BERT模型
print("加载BERT模型...")
try:
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertModel.from_pretrained(MODEL_PATH)
    model.eval()
    print("✓ BERT模型加载成功")
except Exception as e:
    print(f"✗ BERT模型加载失败: {e}")
    exit(1)

def generate_vector(text):
    """生成文本向量"""
    if text and text.strip():
        try:
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            vector = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            return vector.tolist()
        except Exception as e:
            print(f"向量生成失败: {e}")
            return None
    return None

def test_index_status():
    """测试索引状态"""
    print("\n" + "="*60)
    print("1. 索引状态测试")
    print("="*60)
    
    try:
        # 检查索引是否存在
        if not es.indices.exists(index=INDEX_NAME):
            print("✗ 索引不存在")
            return False
        
        # 获取索引统计
        stats = es.indices.stats(index=INDEX_NAME)
        store_size = stats['indices'][INDEX_NAME]['total']['store']['size_in_bytes']
        print(f"✓ 索引存储大小: {store_size / 1024 / 1024:.2f} MB")
        
        # 获取文档数量
        count_result = es.count(index=INDEX_NAME)
        total_docs = count_result['count']
        print(f"✓ 文档总数: {total_docs}")
        
        # 检查映射
        mapping = es.indices.get_mapping(index=INDEX_NAME)
        vector_fields = []
        for field, props in mapping[INDEX_NAME]['mappings']['properties'].items():
            if props.get('type') == 'dense_vector':
                vector_fields.append(field)
        
        print(f"✓ 向量字段: {vector_fields}")
        
        return True
        
    except Exception as e:
        print(f"✗ 索引状态检查失败: {e}")
        return False

def test_vector_fields():
    """测试向量字段数据"""
    print("\n" + "="*60)
    print("2. 向量字段数据测试")
    print("="*60)
    
    try:
        # 检查包含向量的文档数量
        zh_vector_count = es.count(
            index=INDEX_NAME,
            body={"query": {"exists": {"field": "descriptions_zh_vector"}}}
        )['count']
        
        en_vector_count = es.count(
            index=INDEX_NAME,
            body={"query": {"exists": {"field": "descriptions_en_vector"}}}
        )['count']
        
        print(f"✓ 包含中文向量的文档: {zh_vector_count} 个")
        print(f"✓ 包含英文向量的文档: {en_vector_count} 个")
        
        # 抽样检查向量数据
        sample_query = {
            "size": 1,
            "query": {
                "bool": {
                    "should": [
                        {"exists": {"field": "descriptions_zh_vector"}},
                        {"exists": {"field": "descriptions_en_vector"}}
                    ]
                }
            },
            "_source": ["label", "descriptions_zh_vector", "descriptions_en_vector"]
        }
        
        sample_result = es.search(index=INDEX_NAME, body=sample_query)
        if sample_result['hits']['hits']:
            sample_doc = sample_result['hits']['hits'][0]['_source']
            print(f"✓ 样本文档标题: {sample_doc.get('label', 'N/A')}")
            
            if 'descriptions_zh_vector' in sample_doc:
                zh_vec = sample_doc['descriptions_zh_vector']
                print(f"✓ 中文向量维度: {len(zh_vec)}")
            
            if 'descriptions_en_vector' in sample_doc:
                en_vec = sample_doc['descriptions_en_vector']
                print(f"✓ 英文向量维度: {len(en_vec)}")
        
        return True
        
    except Exception as e:
        print(f"✗ 向量字段检查失败: {e}")
        return False

def test_knn_search():
    """测试KNN向量搜索"""
    print("\n" + "="*60)
    print("3. KNN向量搜索测试")
    print("="*60)
    
    test_queries = [
        "军事装备",
        "战斗机",
        "航空母舰",
        "坦克",
        "导弹"
    ]
    
    for i, query_text in enumerate(test_queries, 1):
        print(f"\n--- 测试查询 {i}: '{query_text}' ---")
        
        # 生成查询向量
        query_vector = generate_vector(query_text)
        if not query_vector or len(query_vector) != VECTOR_DIMS:
            print(f"✗ 查询向量生成失败")
            continue
        
        print(f"✓ 查询向量维度: {len(query_vector)}")
        
        try:
            # 方法1: 使用KNN查询（阿里云推荐）
            knn_query = {
                "knn": {
                    "field": "descriptions_zh_vector",
                    "query_vector": query_vector,
                    "k": 5,
                    "num_candidates": 50
                }
            }
            
            response = es.search(
                index=INDEX_NAME,
                body={
                    "size": 5,
                    "_source": ["label", "descriptions_zh", "link"],
                    "knn": knn_query["knn"]
                }
            )
            
            hits = response['hits']['hits']
            print(f"✓ KNN搜索成功! 找到 {len(hits)} 个相关文档")
            
            for j, hit in enumerate(hits, 1):
                score = hit['_score']
                source = hit['_source']
                label = source.get('label', 'N/A')
                desc = source.get('descriptions_zh', '')[:50] + "..." if len(source.get('descriptions_zh', '')) > 50 else source.get('descriptions_zh', '')
                
                print(f"  {j}. 相似度: {score:.4f}")
                print(f"     标题: {label}")
                print(f"     描述: {desc}")
                if source.get('link'):
                    print(f"     链接: {source['link']}")
                print()
                
        except Exception as e:
            print(f"✗ KNN搜索失败: {e}")
            
            # 备用测试：简单关键词搜索
            try:
                print("尝试备用关键词搜索...")
                backup_response = es.search(
                    index=INDEX_NAME,
                    body={
                        "size": 3,
                        "query": {
                            "match": {
                                "label": query_text
                            }
                        },
                        "_source": ["label", "descriptions_zh"]
                    }
                )
                
                backup_hits = backup_response['hits']['hits']
                print(f"✓ 关键词搜索找到 {len(backup_hits)} 个文档")
                for j, hit in enumerate(backup_hits, 1):
                    source = hit['_source']
                    print(f"  {j}. 标题: {source.get('label', 'N/A')}")
                    
            except Exception as e2:
                print(f"✗ 备用搜索也失败: {e2}")

def test_hybrid_search():
    """测试混合搜索（向量+关键词）"""
    print("\n" + "="*60)
    print("4. 混合搜索测试")
    print("="*60)
    
    query_text = "美国战斗机"
    query_vector = generate_vector(query_text)
    
    if not query_vector:
        print("✗ 无法生成查询向量")
        return
    
    try:
        # 混合查询：KNN + 关键词过滤
        hybrid_query = {
            "knn": {
                "field": "descriptions_zh_vector",
                "query_vector": query_vector,
                "k": 10,
                "num_candidates": 100,
                "filter": {
                    "match": {
                        "label": "美国"
                    }
                }
            }
        }
        
        response = es.search(
            index=INDEX_NAME,
            body={
                "size": 5,
                "_source": ["label", "descriptions_zh"],
                "knn": hybrid_query["knn"]
            }
        )
        
        hits = response['hits']['hits']
        print(f"✓ 混合搜索查询: '{query_text}'")
        print(f"✓ 找到 {len(hits)} 个相关文档")
        
        for i, hit in enumerate(hits, 1):
            source = hit['_source']
            print(f"  {i}. 相似度: {hit['_score']:.4f}, 标题: {source.get('label', 'N/A')}")
            
    except Exception as e:
        print(f"✗ 混合搜索失败: {e}")

def test_performance():
    """测试搜索性能"""
    print("\n" + "="*60)
    print("5. 性能测试")
    print("="*60)
    
    import time
    
    test_cases = [
        ("军事", "简单查询"),
        ("战斗机 航空母舰", "多关键词"),
        ("现代军事装备", "长文本")
    ]
    
    for query_text, test_name in test_cases:
        print(f"\n--- {test_name}: '{query_text}' ---")
        
        start_time = time.time()
        
        try:
            # 关键词搜索性能测试
            response = es.search(
                index=INDEX_NAME,
                body={
                    "size": 5,
                    "query": {
                        "match": {
                            "label": query_text
                        }
                    },
                    "_source": ["label"]
                }
            )
            
            search_time = time.time() - start_time
            hits_count = response['hits']['total']['value']
            
            print(f"✓ 关键词搜索: {hits_count} 个结果, 耗时: {search_time:.3f}秒")
            
        except Exception as e:
            print(f"✗ 关键词搜索失败: {e}")
        
        # 如果有向量，测试KNN性能
        query_vector = generate_vector(query_text)
        if query_vector:
            start_time = time.time()
            
            try:
                knn_response = es.search(
                    index=INDEX_NAME,
                    body={
                        "size": 5,
                        "knn": {
                            "field": "descriptions_zh_vector",
                            "query_vector": query_vector,
                            "k": 5,
                            "num_candidates": 50
                        },
                        "_source": ["label"]
                    }
                )
                
                knn_time = time.time() - start_time
                knn_hits = len(knn_response['hits']['hits'])
                
                print(f"✓ KNN搜索: {knn_hits} 个结果, 耗时: {knn_time:.3f}秒")
                
            except Exception as e:
                print(f"✗ KNN搜索失败: {e}")

def main():
    """主测试函数"""
    print("开始向量搜索功能测试")
    print("="*60)
    
    # 测试1: 索引状态
    if not test_index_status():
        print("\n✗ 索引状态测试失败，无法继续")
        return
    
    # 测试2: 向量字段数据
    if not test_vector_fields():
        print("\n✗ 向量字段测试失败，无法继续")
        return
    
    # 测试3: KNN搜索
    test_knn_search()
    
    # 测试4: 混合搜索
    test_hybrid_search()
    
    # 测试5: 性能测试
    test_performance()
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)
    
    # 最终统计
    try:
        final_stats = es.indices.stats(index=INDEX_NAME)
        final_count = es.count(index=INDEX_NAME)['count']
        
        zh_vec_count = es.count(
            index=INDEX_NAME,
            body={"query": {"exists": {"field": "descriptions_zh_vector"}}}
        )['count']
        
        print(f"最终统计:")
        print(f"- 文档总数: {final_count}")
        print(f"- 中文向量文档: {zh_vec_count}")
        print(f"- 索引大小: {final_stats['indices'][INDEX_NAME]['total']['store']['size_in_bytes'] / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"最终统计失败: {e}")

if __name__ == "__main__":
    main()