"""
检查ES中向量数据的完整性
统计有多少文档包含向量字段
并测试向量搜索和文本搜索功能
"""
from es_client import es
import json
import torch
from transformers import BertTokenizer, BertModel

# 初始化BERT模型（用于生成查询向量）
model_name = './model/chinese-roberta-wwm-ext-large'
try:
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()
    print("✓ BERT模型加载成功")
except Exception as e:
    print(f"✗ BERT模型加载失败: {e}")
    tokenizer = None
    model = None

def generate_vector(text):
    """生成文本向量（与search_withllm.py保持一致）"""
    if text and text.strip() and tokenizer and model:
        try:
            import numpy as np
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            vector = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            
            # L2归一化（对余弦相似度很重要）
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            
            # 确保维度是1024（虽然large模型已经是1024维，但保持一致性）
            vector_dim = len(vector)
            target_dim = 1024
            if vector_dim != target_dim:
                if vector_dim < target_dim:
                    vector = np.pad(vector, (0, target_dim - vector_dim), 'constant', constant_values=0)
                    # 重新归一化
                    norm = np.linalg.norm(vector)
                    if norm > 0:
                        vector = vector / norm
                else:
                    vector = vector[:target_dim]
                    # 重新归一化
                    norm = np.linalg.norm(vector)
                    if norm > 0:
                        vector = vector / norm
            
            return vector.tolist()
        except Exception as e:
            print(f"向量生成失败: {e}")
            return None
    return None

def test_vector_search(index_name, query_text, top_k=5):
    """测试向量搜索功能"""
    print(f"\n🔍 测试向量搜索: '{query_text}'")
    print("-" * 50)

    # 生成查询向量
    query_vector = generate_vector(query_text)
    if not query_vector:
        print("❌ 无法生成查询向量")
        return

    print(f"查询向量维度: {len(query_vector)}")

    try:
        # 使用KNN搜索
        knn_query = {
            "knn": {
                "field": "descriptions_zh_vector",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": 50
            }
        }

        response = es.search(
            index=index_name,
            body={
                "size": top_k,
                "_source": ["label", "descriptions_zh", "link"],
                "knn": knn_query["knn"]
            }
        )

        hits = response['hits']['hits']
        total_hits = response['hits']['total']['value']

        print(f"✓ KNN搜索成功! 找到 {total_hits} 个相关文档")

        for i, hit in enumerate(hits, 1):
            score = hit['_score']
            source = hit['_source']
            label = source.get('label', 'N/A')
            desc = source.get('descriptions_zh', '')[:100] + "..." if len(source.get('descriptions_zh', '')) > 100 else source.get('descriptions_zh', '')
            link = source.get('link', '')

            print(f"\n{i}. 相似度: {score:.4f}")
            print(f"   标题: {label}")
            print(f"   描述: {desc}")
            if link:
                print(f"   链接: {link}")

    except Exception as e:
        print(f"❌ 向量搜索失败: {e}")

        # 备用方案：检查是否支持脚本查询
        try:
            print("尝试备用搜索方法...")
            # 简单的匹配查询作为备用
            backup_response = es.search(
                index=index_name,
                body={
                    "size": top_k,
                    "query": {
                        "match": {
                            "descriptions_zh": query_text
                        }
                    },
                    "_source": ["label", "descriptions_zh"]
                }
            )

            backup_hits = backup_response['hits']['hits']
            print(f"备用搜索找到 {len(backup_hits)} 个文档")
            for i, hit in enumerate(backup_hits, 1):
                source = hit['_source']
                print(f"  {i}. 标题: {source.get('label', 'N/A')}")

        except Exception as e2:
            print(f"备用搜索也失败: {e2}")

def test_text_search(index_name, query_text, top_k=5):
    """测试文本搜索功能"""
    print(f"\n🔍 测试文本搜索: '{query_text}'")
    print("-" * 50)

    try:
        # 多字段文本搜索
        text_query = {
            "multi_match": {
                "query": query_text,
                "fields": [
                    "label^3",           # 标题字段权重更高
                    "descriptions_zh^2", # 描述字段中等权重
                    "aliases_zh^2",      # 别名字段中等权重
                    "content"           # 内容字段默认权重
                ],
                "type": "best_fields"
            }
        }

        response = es.search(
            index=index_name,
            body={
                "size": top_k,
                "query": text_query,
                "_source": ["label", "descriptions_zh", "score"],
                "highlight": {
                    "fields": {
                        "descriptions_zh": {},
                        "label": {}
                    }
                }
            }
        )

        hits = response['hits']['hits']
        total_hits = response['hits']['total']['value']

        print(f"✓ 文本搜索成功! 找到 {total_hits} 个相关文档")

        for i, hit in enumerate(hits, 1):
            score = hit['_score']
            source = hit['_source']
            label = source.get('label', 'N/A')
            desc = source.get('descriptions_zh', '')[:100] + "..." if len(source.get('descriptions_zh', '')) > 100 else source.get('descriptions_zh', '')

            print(f"\n{i}. 相关度: {score:.4f}")
            print(f"   标题: {label}")
            print(f"   描述: {desc}")

            # 显示高亮结果
            if 'highlight' in hit:
                highlights = hit['highlight']
                for field, highlights_list in highlights.items():
                    for hl in highlights_list[:2]:  # 显示前2个高亮片段
                        print(f"   高亮({field}): {hl}")

    except Exception as e:
        print(f"❌ 文本搜索失败: {e}")

def test_hybrid_search(index_name, query_text, top_k=5):
    """测试混合搜索（向量+文本）"""
    print(f"\n🔍 测试混合搜索: '{query_text}'")
    print("-" * 50)

    query_vector = generate_vector(query_text)
    if not query_vector:
        print("❌ 无法生成查询向量，跳过混合搜索")
        return

    try:
        # 混合搜索：KNN + 文本过滤
        hybrid_query = {
            "knn": {
                "field": "descriptions_zh_vector",
                "query_vector": query_vector,
                "k": top_k * 2,  # 获取更多候选
                "num_candidates": 100,
                "filter": {
                    "match": {
                        "descriptions_zh": query_text
                    }
                }
            }
        }

        response = es.search(
            index=index_name,
            body={
                "size": top_k,
                "_source": ["label", "descriptions_zh"],
                "knn": hybrid_query["knn"]
            }
        )

        hits = response['hits']['hits']
        print(f"✓ 混合搜索成功! 找到 {len(hits)} 个相关文档")

        for i, hit in enumerate(hits, 1):
            score = hit['_score']
            source = hit['_source']
            print(f"  {i}. 相似度: {score:.4f}, 标题: {source.get('label', 'N/A')}")

    except Exception as e:
        print(f"❌ 混合搜索失败: {e}")

def check_vector_data():
    """检查ES中向量数据的完整性"""
    index_name = "data2"

    print("=" * 60)
    print("检查ES中向量数据的完整性")
    print("=" * 60)

    # 检查索引是否存在
    if not es.indices.exists(index=index_name):
        print(f"❌ 索引 {index_name} 不存在")
        return

    # 获取总文档数
    total_count = es.count(index=index_name)["count"]
    print(f"\n索引 '{index_name}' 总文档数: {total_count}")

    # 统计有向量的文档数
    print("\n正在统计向量字段数据...")

    vector_fields = ["descriptions_zh_vector", "descriptions_en_vector", "content_vector"]
    field_stats = {}

    for field in vector_fields:
        try:
            query = {
                "query": {
                    "exists": {"field": field}
                },
                "size": 0
            }
            result = es.search(index=index_name, body=query)
            count = result["hits"]["total"]["value"]
            field_stats[field] = count
            print(f"✓ 包含 {field} 的文档数: {count}")
        except Exception as e:
            print(f"❌ 查询 {field} 失败: {e}")
            field_stats[field] = 0

    # 统计结果
    print("\n" + "=" * 60)
    print("统计结果:")
    print("=" * 60)
    print(f"总文档数: {total_count}")
    for field, count in field_stats.items():
        percentage = (count / total_count * 100) if total_count > 0 else 0
        print(f"有 {field} 的文档: {count} ({percentage:.2f}%)")

    # 检查几个样本文档
    print("\n" + "=" * 60)
    print("样本文档检查:")
    print("=" * 60)

    sample_query = {
        "query": {
            "exists": {"field": "descriptions_zh_vector"}
        },
        "size": 3
    }

    try:
        sample_result = es.search(index=index_name, body=sample_query)
        hits = sample_result['hits']['hits']

        for i, hit in enumerate(hits, 1):
            source = hit['_source']
            label = source.get('label', 'N/A')

            has_zh_vec = "descriptions_zh_vector" in source and source["descriptions_zh_vector"]
            vec_dims = len(source["descriptions_zh_vector"]) if has_zh_vec and isinstance(source["descriptions_zh_vector"], list) else 0

            print(f"\n样本文档 {i}: {label}")
            print(f"  descriptions_zh_vector: {'✓' if has_zh_vec else '❌'} (维度: {vec_dims})")

    except Exception as e:
        print(f"❌ 获取样本文档失败: {e}")

    # 测试搜索功能
    if field_stats["descriptions_zh_vector"] > 0:
        print("\n" + "=" * 60)
        print("搜索功能测试:")
        print("=" * 60)

        test_queries = [
            "军事装备",
            "战斗机",
            "航空母舰"
        ]

        for query in test_queries:
            # 测试向量搜索
            test_vector_search(index_name, query)

            # 测试文本搜索
            test_text_search(index_name, query)

            # 测试混合搜索
            test_hybrid_search(index_name, query)

            print("\n" + "="*50)

    # 诊断建议
    print("\n" + "=" * 60)
    print("诊断建议:")
    print("=" * 60)

    if all(count == 0 for count in field_stats.values()):
        print("❌ 问题确认: ES中没有任何向量数据！")
        print("\n解决方案:")
        print("1. 运行向量生成和导入脚本")
        print("2. 检查向量生成代码是否正确")
    elif field_stats["descriptions_zh_vector"] < total_count * 0.5:
        print("⚠ 警告: 大部分文档缺少向量数据")
        print("建议重新运行向量生成脚本补充向量数据")
    else:
        print("✓ 向量数据完整性良好")
        print("✓ 搜索功能测试完成")

if __name__ == "__main__":
    check_vector_data()