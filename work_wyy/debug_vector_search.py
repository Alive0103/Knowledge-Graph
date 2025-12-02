#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
调试向量搜索功能的脚本
"""

import json
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from es_client import es

# 模型路径（与向量入库脚本保持一致）
MODEL_PATH = './model/chinese-roberta-wwm-ext-large'
VECTOR_DIMS = 1024

# 初始化BERT模型（支持GPU）
print("加载BERT模型用于调试向量搜索...")
try:
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertModel.from_pretrained(MODEL_PATH)
    model.eval()

    # GPU / CPU 选择
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"✅ 检测到 GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = torch.device('cpu')
        print("⚠️ 未检测到 GPU，使用 CPU 生成测试向量（只跑几条，无所谓性能）")

    model = model.to(device)
    print("✓ BERT模型加载成功，用于调试向量搜索\n")
except Exception as e:
    print(f"✗ BERT模型加载失败: {e}")
    exit(1)

def generate_vector(text):
    """生成 1024 维文本向量（与索引一致）"""
    if text and text.strip():
        try:
            inputs = tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
            vector = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

            # 确保是一维
            if len(vector.shape) == 0:
                vector = vector.reshape(1)
            elif len(vector.shape) > 1:
                vector = vector.flatten()

            # 维度检查（large 模型应为 1024）
            dim = len(vector)
            if dim != VECTOR_DIMS:
                print(f"⚠️ 向量维度异常，期望 {VECTOR_DIMS}，实际 {dim}，将自动截断/补零")
                if dim < VECTOR_DIMS:
                    vector = np.pad(vector, (0, VECTOR_DIMS - dim), 'constant', constant_values=0)
                else:
                    vector = vector[:VECTOR_DIMS]
            
            # L2 归一化
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
                
            return vector.astype(float).tolist()
        except Exception as e:
            print(f"向量生成失败: {e}")
            return None
    return None

def debug_vector_search(query_text, index_name="data2"):
    """调试向量搜索"""
    print(f"调试查询: '{query_text}'")
    
    # 1. 检查索引是否存在
    print("\n1. 检查索引状态:")
    if not es.indices.exists(index=index_name):
        print(f"✗ 索引 '{index_name}' 不存在")
        return
    print(f"✓ 索引 '{index_name}' 存在")
    
    # 2. 生成查询向量
    print("\n2. 生成查询向量:")
    query_vector = generate_vector(query_text)
    if not query_vector:
        print("✗ 查询向量生成失败")
        return
    
    print(f"✓ 查询向量维度: {len(query_vector)}")
    
    # 3. 检查向量字段（mapping）
    print("\n3. 检查向量字段:")
    try:
        mapping = es.indices.get_mapping(index=index_name)
        vector_fields = []
        for field, props in mapping[index_name]['mappings']['properties'].items():
            if props.get('type') == 'dense_vector':
                vector_fields.append(field)
        print(f"✓ 在映射中检测到向量字段: {vector_fields}")
        if "descriptions_zh_vector" in vector_fields or "descriptions_en_vector" in vector_fields:
            print("  优先使用: descriptions_zh_vector / descriptions_en_vector 进行调试")
    except Exception as e:
        print(f"✗ 获取映射失败: {e}")
        return
    
    # 4. 执行KNN搜索
    print("\n4. 执行KNN搜索:")
    if not vector_fields:
        print("✗ 没有找到向量字段")
        return
    
    any_success = False
    for field in vector_fields:
        print(f"\n尝试字段: {field}")
        try:
            knn_query = {
                "knn": {
                    "field": field,
                    "query_vector": query_vector,
                    "k": 5,
                    "num_candidates": 10
                }
            }
            
            response = es.search(
                index=index_name,
                body={
                    "size": 5,
                    "_source": ["label", "descriptions_zh", "link"],
                    "knn": knn_query["knn"]
                }
            )
            
            hits = response['hits']['hits']
            print(f"✓ 使用字段 '{field}' 找到 {len(hits)} 个结果")
            if not hits:
                print("  提示: 结果为空，可能原因：")
                print("    - 该字段虽然是 dense_vector，但未开启向量索引")
                print("    - 或阿里云当前索引的向量检索配置有问题（可在控制台检查）")
            
            for i, hit in enumerate(hits, 1):
                score = hit['_score']
                source = hit['_source']
                label = source.get('label', 'N/A')
                desc = source.get('descriptions_zh', '')[:100] + "..." if len(source.get('descriptions_zh', '')) > 100 else source.get('descriptions_zh', '')
                
                print(f"  {i}. 相似度: {score:.4f}")
                print(f"     标题: {label}")
                print(f"     描述: {desc}")

            if hits:
                any_success = True
                
        except Exception as e:
            print(f"✗ 使用字段 '{field}' 搜索失败: {e}")

    print("\n5. 结果小结:")
    if any_success:
        print("✅ 向量检索接口调用成功，说明：")
        print("   - 查询向量生成正常")
        print("   - 索引中的向量字段可用于 KNN 搜索")
    else:
        print("⚠️ 没有任何字段返回结果，说明：")
        print("   - 要么索引里还没有向量（或向量字段未建索引）")
        print("   - 要么当前阿里云集群不支持对应 KNN 语法，需要在控制台检查：")
        print("     * 索引模板 / 向量检索是否开启")
        print("     * 向量字段是否配置为可检索")

def main():
    """主函数"""
    print("开始调试向量搜索功能")
    print("="*60)
    
    test_queries = [
        "AK47",
        "F-16战斗机", 
        "步枪",
        "坦克"
    ]
    
    for query in test_queries:
        debug_vector_search(query)
        print("\n" + "-"*60 + "\n")

if __name__ == "__main__":
    main()