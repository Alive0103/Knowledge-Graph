# -*- coding: utf-8 -*-
"""
阿里云OpenSearch文本向量化API测试脚本

参照文档：https://help.aliyun.com/zh/open-search/search-platform/developer-reference/text-embedding-api-details

功能：
- 测试文本向量化API
- 输出1024维向量
- 支持批量向量化（最多32条）
"""

import requests
import json
from typing import List, Dict, Optional

# ==================== 配置信息 ====================
# 请根据实际情况修改以下配置

# API-KEY（必需）
# 获取方式：https://help.aliyun.com/zh/open-search/search-platform/developer-reference/get-api-key
API_KEY = "YOUR_API_KEY_HERE"  # 请替换为您的API-KEY

# 服务调用地址（必需）
# 支持公网和VPC两种方式，详情参见：获取服务接入地址
# 示例格式：http://****-hangzhou.opensearch.aliyuncs.com
HOST = "YOUR_HOST_HERE"  # 请替换为您的服务地址

# 工作空间名称（默认：default）
WORKSPACE_NAME = "default"

# 服务ID（选择1024维的服务）
# 可选服务：
# - ops-text-embedding-002: 1024维，支持多语言（100+），最大长度8192
# - ops-qwen3-embedding-0.6b: 1024维，Qwen3系列，最大长度32k
SERVICE_ID = "ops-text-embedding-002"  # 默认使用002服务（1024维）

# ==================== API调用函数 ====================

def get_embedding(
    texts: List[str],
    input_type: str = "document",
    api_key: Optional[str] = None,
    host: Optional[str] = None,
    workspace_name: Optional[str] = None,
    service_id: Optional[str] = None
) -> Dict:
    """
    调用文本向量化API
    
    Args:
        texts: 输入文本列表（最多32条，每条长度取决于选择的模型）
        input_type: 输入类型，可选值："query" 或 "document"（默认）
        api_key: API密钥（如果为None，使用全局配置）
        host: 服务地址（如果为None，使用全局配置）
        workspace_name: 工作空间名称（如果为None，使用全局配置）
        service_id: 服务ID（如果为None，使用全局配置）
    
    Returns:
        API响应结果字典，包含：
        - request_id: 请求ID
        - latency: 请求耗时（ms）
        - usage: 使用信息（token_count）
        - result: 结果（embeddings列表）
    
    Raises:
        ValueError: 配置错误或参数错误
        requests.RequestException: API请求失败
    """
    # 使用传入参数或全局配置
    api_key = api_key or API_KEY
    host = host or HOST
    workspace_name = workspace_name or WORKSPACE_NAME
    service_id = service_id or SERVICE_ID
    
    # 参数验证
    if api_key == "YOUR_API_KEY_HERE" or not api_key:
        raise ValueError("请设置API_KEY配置")
    if host == "YOUR_HOST_HERE" or not host:
        raise ValueError("请设置HOST配置")
    if not texts:
        raise ValueError("输入文本列表不能为空")
    if len(texts) > 32:
        raise ValueError("每次请求最多支持32条文本")
    
    # 移除空字符串
    texts = [text.strip() for text in texts if text.strip()]
    if not texts:
        raise ValueError("输入文本列表不能包含空字符串")
    
    # 构建URL
    url = f"{host}/v3/openapi/workspaces/{workspace_name}/text-embedding/{service_id}"
    
    # 构建请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # 构建请求体
    body = {
        "input": texts,
        "input_type": input_type
    }
    
    # 发送请求
    try:
        response = requests.post(url, headers=headers, json=body, timeout=30)
        response.raise_for_status()  # 如果状态码不是200，抛出异常
        
        result = response.json()
        
        # 检查是否有错误
        if "code" in result and "message" in result:
            raise ValueError(f"API返回错误: {result.get('code')} - {result.get('message')}")
        
        return result
        
    except requests.exceptions.RequestException as e:
        raise requests.RequestException(f"API请求失败: {str(e)}")


def print_embedding_result(result: Dict, show_full_vector: bool = False):
    """
    打印向量化结果
    
    Args:
        result: API返回的结果字典
        show_full_vector: 是否显示完整向量（默认False，只显示前5个和后5个维度）
    """
    print("=" * 80)
    print("文本向量化结果")
    print("=" * 80)
    
    # 基本信息
    print(f"请求ID: {result.get('request_id', 'N/A')}")
    print(f"请求耗时: {result.get('latency', 'N/A')} ms")
    
    # 使用信息
    if "usage" in result:
        usage = result["usage"]
        print(f"Token数量: {usage.get('token_count', 'N/A')}")
    
    # 向量结果
    if "result" in result and "embeddings" in result["result"]:
        embeddings = result["result"]["embeddings"]
        print(f"\n向量数量: {len(embeddings)}")
        print("-" * 80)
        
        for emb in embeddings:
            index = emb.get("index", "N/A")
            vector = emb.get("embedding", [])
            dim = len(vector)
            
            print(f"\n文本索引: {index}")
            print(f"向量维度: {dim}")
            
            if show_full_vector:
                print(f"完整向量: {vector}")
            else:
                # 只显示前5个和后5个维度
                if dim > 10:
                    preview = vector[:5] + ["..."] + vector[-5:]
                    print(f"向量预览: {preview}")
                else:
                    print(f"完整向量: {vector}")
    
    print("=" * 80)


# ==================== 测试函数 ====================

def test_single_text():
    """测试单个文本向量化"""
    print("\n" + "=" * 80)
    print("测试1: 单个文本向量化")
    print("=" * 80)
    
    test_text = "科学技术是第一生产力"
    
    try:
        result = get_embedding([test_text], input_type="document")
        print(f"\n输入文本: {test_text}")
        print_embedding_result(result, show_full_vector=False)
        
        # 验证维度
        if "result" in result and "embeddings" in result["result"]:
            embeddings = result["result"]["embeddings"]
            if embeddings:
                dim = len(embeddings[0].get("embedding", []))
                if dim == 1024:
                    print("\n✓ 向量维度正确：1024维")
                else:
                    print(f"\n⚠ 向量维度为 {dim}，期望1024维")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")


def test_multiple_texts():
    """测试多个文本向量化"""
    print("\n" + "=" * 80)
    print("测试2: 多个文本向量化（批量）")
    print("=" * 80)
    
    test_texts = [
        "科学技术是第一生产力",
        "opensearch产品文档",
        "人工智能是未来发展的方向",
        "机器学习算法优化",
        "深度学习神经网络"
    ]
    
    try:
        result = get_embedding(test_texts, input_type="document")
        print(f"\n输入文本数量: {len(test_texts)}")
        for i, text in enumerate(test_texts):
            print(f"  {i+1}. {text}")
        
        print_embedding_result(result, show_full_vector=False)
        
        # 验证所有向量维度
        if "result" in result and "embeddings" in result["result"]:
            embeddings = result["result"]["embeddings"]
            all_1024 = all(len(emb.get("embedding", [])) == 1024 for emb in embeddings)
            if all_1024:
                print("\n✓ 所有向量维度正确：1024维")
            else:
                dims = [len(emb.get("embedding", [])) for emb in embeddings]
                print(f"\n⚠ 向量维度不一致: {dims}")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")


def test_query_type():
    """测试query类型的向量化"""
    print("\n" + "=" * 80)
    print("测试3: query类型向量化")
    print("=" * 80)
    
    test_text = "F-22战斗机"
    
    try:
        result = get_embedding([test_text], input_type="query")
        print(f"\n输入文本: {test_text}")
        print(f"输入类型: query")
        print_embedding_result(result, show_full_vector=False)
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")


def test_different_services():
    """测试不同的服务（如果可用）"""
    print("\n" + "=" * 80)
    print("测试4: 测试不同服务（Qwen3模型）")
    print("=" * 80)
    
    test_text = "人工智能技术发展"
    
    # 测试Qwen3服务（如果可用）
    try:
        result = get_embedding(
            [test_text],
            input_type="document",
            service_id="ops-qwen3-embedding-0.6b"
        )
        print(f"\n输入文本: {test_text}")
        print(f"使用服务: ops-qwen3-embedding-0.6b")
        print_embedding_result(result, show_full_vector=False)
        
    except Exception as e:
        print(f"⚠ Qwen3服务测试失败（可能未开通）: {e}")


def main():
    """主函数：运行所有测试"""
    print("\n" + "=" * 80)
    print("阿里云OpenSearch文本向量化API测试")
    print("=" * 80)
    print(f"服务ID: {SERVICE_ID}")
    print(f"工作空间: {WORKSPACE_NAME}")
    print(f"目标维度: 1024")
    
    # 检查配置
    if API_KEY == "YOUR_API_KEY_HERE" or HOST == "YOUR_HOST_HERE":
        print("\n⚠ 警告: 请先配置API_KEY和HOST")
        print("请在脚本中修改以下配置：")
        print("  - API_KEY: 您的API密钥")
        print("  - HOST: 您的服务地址")
        print("\n配置示例：")
        print("  API_KEY = 'OS-d1**2a'")
        print("  HOST = 'http://****-hangzhou.opensearch.aliyuncs.com'")
        return
    
    # 运行测试
    try:
        test_single_text()
        test_multiple_texts()
        test_query_type()
        test_different_services()
        
        print("\n" + "=" * 80)
        print("所有测试完成")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

