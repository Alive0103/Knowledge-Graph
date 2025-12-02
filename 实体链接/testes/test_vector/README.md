# 阿里云OpenSearch文本向量化API测试

## 功能说明

本目录包含测试脚本，用于测试阿里云OpenSearch的文本向量化API，输出1024维向量。

## 文件说明

- `test_opensearch_embedding.py`: 主测试脚本（阿里云OpenSearch API测试）
- `vector_test.py`: 本地模型向量生成测试
- `README.md`: 本说明文档

## 使用前准备

### 1. 获取API-KEY

参考文档：https://help.aliyun.com/zh/open-search/search-platform/developer-reference/get-api-key

### 2. 获取服务调用地址

支持公网和VPC两种方式，详情参见：获取服务接入地址

### 3. 配置脚本

编辑 `test_opensearch_embedding.py`，修改以下配置：

```python
# API-KEY（必需）
API_KEY = "YOUR_API_KEY_HERE"  # 替换为您的API-KEY

# 服务调用地址（必需）
HOST = "YOUR_HOST_HERE"  # 替换为您的服务地址，例如：http://****-hangzhou.opensearch.aliyuncs.com

# 工作空间名称（默认：default）
WORKSPACE_NAME = "default"

# 服务ID（1024维服务）
SERVICE_ID = "ops-text-embedding-002"  # 或使用 "ops-qwen3-embedding-0.6b"
```

## 支持的1024维服务

根据文档，以下服务支持1024维向量输出：

| 服务名称 | 服务ID | 说明 |
|---------|--------|------|
| OpenSearch通用文本向量服务-002 | ops-text-embedding-002 | 支持多语言（100+），最大长度8192 |
| Qwen3文本向量-0.6B | ops-qwen3-embedding-0.6b | Qwen3系列，最大长度32k |

## 运行测试

### 测试阿里云OpenSearch API

```bash
cd 实体链接/testes/test_vector
python test_opensearch_embedding.py
```

### 测试本地模型

```bash
cd 实体链接/testes/test_vector
python vector_test.py
```

## 测试内容

### test_opensearch_embedding.py 包含以下测试：

1. **单个文本向量化**：测试单个文本的向量化
2. **多个文本向量化**：测试批量向量化（最多32条）
3. **query类型向量化**：测试query类型的输入
4. **不同服务测试**：测试Qwen3服务（如果可用）

### vector_test.py 测试内容：

- 测试本地BERT模型生成1024维向量
- 验证向量归一化
- 验证向量维度

## API参数说明

### 请求参数

- `input`: 输入文本列表（Array/String），最多32条
- `input_type`: 输入类型（String），可选值：
  - `"query"`: 查询类型
  - `"document"`: 文档类型（默认）

### 返回参数

- `request_id`: 请求ID
- `latency`: 请求耗时（ms）
- `usage.token_count`: Token数量
- `result.embeddings`: 向量结果列表
  - `index`: 文本在input中的序号
  - `embedding`: 向量化结果（1024维浮点数列表）

## 注意事项

1. **API限制**：
   - 每次请求最多32条文本
   - 请求body最大不能超过8MB
   - 文本长度取决于选择的模型

2. **QPS限制**：
   - 不同服务有不同的QPS限制
   - 如需扩充QPS，请通过工单联系技术支持

3. **错误处理**：
   - 如果API返回错误，会显示错误码和错误信息
   - 常见错误：
     - `InvalidParameter`: 参数错误
     - `Unauthorized`: 认证失败
     - `ServiceUnavailable`: 服务不可用

## 参考文档

- [文本向量API文档](https://help.aliyun.com/zh/open-search/search-platform/developer-reference/text-embedding-api-details)
- [获取API-KEY](https://help.aliyun.com/zh/open-search/search-platform/developer-reference/get-api-key)
- [获取服务接入地址](https://help.aliyun.com/zh/open-search/search-platform/developer-reference/get-service-access-address)

