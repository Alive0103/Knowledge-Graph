"""Elasticsearch 客户端配置模块
统一管理阿里云 ES 连接配置，解决版本兼容性问题

根据阿里云文档和实际测试，使用以下配置方式：
- 使用 headers 参数设置兼容性版本
- 使用 basic_auth 进行认证
- 启用 http_compress 压缩
"""
from elasticsearch import Elasticsearch

# 阿里云 Elasticsearch 配置
ES_URL = "http://kgcode-xw7.public.cn-hangzhou.es-serverless.aliyuncs.com:9200"
ES_USERNAME = "kgcode-xw7"
ES_PASSWORD = "Ln216812_"

def create_es_client():
    """
    创建 Elasticsearch 客户端，配置阿里云 ES 连接
    
    参考：https://help.aliyun.com/zh/es/developer-reference/use-a-client-to-access-an-alibaba-cloud-elasticsearch-cluster
    """
    es = Elasticsearch(
        ES_URL,
        basic_auth=(ES_USERNAME, ES_PASSWORD),
        headers={"accept": "application/vnd.elasticsearch+json;compatible-with=8"},
        http_compress=True
    )
    
    return es

# 创建默认的 ES 客户端实例
es = create_es_client()