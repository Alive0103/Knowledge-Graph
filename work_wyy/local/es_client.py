"""Elasticsearch 客户端配置模块
使用配置文件统一管理ES连接
"""

from config import create_es_client, ES_INDEX_NAME

# 创建ES客户端实例
es = create_es_client()

# 导出索引名称
__all__ = ['es', 'ES_INDEX_NAME']

