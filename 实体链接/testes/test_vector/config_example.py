# -*- coding: utf-8 -*-
"""
配置示例文件

请复制此文件为 config.py 并填入您的实际配置信息
或者直接在 test_opensearch_embedding.py 中修改配置
"""

# API-KEY（必需）
# 获取方式：https://help.aliyun.com/zh/open-search/search-platform/developer-reference/get-api-key
API_KEY = "OS-d1**2a"  # 请替换为您的实际API-KEY

# 服务调用地址（必需）
# 支持公网和VPC两种方式
# 示例格式：http://****-hangzhou.opensearch.aliyuncs.com
HOST = "http://****-hangzhou.opensearch.aliyuncs.com"  # 请替换为您的实际服务地址

# 工作空间名称（默认：default）
WORKSPACE_NAME = "default"

# 服务ID（选择1024维的服务）
# 可选服务：
# - ops-text-embedding-002: 1024维，支持多语言（100+），最大长度8192（推荐）
# - ops-qwen3-embedding-0.6b: 1024维，Qwen3系列，最大长度32k
SERVICE_ID = "ops-text-embedding-002"

