#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地 Elasticsearch 连接测试脚本

网络架构说明：
- ES和Jupyter都在Docker容器中，通过Docker网络连接
- ES服务名为 "elasticsearch"，运行在9200端口
- 在Docker容器内，通过服务名 elasticsearch:9200 直接访问ES
"""

from elasticsearch import Elasticsearch
import sys
from datetime import datetime

def test_es_connection(es_url, description):
    """
    测试ES连接
    
    Args:
        es_url: ES连接URL
        description: 连接描述信息
    """
    print(f"\n{'='*60}")
    print(f"测试连接: {description}")
    print(f"ES URL: {es_url}")
    print(f"{'='*60}")
    
    try:
        # 创建ES客户端（本地ES通常不需要认证）
        es = Elasticsearch(
            es_url,
            request_timeout=10,  # 10秒超时
            max_retries=3,
            retry_on_timeout=True
        )
        
        # 测试1: 检查ES是否可达
        print("\n[测试1] 检查ES服务是否可达...")
        if not es.ping():
            print("❌ ES服务不可达")
            return False
        print("✅ ES服务可达")
        
        # 测试2: 获取ES集群信息
        print("\n[测试2] 获取ES集群信息...")
        info = es.info()
        print(f"✅ ES版本: {info.get('version', {}).get('number', 'Unknown')}")
        print(f"✅ 集群名称: {info.get('cluster_name', 'Unknown')}")
        print(f"✅ 节点名称: {info.get('name', 'Unknown')}")
        
        # 测试3: 获取集群健康状态
        print("\n[测试3] 检查集群健康状态...")
        health = es.cluster.health()
        print(f"✅ 集群状态: {health.get('status', 'Unknown')}")
        print(f"✅ 节点数: {health.get('number_of_nodes', 'Unknown')}")
        print(f"✅ 数据节点数: {health.get('number_of_data_nodes', 'Unknown')}")
        
        # 测试4: 列出所有索引（排除系统索引）
        print("\n[测试4] 列出所有索引...")
        try:
            # 使用 get_alias 获取所有索引，然后过滤系统索引
            all_indices = es.indices.get_alias(index="*")
            # 过滤掉系统索引（以.开头的索引）
            user_indices = {k: v for k, v in all_indices.items() if not k.startswith('.')}
            system_indices = {k: v for k, v in all_indices.items() if k.startswith('.')}
            
            if all_indices:
                print(f"✅ 找到 {len(all_indices)} 个索引（{len(user_indices)} 个用户索引，{len(system_indices)} 个系统索引）")
                if user_indices:
                    print("   用户索引:")
                    for index_name in sorted(user_indices.keys()):
                        print(f"   - {index_name}")
            else:
                print("⚠️  当前没有索引")
        except Exception as e:
            print(f"⚠️  获取索引列表失败: {str(e)}")
            user_indices = {}
            all_indices = {}
        
        # 测试5: 测试简单搜索（使用用户索引，避免系统索引）
        if user_indices:
            print("\n[测试5] 测试简单搜索...")
            first_index = list(user_indices.keys())[0]
            try:
                # 使用新的API格式，避免deprecation警告
                result = es.search(
                    index=first_index,
                    query={"match_all": {}},
                    size=1
                )
                print(f"✅ 在索引 '{first_index}' 中搜索成功")
                print(f"   总文档数: {result.get('hits', {}).get('total', {}).get('value', 0)}")
            except Exception as e:
                print(f"⚠️  搜索测试失败: {str(e)}")
        elif all_indices:
            print("\n[测试5] 跳过搜索测试（只有系统索引，无用户索引）")
        
        print(f"\n{'='*60}")
        print(f"✅ 连接测试成功: {description}")
        print(f"{'='*60}\n")
        return True
        
    except Exception as e:
        print(f"\n❌ 连接失败: {str(e)}")
        print(f"   错误类型: {type(e).__name__}")
        print(f"{'='*60}\n")
        return False

def main():
    """主函数"""
    print("\n" + "="*60)
    print("本地 Elasticsearch 连接测试")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 测试配置：通过Docker服务名访问ES
    # 架构：ES和Jupyter在同一Docker网络中，通过服务名访问
    test_configs = [
        {
            "url": "http://elasticsearch:9200",
            "description": "通过Docker服务名访问ES (elasticsearch:9200)"
        }
    ]
    
    # 测试结果
    results = []
    
    # 依次测试每个配置
    for config in test_configs:
        success = test_es_connection(config["url"], config["description"])
        results.append({
            "url": config["url"],
            "description": config["description"],
            "success": success
        })
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    for result in results:
        status = "✅ 成功" if result["success"] else "❌ 失败"
        print(f"{status} - {result['description']}")
        print(f"        URL: {result['url']}")
    
    # 检查连接结果
    successful_connections = [r for r in results if r["success"]]
    if successful_connections:
        print(f"\n✅ ES连接测试成功")
        print(f"   连接地址: {successful_connections[0]['url']}")
        print(f"   连接方式: {successful_connections[0]['description']}")
        return 0
    else:
        print("\n❌ ES连接测试失败")
        print("\n可能的原因:")
        print("1. ES服务未启动或服务名不是 'elasticsearch'")
        print("2. ES和Jupyter不在同一个Docker网络中")
        print("3. 网络连接问题")
        print("\n检查步骤:")
        print("1. 确认ES服务名: docker ps | grep elasticsearch")
        print("2. 测试连接: curl http://elasticsearch:9200")
        print("3. 检查Docker网络: docker network ls 和 docker network inspect <network_name>")
        return 1

if __name__ == "__main__":
    sys.exit(main())

