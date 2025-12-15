# coding="utf-8"
"""
阿里云 Elasticsearch 使用示例 Demo

演示如何使用统一的 ES 客户端进行：
1. 连接测试
2. 创建索引
3. 批量写入数据
4. 查询数据
"""
from ..es_client import es
import time
import string
import random

letters = string.ascii_lowercase + string.digits

def demo_bulk_write():
    """批量写入示例"""
    index_name = "demo_index_%s" % time.strftime('%H%M%S')
    
    # 创建索引
    try:
        es.indices.create(
            index=index_name,
            body={"settings": {"number_of_shards": 1}}
        )
        print(f"✓ 索引创建成功: {index_name}")
    except Exception as e:
        print(f"✗ 索引创建失败: {e}")
        return
    
    # 构造批量数据
    bulk_body = []
    for i in range(10):  # 示例：写入10条数据
        _id = ''.join([random.choice(letters) for _ in range(17)])
        bulk_body.extend([
            {"create": {"_index": index_name, "_id": _id}},
            {
                "title": f"示例文档 {i+1}",
                "content": "这是一个测试文档的内容",
                "timestamp": int(time.time())
            }
        ])
    
    # 批量写入
    try:
        response = es.bulk(body=bulk_body, timeout='10s')
        if response.get('errors'):
            print("⚠ 部分文档写入失败")
        else:
            print(f"✓ 成功写入 {len(bulk_body)//2} 条文档")
    except Exception as e:
        print(f"✗ 批量写入失败: {e}")
        return
    
    # 刷新索引
    es.indices.refresh(index=index_name)
    
    # 查询数据
    try:
        query_body = {"query": {"match_all": {}}, "size": 5}
        search_response = es.search(body=query_body, index=index_name)
        print(f"✓ 查询成功，找到 {search_response['hits']['total']['value']} 条结果")
    except Exception as e:
        print(f"✗ 查询失败: {e}")

if __name__ == '__main__':
    print("=" * 60)
    print("阿里云 ES 使用示例")
    print("=" * 60)
    
    # 测试连接
    try:
        if es.ping():
            info = es.info()
            print(f"✓ 连接成功 - ES版本: {info['version']['number']}")
            print()
        else:
            print("✗ 连接失败")
            exit(1)
    except Exception as e:
        print(f"✗ 连接错误: {e}")
        exit(1)
    
    # 运行示例
    demo_bulk_write()
    
    print()
    print("=" * 60)
    print("示例完成")
    print("=" * 60)

