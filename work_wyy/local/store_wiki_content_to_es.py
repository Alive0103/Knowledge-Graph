#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将find_wiki_content目录下的文本文件存储到ES索引

功能：
1. 读取find_wiki_content目录下的所有txt文件
2. 解析文件内容（查询词、链接、文本内容）
3. 存储到ES索引（支持更新现有文档或创建新文档）
注意：本脚本只存储文本内容，不生成向量，用于文本检索测评
"""

import os
import sys
import re
from tqdm import tqdm
from datetime import datetime
from urllib.parse import unquote

# 导入配置和ES客户端
try:
    from config import WORK_DIR, ES_INDEX_NAME
    from es_client import es
except ImportError:
    WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ES_INDEX_NAME = 'data2'
    from es_client import es


def parse_txt_file(file_path):
    """解析txt文件，提取元数据和内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析元数据（文件开头的注释行）
        query = None
        link = None
        download_time = None
        
        lines = content.split('\n')
        metadata_end = 0
        
        for i, line in enumerate(lines):
            if line.startswith('# 查询词:'):
                query = line.replace('# 查询词:', '').strip()
            elif line.startswith('# 链接:'):
                link = line.replace('# 链接:', '').strip()
            elif line.startswith('# 下载时间:'):
                download_time = line.replace('# 下载时间:', '').strip()
            elif line.startswith('# ='):
                metadata_end = i + 1
                break
        
        # 提取正文内容（跳过元数据部分）
        text_content = '\n'.join(lines[metadata_end:]).strip()
        
        return {
            'query': query,
            'link': link,
            'download_time': download_time,
            'content': text_content,
            'filename': os.path.basename(file_path)
        }
    except Exception as e:
        print(f"解析文件失败 {file_path}: {e}")
        return None


def find_document_by_link(link, index_name):
    """在ES中根据link查找现有文档"""
    if not link:
        return None
    
    try:
        search_body = {
            "query": {
                "term": {
                    "link": link
                }
            },
            "size": 1
        }
        resp = es.search(index=index_name, body=search_body)
        hits = resp.get("hits", {}).get("hits", [])
        if hits:
            return hits[0].get("_id")
    except Exception as e:
        pass
    
    return None


def ensure_wiki_content_field(index_name):
    """确保索引有wiki_content字段（如果不存在则添加）"""
    try:
        mapping = es.indices.get_mapping(index=index_name)
        properties = mapping[index_name]['mappings'].get('properties', {})
        
        # 检查是否有wiki_content字段
        if 'wiki_content' not in properties:
            print(f"索引 {index_name} 缺少 wiki_content 字段，尝试添加...")
            try:
                # 添加wiki_content字段到映射（用于文本检索）
                es.indices.put_mapping(
                    index=index_name,
                    body={
                        "properties": {
                            "wiki_content": {
                                "type": "text",
                                "analyzer": "ik_max_word",
                                "search_analyzer": "ik_smart"
                            }
                        }
                    }
                )
                print(f"✓ 成功添加 wiki_content 字段（用于文本检索）")
            except Exception as e:
                print(f"警告: 添加 wiki_content 字段失败: {e}")
                print("  将尝试使用默认的text类型")
    except Exception as e:
        print(f"检查索引映射失败: {e}")


def create_or_update_document(file_data, index_name, match_existing=True):
    """创建或更新ES文档（只存储文本，不生成向量）"""
    query = file_data.get('query', '')
    link = file_data.get('link', '')
    content = file_data.get('content', '')
    
    if not content:
        return False
    
    try:
        # 准备文档数据（只包含文本内容，不生成向量）
        doc_data = {
            'wiki_content': content,
            'wiki_content_length': len(content),
            'wiki_download_time': file_data.get('download_time', ''),
            'wiki_filename': file_data.get('filename', '')
        }
        
        # 如果match_existing=True，尝试查找并更新现有文档
        if match_existing and link:
            doc_id = find_document_by_link(link, index_name)
            if doc_id:
                # 更新现有文档
                update_data = {
                    'doc': doc_data
                }
                es.update(index=index_name, id=doc_id, body=update_data)
                return True
        
        # 创建新文档（如果match_existing=False或找不到现有文档）
        # 注意：这里创建新文档时，需要确保有link字段用于匹配
        if link:
            doc_data['link'] = link
        if query:
            doc_data['query_from_file'] = query
        
        es.index(index=index_name, body=doc_data)
        return True
        
    except Exception as e:
        print(f"创建/更新文档失败 ({query}): {e}")
        return False


def store_content_to_es(content_dir, index_name, match_existing=True):
    """将目录下的所有txt文件存储到ES（只存储文本，不生成向量）"""
    print(f"\n开始处理目录: {content_dir}")
    print(f"目标索引: {index_name}")
    print(f"匹配模式: {'更新现有文档' if match_existing else '创建新文档'}")
    print(f"注意: 只存储文本内容，不生成向量（用于文本检索测评）")
    print("=" * 70)
    
    # 确保索引有wiki_content字段
    ensure_wiki_content_field(index_name)
    
    # 获取所有txt文件
    txt_files = [f for f in os.listdir(content_dir) if f.endswith('.txt')]
    
    if not txt_files:
        print(f"错误: 目录 {content_dir} 中没有找到txt文件")
        return
    
    print(f"找到 {len(txt_files)} 个txt文件")
    
    success_count = 0
    failed_count = 0
    
    for filename in tqdm(txt_files, desc="处理文件"):
        file_path = os.path.join(content_dir, filename)
        file_data = parse_txt_file(file_path)
        
        if not file_data:
            failed_count += 1
            continue
        
        if create_or_update_document(file_data, index_name, match_existing):
            success_count += 1
        else:
            failed_count += 1
    
    print("\n" + "=" * 70)
    print("处理完成统计")
    print("=" * 70)
    print(f"总文件数: {len(txt_files)}")
    print(f"成功: {success_count} 个")
    print(f"失败: {failed_count} 个")
    print(f"成功率: {success_count/len(txt_files)*100:.1f}%")
    
    # 验证索引
    try:
        result = es.count(index=index_name)
        print(f"\n索引 {index_name} 当前文档数: {result['count']}")
    except Exception as e:
        print(f"验证索引时出错: {e}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='将find_wiki_content目录下的文本文件存储到ES索引（只存储文本，不生成向量）',
        epilog='注意：本脚本只存储文本内容到wiki_content字段，不生成向量，用于文本检索测评'
    )
    parser.add_argument('--content-dir', type=str, default=None,
                       help='文本文件目录（默认: work_wyy/data/find_wiki_content）')
    parser.add_argument('--index', type=str, default=None,
                       help='ES索引名称（默认: 使用config.py中的ES_INDEX_NAME）')
    parser.add_argument('--match-existing', action='store_true', default=True,
                       help='如果找到相同link的文档则更新，否则创建新文档（默认: True）')
    parser.add_argument('--no-match-existing', dest='match_existing', action='store_false',
                       help='总是创建新文档，不更新现有文档')
    
    args = parser.parse_args()
    
    # 获取work_wyy目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    work_wyy_dir = os.path.dirname(script_dir)
    
    # 确定内容目录
    if args.content_dir:
        content_dir = args.content_dir
        if not os.path.isabs(content_dir):
            content_dir = os.path.abspath(content_dir)
    else:
        content_dir = os.path.join(work_wyy_dir, 'data', 'find_wiki_content')
    
    if not os.path.exists(content_dir):
        print(f"错误: 目录不存在: {content_dir}")
        sys.exit(1)
    
    # 确定索引名称
    index_name = args.index if args.index else ES_INDEX_NAME
    
    print("=" * 70)
    print("将wiki文本内容存储到ES索引（文本检索模式）")
    print("=" * 70)
    print(f"内容目录: {content_dir}")
    print(f"目标索引: {index_name}")
    print(f"注意: 只存储文本内容，不生成向量")
    print("=" * 70)
    
    # 检查索引是否存在
    if not es.indices.exists(index=index_name):
        print(f"警告: 索引 {index_name} 不存在，将自动创建")
        # 这里可以调用create_vector_index，但为了简化，假设索引已存在
        # 如果需要自动创建，可以导入create_vector_index函数
    
    # 存储内容到ES
    store_content_to_es(content_dir, index_name, match_existing=args.match_existing)
    
    print("\n" + "=" * 70)
    print("✓ 完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
