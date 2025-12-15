#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将find_wiki_content文件夹中的文本内容存入ES索引

功能：
1. 读取find_wiki_content文件夹中的所有txt文件
2. 解析文件内容（查询词、链接、文本）
3. 匹配到ES索引中的对应实体（通过link匹配）
4. 更新或创建文档，添加content字段和对应的向量
"""

import os
import sys
import re
import json
import glob
from urllib.parse import unquote
from tqdm import tqdm
from datetime import datetime

# 导入配置和ES客户端
try:
    from config import WORK_DIR
    from es_client import es
except ImportError:
    WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    from es_client import es

# 导入向量生成相关模块
try:
    from vector_model import load_vector_model, generate_vector as _generate_vector_module, batch_generate_vectors
    model, tokenizer, device = load_vector_model(use_finetuned=True)
    print(f"✓ 向量生成模型加载成功 (使用微调后的模型，设备: {device})")
except Exception as e:
    print(f"警告: 向量生成模型加载失败 ({e})")
    model = None
    tokenizer = None

# 导入vector2ES的相关函数
try:
    from vector2ES import (
        VECTOR_DIMS,
        generate_vector
    )
except ImportError:
    print("错误: 无法导入vector2ES模块")
    sys.exit(1)

# 目标索引名称
TARGET_INDEX = 'data_config_5_all_except_msra'


def normalize_url(url):
    """归一化URL，处理URL编码问题"""
    if not url:
        return ""
    
    url = str(url).strip()
    
    if "wikipedia.org/wiki/" in url:
        try:
            if "/wiki/" in url:
                parts = url.split("/wiki/", 1)
                if len(parts) == 2:
                    title = parts[1]
                    # 移除URL参数
                    if '#' in title:
                        title = title.split('#')[0]
                    if '?' in title:
                        title = title.split('?')[0]
                    try:
                        decoded_title = unquote(title)
                        return decoded_title
                    except:
                        return title
        except Exception:
            pass
    
    return url


def parse_txt_file(file_path):
    """解析txt文件，提取元数据和内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析元数据
        query = None
        link = None
        download_time = None
        
        # 提取查询词
        query_match = re.search(r'# 查询词:\s*(.+)', content)
        if query_match:
            query = query_match.group(1).strip()
        
        # 提取链接
        link_match = re.search(r'# 链接:\s*(.+)', content)
        if link_match:
            link = link_match.group(1).strip()
        
        # 提取下载时间
        time_match = re.search(r'# 下载时间:\s*(.+)', content)
        if time_match:
            download_time = time_match.group(1).strip()
        
        # 提取正文（跳过元数据行）
        lines = content.split('\n')
        text_lines = []
        skip_meta = True
        
        for line in lines:
            if line.startswith('# ='):
                skip_meta = False
                continue
            if skip_meta:
                continue
            if line.strip():
                text_lines.append(line)
        
        text_content = '\n'.join(text_lines).strip()
        
        return {
            'query': query,
            'link': link,
            'download_time': download_time,
            'content': text_content,
            'file_path': file_path
        }
    except Exception as e:
        print(f"解析文件失败 {file_path}: {e}")
        return None


def find_document_by_link(link, index_name):
    """在ES中通过link查找文档"""
    if not link:
        return None
    
    normalized_link = normalize_url(link)
    
    try:
        # 尝试精确匹配
        search_body = {
            "query": {
                "term": {
                    "link": normalized_link
                }
            },
            "size": 1
        }
        
        resp = es.search(index=index_name, body=search_body)
        hits = resp.get("hits", {}).get("hits", [])
        
        if hits:
            return hits[0]
        
        # 尝试模糊匹配（wildcard）
        search_body = {
            "query": {
                "wildcard": {
                    "link": f"*{normalized_link}*"
                }
            },
            "size": 1
        }
        
        resp = es.search(index=index_name, body=search_body)
        hits = resp.get("hits", {}).get("hits", [])
        
        if hits:
            return hits[0]
        
        # 尝试原始链接匹配
        if link != normalized_link:
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
                return hits[0]
        
        return None
    except Exception as e:
        print(f"搜索文档失败: {e}")
        return None


def ensure_content_vector_field(index_name):
    """确保索引有content_vector字段（如果不存在则添加）"""
    try:
        # 获取当前索引映射
        mapping = es.indices.get_mapping(index=index_name)
        properties = mapping[index_name]['mappings'].get('properties', {})
        
        # 检查是否有content_vector字段
        if 'content_vector' not in properties:
            print(f"索引 {index_name} 缺少 content_vector 字段，尝试添加...")
            
            # 添加content_vector字段到映射
            update_mapping = {
                "properties": {
                    "content_vector": {
                        "type": "dense_vector",
                        "dims": VECTOR_DIMS,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }
            
            try:
                es.indices.put_mapping(index=index_name, body=update_mapping)
                print(f"✓ 成功添加 content_vector 字段")
            except Exception as e:
                print(f"警告: 添加 content_vector 字段失败: {e}")
                print(f"  将使用 descriptions_zh_vector 字段存储content向量")
    except Exception as e:
        print(f"检查索引映射失败: {e}")


def create_document(file_data, index_name, match_existing=False):
    """创建ES文档（直接存储，用于检索）"""
    link = file_data.get('link')
    content = file_data.get('content', '')
    query = file_data.get('query', '')
    
    if not link or not content:
        return False, 'skipped', None
    
    try:
        # 如果启用匹配模式，先查找现有文档
        if match_existing:
            existing_doc = find_document_by_link(link, index_name)
            if existing_doc:
                # 更新现有文档
                doc_id = existing_doc.get('_id')
                existing_source = existing_doc.get('_source', {})
                
                update_data = {'content': content}
                
                # 为content生成向量
                if content and len(content.strip()) > 10:
                    content_for_vector = content[:5000] if len(content) > 5000 else content
                    content_vector = generate_vector(content_for_vector)
                    
                    if content_vector and len(content_vector) == VECTOR_DIMS:
                        try:
                            mapping = es.indices.get_mapping(index=index_name)
                            has_content_vector = 'content_vector' in mapping[index_name]['mappings'].get('properties', {})
                            if has_content_vector:
                                update_data['content_vector'] = content_vector
                            elif not existing_source.get('descriptions_zh_vector'):
                                update_data['descriptions_zh_vector'] = content_vector
                        except:
                            update_data['content_vector'] = content_vector
                
                es.update(index=index_name, id=doc_id, body={'doc': update_data})
                return True, 'updated', doc_id
        
        # 创建新文档（直接存储，不匹配）
        new_doc = {
            'label': query or '',
            'link': link,
            'content': content,
            'aliases_zh': [],
            'aliases_en': [],
            'descriptions_zh': content[:500] if content else '',  # 前500字符作为描述
            'descriptions_en': ''
        }
        
        # 生成向量
        if query:
            label_vector = generate_vector(query)
            if label_vector and len(label_vector) == VECTOR_DIMS:
                new_doc['label_vector'] = label_vector
                new_doc['label_zh_vector'] = label_vector
        
        if content and len(content.strip()) > 10:
            # 为content生成向量
            content_for_vector = content[:5000] if len(content) > 5000 else content
            content_vector = generate_vector(content_for_vector)
            if content_vector and len(content_vector) == VECTOR_DIMS:
                try:
                    mapping = es.indices.get_mapping(index=index_name)
                    has_content_vector = 'content_vector' in mapping[index_name]['mappings'].get('properties', {})
                    if has_content_vector:
                        new_doc['content_vector'] = content_vector
                    else:
                        new_doc['descriptions_zh_vector'] = content_vector
                except:
                    new_doc['content_vector'] = content_vector
            
            # 为描述生成向量
            if new_doc.get('descriptions_zh'):
                desc_vector = generate_vector(new_doc['descriptions_zh'])
                if desc_vector and len(desc_vector) == VECTOR_DIMS:
                    new_doc['descriptions_zh_vector'] = desc_vector
        
        # 创建文档
        es.index(index=index_name, body=new_doc)
        
        return True, 'created', None
    except Exception as e:
        print(f"创建文档失败 ({query}): {e}")
        import traceback
        traceback.print_exc()
        return False, 'error', None


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='将find_wiki_content文件夹中的文本内容存入ES索引')
    parser.add_argument('--content-dir', type=str,
                       default=None,
                       help='find_wiki_content文件夹路径（默认: work_wyy/data/find_wiki_content）')
    parser.add_argument('--index', type=str,
                       default=TARGET_INDEX,
                       help=f'目标ES索引名称（默认: {TARGET_INDEX}）')
    parser.add_argument('--batch-size', type=int, default=20,
                       help='批量处理大小（默认: 20）')
    parser.add_argument('--match-existing', action='store_true',
                       help='如果启用，会尝试匹配现有文档并更新；否则直接创建新文档')
    
    args = parser.parse_args()
    
    # 获取work_wyy目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    work_wyy_dir = os.path.dirname(script_dir)
    
    # 确定内容文件夹路径
    if args.content_dir:
        content_dir = args.content_dir
        if not os.path.isabs(content_dir):
            content_dir = os.path.abspath(content_dir)
    else:
        content_dir = os.path.join(work_wyy_dir, 'data', 'find_wiki_content')
    
    if not os.path.exists(content_dir):
        print(f"错误: 内容文件夹不存在: {content_dir}")
        sys.exit(1)
    
    # 检查索引是否存在
    if not es.indices.exists(index=args.index):
        print(f"错误: 索引 {args.index} 不存在")
        print(f"  请先确保索引已创建")
        sys.exit(1)
    
    print("=" * 70)
    print("将find_wiki_content文件夹中的文本内容存入ES索引")
    print("=" * 70)
    print(f"内容文件夹: {content_dir}")
    print(f"目标索引: {args.index}")
    print(f"匹配模式: {'是（更新现有文档）' if args.match_existing else '否（直接创建新文档）'}")
    print("=" * 70)
    
    # 0. 确保索引有content_vector字段
    print(f"\n检查索引映射...")
    ensure_content_vector_field(args.index)
    
    # 1. 读取所有txt文件
    txt_files = glob.glob(os.path.join(content_dir, '*.txt'))
    # 排除failed_links.txt
    txt_files = [f for f in txt_files if not os.path.basename(f) == 'failed_links.txt']
    
    print(f"\n找到 {len(txt_files)} 个txt文件")
    
    if not txt_files:
        print("错误: 没有找到txt文件")
        sys.exit(1)
    
    # 2. 解析所有文件
    print(f"\n开始解析文件...")
    file_data_list = []
    
    for txt_file in tqdm(txt_files, desc="解析文件"):
        data = parse_txt_file(txt_file)
        if data and data.get('content'):
            file_data_list.append(data)
    
    print(f"✓ 成功解析 {len(file_data_list)} 个文件")
    
    if not file_data_list:
        print("错误: 没有成功解析的文件")
        sys.exit(1)
    
    # 3. 更新或创建ES文档
    print(f"\n开始更新/创建ES文档...")
    
    stats = {
        'total': len(file_data_list),
        'updated': 0,
        'created': 0,
        'failed': 0,
        'not_found': 0
    }
    
    failed_items = []
    
    for file_data in tqdm(file_data_list, desc="处理文档"):
        try:
            success, action, doc_id = create_document(file_data, args.index, match_existing=args.match_existing)
            
            if success:
                if action == 'updated':
                    stats['updated'] += 1
                elif action == 'created':
                    stats['created'] += 1
                elif action == 'skipped':
                    stats['not_found'] += 1
            else:
                stats['failed'] += 1
                failed_items.append(file_data)
        except Exception as e:
            print(f"处理文档时出错: {e}")
            stats['failed'] += 1
            failed_items.append(file_data)
    
    # 4. 生成报告
    print("\n" + "=" * 70)
    print("处理结果统计")
    print("=" * 70)
    print(f"总文件数: {stats['total']}")
    print(f"更新文档: {stats['updated']} 个")
    print(f"创建文档: {stats['created']} 个")
    print(f"跳过: {stats['not_found']} 个")
    print(f"失败: {stats['failed']} 个")
    if stats['total'] > 0:
        success_rate = (stats['updated'] + stats['created']) / stats['total'] * 100
        print(f"成功率: {success_rate:.1f}%")
    
    # 保存失败列表
    if failed_items:
        import datetime
        failed_file = os.path.join(work_wyy_dir, 'trainlog', 
                                   f'failed_content_upload_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        os.makedirs(os.path.dirname(failed_file), exist_ok=True)
        
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(failed_items, f, ensure_ascii=False, indent=2)
        
        print(f"\n失败项已保存到: {failed_file}")
    
    # 验证索引
    try:
        result = es.count(index=args.index)
        print(f"\n索引 {args.index} 中的文档总数: {result['count']}")
    except Exception as e:
        print(f"验证索引时出错: {e}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
