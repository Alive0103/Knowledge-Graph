#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find索引管理工具

功能：
1. build-entities: 从find.xlsx构建实体索引（data2_find）
2. build-queries: 从find.xlsx构建查询词索引（data2_find_queries）
3. compare: 对比两个索引的检索效果

使用方法:
    python find_index_manager.py build-entities [选项]
    python find_index_manager.py build-queries [选项]
    python find_index_manager.py compare [选项]
"""

import os
import sys
import json
import pandas as pd
from urllib.parse import unquote
from tqdm import tqdm
from datetime import datetime

# 导入配置和ES客户端
try:
    from config import (
        ES_INDEX_NAME,
        WORK_DIR,
        ZH_WIKI_FILE
    )
    from es_client import es
except ImportError:
    ES_INDEX_NAME = 'data2'
    WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ZH_WIKI_FILE = os.path.join(WORK_DIR, 'data', 'zh_wiki_v2.jsonl')
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
    device = None

# 导入vector2ES的相关函数
try:
    from vector2ES import (
        VECTOR_DIMS,
        create_vector_index,
        process_single_item,
        generate_vector
    )
except ImportError:
    print("错误: 无法导入vector2ES模块")
    sys.exit(1)

# 索引名称
FIND_INDEX_NAME = 'data2_find'
QUERIES_INDEX_NAME = 'data2_find_queries'


# ============================================================================
# 通用工具函数
# ============================================================================

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


def extract_title_from_url(url):
    """从URL中提取维基百科标题"""
    return normalize_url(url) if url else None


def read_find_excel(file_path):
    """读取find.xlsx文件"""
    df = pd.read_excel(file_path, header=None)
    queries = df[0].tolist()
    correct_links = df[1].tolist()
    
    # 清理数据，移除NaN
    queries = [str(q).strip() if pd.notna(q) else "" for q in queries]
    correct_links = [str(link).strip() if pd.notna(link) else "" for link in correct_links]
    
    # 过滤空值
    valid_pairs = [(q, link) for q, link in zip(queries, correct_links) if q and link]
    
    print(f"从 {file_path} 读取了 {len(valid_pairs)} 个有效查询-链接对")
    return valid_pairs


# ============================================================================
# 功能1: 构建实体索引
# ============================================================================

def load_wiki_data(wiki_file_path):
    """加载wiki数据到内存，建立label、别名、link和URL标题到数据的映射"""
    print(f"正在加载wiki数据: {wiki_file_path}")
    
    label_to_data = {}
    alias_to_data = {}
    link_to_data = {}
    url_title_to_data = {}
    total_lines = 0
    
    if not os.path.exists(wiki_file_path):
        print(f"错误: 文件不存在: {wiki_file_path}")
        return label_to_data, alias_to_data, link_to_data, url_title_to_data
    
    # 先统计行数
    with open(wiki_file_path, 'r', encoding='utf-8') as f:
        for _ in f:
            total_lines += 1
    
    # 读取数据
    sample_printed = False
    with open(wiki_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, total=total_lines, desc="加载wiki数据"), 1):
            try:
                data = json.loads(line.strip())
                if not data:
                    continue
                
                # 打印前几行的数据结构（用于调试）
                if not sample_printed and line_num <= 3:
                    print(f"\n  第{line_num}行数据字段: {list(data.keys())[:10]}")
                    if 'label' in data:
                        print(f"    label字段值: {data.get('label')}")
                    if line_num == 3:
                        sample_printed = True
                
                # 获取label字段
                label = None
                if 'label' in data:
                    label = data.get('label')
                elif 'itemLabel' in data:
                    item_label = data.get('itemLabel')
                    if isinstance(item_label, dict):
                        label = item_label.get('value', '')
                    elif isinstance(item_label, str):
                        label = item_label
                
                if not label:
                    for key in ['name', 'title', 'entity_name', 'entityLabel']:
                        if key in data:
                            label = data.get(key)
                            break
                
                if label and not isinstance(label, str):
                    if isinstance(label, dict):
                        label = label.get('value', '') or str(label)
                    else:
                        label = str(label)
                
                # 获取link字段
                link = data.get('link') or data.get('wikipedia') or data.get('wikipediaLink') or \
                       data.get('wikipediaLink_zh') or data.get('wikipediaLink_en')
                
                # 获取别名字段
                aliases_zh = data.get('zh_aliases') or data.get('aliases_zh') or []
                aliases_en = data.get('en_aliases') or data.get('aliases_en') or []
                if not isinstance(aliases_zh, list):
                    aliases_zh = [aliases_zh] if aliases_zh else []
                if not isinstance(aliases_en, list):
                    aliases_en = [aliases_en] if aliases_en else []
                
                # 存储label映射
                if label and label.strip():
                    label_clean = label.strip()
                    if label_clean not in label_to_data:
                        label_to_data[label_clean] = data
                
                # 存储别名映射
                all_aliases = aliases_zh + aliases_en
                for alias in all_aliases:
                    if alias and isinstance(alias, str) and alias.strip():
                        alias_clean = alias.strip()
                        if alias_clean not in alias_to_data:
                            alias_to_data[alias_clean] = data
                
                # 存储link映射
                if link:
                    normalized_link = normalize_url(link)
                    if normalized_link:
                        link_to_data[normalized_link] = data
                        if link != normalized_link:
                            link_to_data[link] = data
                    
                    # 从URL提取标题并建立映射
                    url_title = extract_title_from_url(link)
                    if url_title and url_title.strip():
                        url_title_clean = url_title.strip()
                        if url_title_clean not in url_title_to_data:
                            url_title_to_data[url_title_clean] = data
                
            except json.JSONDecodeError:
                continue
            except Exception:
                continue
    
    print(f"加载完成:")
    print(f"  - label映射: {len(label_to_data)} 个")
    print(f"  - 别名映射: {len(alias_to_data)} 个")
    print(f"  - link映射: {len(link_to_data)} 个")
    print(f"  - URL标题映射: {len(url_title_to_data)} 个")
    return label_to_data, alias_to_data, link_to_data, url_title_to_data


def calculate_similarity(str1, str2):
    """计算两个字符串的相似度"""
    if not str1 or not str2:
        return 0.0
    
    str1_lower = str1.lower()
    str2_lower = str2.lower()
    
    if str1_lower == str2_lower:
        return 1.0
    
    if str1_lower in str2_lower or str2_lower in str1_lower:
        shorter = min(len(str1_lower), len(str2_lower))
        longer = max(len(str1_lower), len(str2_lower))
        return shorter / longer if longer > 0 else 0.0
    
    return 0.0


def match_entities_from_find(find_pairs, label_to_data, alias_to_data, link_to_data, url_title_to_data):
    """从wiki数据中匹配find.xlsx中的实体"""
    matched_entities = []
    unmatched_pairs = []
    match_methods = {
        'label_exact': 0, 'label_fuzzy': 0, 'alias_exact': 0, 'alias_fuzzy': 0,
        'url_title_exact': 0, 'url_title_fuzzy': 0, 'link_exact': 0, 'link_fuzzy': 0
    }
    used_data_ids = set()
    
    print("\n开始匹配实体（使用多种匹配策略）...")
    
    for query, correct_link in tqdm(find_pairs, desc="匹配实体"):
        matched_data = None
        match_method = None
        best_similarity = 0.0
        
        query_clean = query.strip()
        normalized_link = normalize_url(correct_link)
        url_title = extract_title_from_url(correct_link)
        
        # 策略1: 精确匹配label
        if query_clean in label_to_data:
            candidate = label_to_data[query_clean]
            data_id = candidate.get('link') or candidate.get('wikipedia') or str(candidate)
            if data_id not in used_data_ids:
                matched_data = candidate
                match_method = 'label_exact'
                match_methods['label_exact'] += 1
                used_data_ids.add(data_id)
        
        # 策略2: 精确匹配别名
        if not matched_data and query_clean in alias_to_data:
            candidate = alias_to_data[query_clean]
            data_id = candidate.get('link') or candidate.get('wikipedia') or str(candidate)
            if data_id not in used_data_ids:
                matched_data = candidate
                match_method = 'alias_exact'
                match_methods['alias_exact'] += 1
                used_data_ids.add(data_id)
        
        # 策略3: 精确匹配URL标题
        if not matched_data and url_title and url_title.strip() in url_title_to_data:
            candidate = url_title_to_data[url_title.strip()]
            data_id = candidate.get('link') or candidate.get('wikipedia') or str(candidate)
            if data_id not in used_data_ids:
                matched_data = candidate
                match_method = 'url_title_exact'
                match_methods['url_title_exact'] += 1
                used_data_ids.add(data_id)
        
        # 策略4: 精确匹配link
        if not matched_data:
            if normalized_link in link_to_data:
                candidate = link_to_data[normalized_link]
                data_id = normalized_link
                if data_id not in used_data_ids:
                    matched_data = candidate
                    match_method = 'link_exact'
                    match_methods['link_exact'] += 1
                    used_data_ids.add(data_id)
            elif correct_link in link_to_data:
                candidate = link_to_data[correct_link]
                data_id = correct_link
                if data_id not in used_data_ids:
                    matched_data = candidate
                    match_method = 'link_exact'
                    match_methods['link_exact'] += 1
                    used_data_ids.add(data_id)
        
        # 策略5: 模糊匹配label
        if not matched_data:
            for label_key, candidate in label_to_data.items():
                similarity = calculate_similarity(query_clean, label_key)
                if similarity > 0.3:
                    data_id = candidate.get('link') or candidate.get('wikipedia') or str(candidate)
                    if data_id not in used_data_ids and similarity > best_similarity:
                        matched_data = candidate
                        match_method = 'label_fuzzy'
                        best_similarity = similarity
        
        if matched_data and match_method == 'label_fuzzy':
            data_id = matched_data.get('link') or matched_data.get('wikipedia') or str(matched_data)
            used_data_ids.add(data_id)
            match_methods['label_fuzzy'] += 1
        
        # 策略6: 模糊匹配别名
        if not matched_data:
            for alias_key, candidate in alias_to_data.items():
                similarity = calculate_similarity(query_clean, alias_key)
                if similarity > 0.3:
                    data_id = candidate.get('link') or candidate.get('wikipedia') or str(candidate)
                    if data_id not in used_data_ids and similarity > best_similarity:
                        matched_data = candidate
                        match_method = 'alias_fuzzy'
                        best_similarity = similarity
        
        if matched_data and match_method == 'alias_fuzzy':
            data_id = matched_data.get('link') or matched_data.get('wikipedia') or str(matched_data)
            used_data_ids.add(data_id)
            match_methods['alias_fuzzy'] += 1
        
        # 策略7: 模糊匹配URL标题
        if not matched_data and url_title:
            url_title_clean = url_title.strip()
            for title_key, candidate in url_title_to_data.items():
                similarity = calculate_similarity(query_clean, title_key)
                if similarity > 0.3:
                    data_id = candidate.get('link') or candidate.get('wikipedia') or str(candidate)
                    if data_id not in used_data_ids and similarity > best_similarity:
                        matched_data = candidate
                        match_method = 'url_title_fuzzy'
                        best_similarity = similarity
        
        if matched_data and match_method == 'url_title_fuzzy':
            data_id = matched_data.get('link') or matched_data.get('wikipedia') or str(matched_data)
            used_data_ids.add(data_id)
            match_methods['url_title_fuzzy'] += 1
        
        # 策略8: 模糊匹配link
        if not matched_data:
            for link_key, candidate in link_to_data.items():
                link_normalized = normalize_url(link_key)
                if (normalized_link in link_normalized or 
                    link_normalized in normalized_link or
                    correct_link in link_key or
                    link_key in correct_link):
                    data_id = link_key
                    if data_id not in used_data_ids:
                        matched_data = candidate
                        match_method = 'link_fuzzy'
                        match_methods['link_fuzzy'] += 1
                        used_data_ids.add(data_id)
                        break
        
        if matched_data:
            matched_data_copy = matched_data.copy() if isinstance(matched_data, dict) else matched_data
            if isinstance(matched_data_copy, dict):
                matched_data_copy['_query'] = query
                matched_data_copy['_correct_link'] = correct_link
                matched_data_copy['_match_method'] = match_method
            matched_entities.append(matched_data_copy)
        else:
            unmatched_pairs.append((query, correct_link))
    
    matched_count = len(matched_entities)
    total_count = len(find_pairs)
    
    print(f"\n匹配结果:")
    print(f"  总查询数: {total_count}")
    print(f"  成功匹配: {matched_count} 个实体 ({matched_count/total_count*100:.1f}%)")
    print(f"  未匹配: {len(unmatched_pairs)} 个 ({len(unmatched_pairs)/total_count*100:.1f}%)")
    print(f"\n匹配方法统计:")
    if matched_count > 0:
        print(f"    - label精确匹配: {match_methods['label_exact']} 个 ({match_methods['label_exact']/matched_count*100:.1f}%)")
        print(f"    - label模糊匹配: {match_methods['label_fuzzy']} 个 ({match_methods['label_fuzzy']/matched_count*100:.1f}%)")
        print(f"    - 别名精确匹配: {match_methods['alias_exact']} 个 ({match_methods['alias_exact']/matched_count*100:.1f}%)")
        print(f"    - 别名模糊匹配: {match_methods['alias_fuzzy']} 个 ({match_methods['alias_fuzzy']/matched_count*100:.1f}%)")
        print(f"    - URL标题精确匹配: {match_methods['url_title_exact']} 个 ({match_methods['url_title_exact']/matched_count*100:.1f}%)")
        print(f"    - URL标题模糊匹配: {match_methods['url_title_fuzzy']} 个 ({match_methods['url_title_fuzzy']/matched_count*100:.1f}%)")
        print(f"    - link精确匹配: {match_methods['link_exact']} 个 ({match_methods['link_exact']/matched_count*100:.1f}%)")
        print(f"    - link模糊匹配: {match_methods['link_fuzzy']} 个 ({match_methods['link_fuzzy']/matched_count*100:.1f}%)")
    
    if unmatched_pairs:
        print(f"\n未匹配的实体（前10个，共{len(unmatched_pairs)}个）:")
        for i, (query, link) in enumerate(unmatched_pairs[:10], 1):
            print(f"  {i}. 查询: {query}")
            print(f"     链接: {link}")
    
    return matched_entities, unmatched_pairs, match_methods


def calculate_field_statistics(processed_entities):
    """计算字段非空率统计"""
    if not processed_entities:
        return {}
    
    fields_to_check = [
        'label', 'link', 'aliases_zh', 'aliases_en', 'descriptions_zh', 'descriptions_en', 'content',
        'label_zh_vector', 'label_en_vector', 'descriptions_zh_vector', 'descriptions_en_vector',
        'entity_words_zh_vector', 'entity_words_en_vector'
    ]
    
    stats = {}
    total = len(processed_entities)
    
    for field in fields_to_check:
        non_empty_count = 0
        for entity in processed_entities:
            value = entity.get(field)
            if value is not None:
                if isinstance(value, (list, dict)):
                    if value:
                        non_empty_count += 1
                elif isinstance(value, str):
                    if value.strip():
                        non_empty_count += 1
                else:
                    non_empty_count += 1
        
        non_empty_rate = (non_empty_count / total * 100) if total > 0 else 0
        stats[field] = {
            'non_empty_count': non_empty_count,
            'non_empty_rate': non_empty_rate,
            'empty_count': total - non_empty_count
        }
    
    return stats


def build_entities_index(matched_entities):
    """构建实体索引"""
    print(f"\n开始构建ES索引: {FIND_INDEX_NAME}")
    
    # 创建索引
    from vector2ES import create_vector_index as _create_index
    import vector2ES
    
    original_index = vector2ES.INDEX_NAME
    vector2ES.INDEX_NAME = FIND_INDEX_NAME
    
    try:
        if not _create_index():
            print("创建索引失败")
            return False, []
    finally:
        vector2ES.INDEX_NAME = original_index
    
    # 处理实体并导入ES
    print(f"\n开始处理 {len(matched_entities)} 个实体...")
    
    from elasticsearch import helpers
    
    batch_size = 20
    actions = []
    processed_count = 0
    processed_entities = []
    
    for entity in tqdm(matched_entities, desc="处理实体"):
        try:
            processed_data = process_single_item(entity, use_batch=False)
            
            if processed_data:
                processed_entities.append(processed_data.copy())
                
                processed_data.pop('_query', None)
                processed_data.pop('_correct_link', None)
                processed_data.pop('_match_method', None)
                
                action = {
                    "_index": FIND_INDEX_NAME,
                    "_source": processed_data
                }
                actions.append(action)
                
                if len(actions) >= batch_size:
                    helpers.bulk(es, actions, request_timeout=180)
                    processed_count += len(actions)
                    actions = []
        except Exception as e:
            print(f"处理实体失败: {e}")
            continue
    
    if actions:
        helpers.bulk(es, actions, request_timeout=180)
        processed_count += len(actions)
    
    print(f"\n✓ 索引构建完成！")
    print(f"  索引名称: {FIND_INDEX_NAME}")
    print(f"  导入实体数: {processed_count}")
    
    try:
        result = es.count(index=FIND_INDEX_NAME)
        print(f"  索引中的文档数: {result['count']}")
    except Exception as e:
        print(f"  验证索引时出错: {e}")
    
    return True, processed_entities


def cmd_build_entities(args):
    """构建实体索引命令"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    work_wyy_dir = os.path.dirname(script_dir)
    
    # 确定文件路径
    if args.wiki_file:
        wiki_file = args.wiki_file
        if not os.path.isabs(wiki_file):
            wiki_file = os.path.abspath(wiki_file)
    else:
        wiki_file = os.path.join(work_wyy_dir, 'data', 'zh_wiki_v2.jsonl')
    
    if args.find_file:
        find_file = args.find_file
        if not os.path.isabs(find_file):
            find_file = os.path.abspath(find_file)
    else:
        find_file = os.path.join(work_wyy_dir, 'data', 'find.xlsx')
    
    if not os.path.exists(wiki_file):
        print(f"错误: 找不到wiki文件: {wiki_file}")
        sys.exit(1)
    
    if not os.path.exists(find_file):
        print(f"错误: 找不到find.xlsx文件: {find_file}")
        sys.exit(1)
    
    print("=" * 70)
    print("从find.xlsx构建独立的ES索引")
    print("=" * 70)
    print(f"find.xlsx文件: {find_file}")
    print(f"wiki文件: {wiki_file}")
    print(f"目标索引: {FIND_INDEX_NAME}")
    print("=" * 70)
    
    # 读取find.xlsx
    find_pairs = read_find_excel(find_file)
    if not find_pairs:
        print("错误: find.xlsx中没有有效数据")
        sys.exit(1)
    
    # 加载wiki数据
    label_to_data, alias_to_data, link_to_data, url_title_to_data = load_wiki_data(wiki_file)
    if not label_to_data and not alias_to_data and not link_to_data and not url_title_to_data:
        print("错误: 无法加载wiki数据")
        sys.exit(1)
    
    # 匹配实体
    matched_entities, unmatched_pairs, match_methods = match_entities_from_find(
        find_pairs, label_to_data, alias_to_data, link_to_data, url_title_to_data
    )
    
    if not matched_entities:
        print("错误: 没有匹配到任何实体")
        sys.exit(1)
    
    # 构建索引
    success, processed_entities = build_entities_index(matched_entities)
    if not success:
        print("\n错误: 索引构建失败")
        sys.exit(1)
    
    # 保存未匹配的实体
    if unmatched_pairs:
        unmatched_file = os.path.join(WORK_DIR, 'trainlog', 
                                     f'unmatched_entities_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        os.makedirs(os.path.dirname(unmatched_file), exist_ok=True)
        
        unmatched_data = {
            'timestamp': datetime.now().isoformat(),
            'total_unmatched': len(unmatched_pairs),
            'unmatched_entities': [
                {
                    'query': query,
                    'link': link,
                    'normalized_link': normalize_url(link),
                    'url_title': extract_title_from_url(link)
                }
                for query, link in unmatched_pairs
            ]
        }
        
        with open(unmatched_file, 'w', encoding='utf-8') as f:
            json.dump(unmatched_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n未匹配的实体已保存到: {unmatched_file}")
    
    # 生成统计报告
    print("\n" + "=" * 70)
    print("数据统计报告")
    print("=" * 70)
    
    total_find = len(find_pairs)
    matched_count = len(matched_entities)
    unmatched_count = len(unmatched_pairs)
    match_rate = (matched_count / total_find * 100) if total_find > 0 else 0
    
    print(f"\n【匹配率统计】")
    print(f"  find.xlsx总记录数: {total_find}")
    print(f"  成功匹配数: {matched_count}")
    print(f"  未匹配数: {unmatched_count}")
    print(f"  匹配率: {match_rate:.2f}%")
    
    # 字段非空率统计
    field_stats = calculate_field_statistics(processed_entities)
    
    print(f"\n【字段非空率统计】")
    print(f"  统计实体数: {len(processed_entities)}")
    print(f"\n  字段详情:")
    print(f"  {'字段名':<30} {'非空数':<10} {'非空率':<10} {'空值数':<10}")
    print(f"  {'-'*70}")
    
    sorted_fields = sorted(field_stats.items(), key=lambda x: x[1]['non_empty_rate'], reverse=True)
    for field, stat in sorted_fields:
        print(f"  {field:<30} {stat['non_empty_count']:<10} {stat['non_empty_rate']:>6.2f}%   {stat['empty_count']:<10}")
    
    # 保存统计报告
    report_file = os.path.join(WORK_DIR, 'trainlog', 
                               f'find_index_statistics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'index_name': FIND_INDEX_NAME,
        'match_statistics': {
            'total_find_records': total_find,
            'matched_count': matched_count,
            'unmatched_count': unmatched_count,
            'match_rate': match_rate,
            'match_methods': match_methods
        },
        'field_statistics': field_stats,
        'unmatched_pairs': unmatched_pairs[:50]
    }
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n  统计报告已保存到: {report_file}")
    print("\n" + "=" * 70)
    print("✓ 完成！")
    print(f"索引 {FIND_INDEX_NAME} 已创建，包含 {matched_count} 个实体")
    print(f"匹配率: {match_rate:.2f}%")
    print("=" * 70)


# ============================================================================
# 功能2: 构建查询词索引
# ============================================================================

def create_queries_index():
    """创建查询词索引"""
    index_mapping = {
        "mappings": {
            "properties": {
                "query": {"type": "text"},
                "correct_link": {"type": "keyword"},
                "query_vector": {
                    "type": "dense_vector",
                    "dims": VECTOR_DIMS,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }
    
    if es.indices.exists(index=QUERIES_INDEX_NAME):
        print(f"删除现有索引: {QUERIES_INDEX_NAME}")
        try:
            es.indices.delete(index=QUERIES_INDEX_NAME)
            import time
            time.sleep(2)
        except Exception as e:
            print(f"删除索引失败: {e}")
            return False
    
    try:
        try:
            es.indices.create(index=QUERIES_INDEX_NAME, body=index_mapping)
        except TypeError:
            es.indices.create(index=QUERIES_INDEX_NAME, mappings=index_mapping.get("mappings", {}))
        print(f"成功创建查询词索引: {QUERIES_INDEX_NAME}, 向量维度: {VECTOR_DIMS}")
        return True
    except Exception as e:
        print(f"创建索引失败: {e}")
        return False


def build_queries_index(find_pairs):
    """构建查询词索引"""
    print(f"\n开始构建ES索引: {QUERIES_INDEX_NAME}")
    
    if not create_queries_index():
        print("创建索引失败")
        return False
    
    # 批量生成向量
    print(f"\n开始为 {len(find_pairs)} 个查询词生成向量...")
    queries = [q for q, _ in find_pairs]
    
    try:
        query_vectors = batch_generate_vectors(
            queries,
            use_finetuned=True,
            target_dim=VECTOR_DIMS,
            batch_size=32
        )
        print(f"✓ 向量生成完成")
    except Exception as e:
        print(f"批量生成向量失败: {e}")
        print("回退到单条生成...")
        query_vectors = []
        for query in tqdm(queries, desc="生成向量"):
            vec = generate_vector(query)
            query_vectors.append(vec)
    
    # 导入ES
    print(f"\n开始导入ES...")
    from elasticsearch import helpers
    
    batch_size = 50
    actions = []
    processed_count = 0
    
    for (query, correct_link), vector in zip(tqdm(find_pairs, desc="导入ES"), query_vectors):
        if vector and len(vector) == VECTOR_DIMS:
            action = {
                "_index": QUERIES_INDEX_NAME,
                "_source": {
                    "query": query,
                    "correct_link": correct_link,
                    "query_vector": vector
                }
            }
            actions.append(action)
            
            if len(actions) >= batch_size:
                helpers.bulk(es, actions, request_timeout=180)
                processed_count += len(actions)
                actions = []
        else:
            print(f"警告: 查询词 '{query}' 的向量生成失败或维度不正确")
    
    if actions:
        helpers.bulk(es, actions, request_timeout=180)
        processed_count += len(actions)
    
    print(f"\n✓ 索引构建完成！")
    print(f"  索引名称: {QUERIES_INDEX_NAME}")
    print(f"  导入查询词数: {processed_count}")
    
    try:
        result = es.count(index=QUERIES_INDEX_NAME)
        print(f"  索引中的文档数: {result['count']}")
    except Exception as e:
        print(f"  验证索引时出错: {e}")
    
    return True


def cmd_build_queries(args):
    """构建查询词索引命令"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    work_wyy_dir = os.path.dirname(script_dir)
    
    if args.find_file:
        find_file = args.find_file
        if not os.path.isabs(find_file):
            find_file = os.path.abspath(find_file)
    else:
        find_file = os.path.join(work_wyy_dir, 'data', 'find.xlsx')
    
    if not os.path.exists(find_file):
        print(f"错误: 找不到find.xlsx文件: {find_file}")
        sys.exit(1)
    
    print("=" * 70)
    print("将find.xlsx中的查询词向量化并存入ES")
    print("=" * 70)
    print(f"find.xlsx文件: {find_file}")
    print(f"目标索引: {QUERIES_INDEX_NAME}")
    print("=" * 70)
    
    find_pairs = read_find_excel(find_file)
    if not find_pairs:
        print("错误: find.xlsx中没有有效数据")
        sys.exit(1)
    
    success = build_queries_index(find_pairs)
    
    if success:
        print("\n" + "=" * 70)
        print("✓ 完成！")
        print(f"索引 {QUERIES_INDEX_NAME} 已创建，包含 {len(find_pairs)} 个查询词")
        print("=" * 70)
    else:
        print("\n错误: 索引构建失败")
        sys.exit(1)


# ============================================================================
# 功能3: 对比索引
# ============================================================================

def load_queries_from_index(index_name=QUERIES_INDEX_NAME):
    """从查询词索引中加载所有查询词"""
    print(f"从索引 {index_name} 加载查询词...")
    
    queries = []
    
    try:
        resp = es.search(
            index=index_name,
            body={"query": {"match_all": {}}, "size": 1000},
            scroll='2m'
        )
        
        scroll_id = resp.get('_scroll_id')
        hits = resp.get('hits', {}).get('hits', [])
        
        while hits:
            for hit in hits:
                source = hit.get('_source', {})
                queries.append({
                    'query': source.get('query', ''),
                    'correct_link': source.get('correct_link', ''),
                    'query_vector': source.get('query_vector', [])
                })
            
            if scroll_id:
                resp = es.scroll(scroll_id=scroll_id, scroll='2m')
                scroll_id = resp.get('_scroll_id')
                hits = resp.get('hits', {}).get('hits', [])
            else:
                break
        
        print(f"✓ 加载了 {len(queries)} 个查询词")
        return queries
    except Exception as e:
        print(f"错误: 加载查询词失败: {e}")
        return []


def search_in_index(query_vector, index_name, top_k=10):
    """在指定索引中使用向量检索"""
    if not query_vector or len(query_vector) != 1024:
        return []
    
    try:
        search_body = {
            "knn": {
                "field": "query_vector" if index_name == QUERIES_INDEX_NAME else "descriptions_zh_vector",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": top_k * 5
            },
            "size": top_k
        }
        
        if index_name == FIND_INDEX_NAME:
            vector_fields = [
                "descriptions_zh_vector", "descriptions_en_vector", "entity_words_zh_vector",
                "entity_words_en_vector", "label_vector", "label_zh_vector", "label_en_vector"
            ]
            
            all_results = {}
            
            for field in vector_fields:
                try:
                    search_body["knn"]["field"] = field
                    resp = es.search(index=index_name, body=search_body)
                    hits = resp.get("hits", {}).get("hits", [])
                    
                    for hit in hits:
                        doc_id = hit.get("_id")
                        score = hit.get("_score", 0)
                        source = hit.get("_source", {})
                        
                        if doc_id not in all_results or score > all_results[doc_id]["score"]:
                            all_results[doc_id] = {
                                "source": source,
                                "score": score,
                                "field": field
                            }
                except:
                    continue
            
            sorted_results = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)[:top_k]
            return sorted_results
        else:
            resp = es.search(index=index_name, body=search_body)
            hits = resp.get("hits", {}).get("hits", [])
            
            results = []
            for hit in hits:
                results.append({
                    "source": hit.get("_source", {}),
                    "score": hit.get("_score", 0),
                    "field": "query_vector"
                })
            return results
            
    except Exception as e:
        print(f"搜索失败: {e}")
        return []


def compare_search_results(queries, top_k=10):
    """对比两个索引的检索结果"""
    print(f"\n开始对比检索结果（top_k={top_k}）...")
    print("=" * 70)
    
    results = []
    queries_index_stats = {
        'total': 0, 'found_correct': 0, 'correct_at_1': 0, 'correct_at_5': 0, 'correct_at_10': 0
    }
    entities_index_stats = {
        'total': 0, 'found_correct': 0, 'correct_at_1': 0, 'correct_at_5': 0, 'correct_at_10': 0
    }
    
    for query_data in tqdm(queries, desc="检索对比"):
        query = query_data['query']
        correct_link = query_data['correct_link']
        query_vector = query_data.get('query_vector', [])
        
        if not query_vector:
            continue
        
        queries_results = search_in_index(query_vector, QUERIES_INDEX_NAME, top_k=top_k)
        entities_results = search_in_index(query_vector, FIND_INDEX_NAME, top_k=top_k)
        
        normalized_correct_link = normalize_url(correct_link)
        
        # 统计查询词索引
        queries_index_stats['total'] += 1
        queries_rank = None
        for i, result in enumerate(queries_results, 1):
            result_link = result['source'].get('correct_link', '')
            if normalize_url(result_link) == normalized_correct_link:
                queries_rank = i
                queries_index_stats['found_correct'] += 1
                if i <= 1:
                    queries_index_stats['correct_at_1'] += 1
                if i <= 5:
                    queries_index_stats['correct_at_5'] += 1
                if i <= 10:
                    queries_index_stats['correct_at_10'] += 1
                break
        
        # 统计实体索引
        entities_index_stats['total'] += 1
        entities_rank = None
        for i, result in enumerate(entities_results, 1):
            result_link = result['source'].get('link', '')
            if normalize_url(result_link) == normalized_correct_link:
                entities_rank = i
                entities_index_stats['found_correct'] += 1
                if i <= 1:
                    entities_index_stats['correct_at_1'] += 1
                if i <= 5:
                    entities_index_stats['correct_at_5'] += 1
                if i <= 10:
                    entities_index_stats['correct_at_10'] += 1
                break
        
        results.append({
            'query': query,
            'correct_link': correct_link,
            'queries_index': {
                'rank': queries_rank,
                'top_results': [
                    {
                        'query': r['source'].get('query', ''),
                        'link': r['source'].get('correct_link', ''),
                        'score': r['score']
                    }
                    for r in queries_results[:3]
                ]
            },
            'entities_index': {
                'rank': entities_rank,
                'top_results': [
                    {
                        'label': r['source'].get('label', ''),
                        'link': r['source'].get('link', ''),
                        'score': r['score'],
                        'field': r.get('field', '')
                    }
                    for r in entities_results[:3]
                ]
            }
        })
    
    return results, queries_index_stats, entities_index_stats


def cmd_compare(args):
    """对比索引命令"""
    print("=" * 70)
    print("对比两个find索引的检索效果")
    print("=" * 70)
    print(f"查询词索引: {QUERIES_INDEX_NAME}")
    print(f"实体索引: {FIND_INDEX_NAME}")
    print(f"Top-K: {args.top_k}")
    print("=" * 70)
    
    # 检查索引是否存在
    if not es.indices.exists(index=QUERIES_INDEX_NAME):
        print(f"错误: 索引 {QUERIES_INDEX_NAME} 不存在")
        print(f"  请先运行: python find_index_manager.py build-queries")
        sys.exit(1)
    
    if not es.indices.exists(index=FIND_INDEX_NAME):
        print(f"错误: 索引 {FIND_INDEX_NAME} 不存在")
        print(f"  请先运行: python find_index_manager.py build-entities")
        sys.exit(1)
    
    # 加载查询词
    queries = load_queries_from_index(QUERIES_INDEX_NAME)
    if not queries:
        print("错误: 没有加载到查询词")
        sys.exit(1)
    
    # 对比检索结果
    results, queries_stats, entities_stats = compare_search_results(queries, top_k=args.top_k)
    
    # 生成报告
    print("\n" + "=" * 70)
    print("检索对比结果")
    print("=" * 70)
    
    print(f"\n【查询词索引 ({QUERIES_INDEX_NAME})】")
    print(f"  总查询数: {queries_stats['total']}")
    print(f"  找到正确结果: {queries_stats['found_correct']} ({queries_stats['found_correct']/queries_stats['total']*100:.1f}%)")
    print(f"  Hit@1: {queries_stats['correct_at_1']} ({queries_stats['correct_at_1']/queries_stats['total']*100:.1f}%)")
    print(f"  Hit@5: {queries_stats['correct_at_5']} ({queries_stats['correct_at_5']/queries_stats['total']*100:.1f}%)")
    print(f"  Hit@10: {queries_stats['correct_at_10']} ({queries_stats['correct_at_10']/queries_stats['total']*100:.1f}%)")
    
    print(f"\n【实体索引 ({FIND_INDEX_NAME})】")
    print(f"  总查询数: {entities_stats['total']}")
    print(f"  找到正确结果: {entities_stats['found_correct']} ({entities_stats['found_correct']/entities_stats['total']*100:.1f}%)")
    print(f"  Hit@1: {entities_stats['correct_at_1']} ({entities_stats['correct_at_1']/entities_stats['total']*100:.1f}%)")
    print(f"  Hit@5: {entities_stats['correct_at_5']} ({entities_stats['correct_at_5']/entities_stats['total']*100:.1f}%)")
    print(f"  Hit@10: {entities_stats['correct_at_10']} ({entities_stats['correct_at_10']/entities_stats['total']*100:.1f}%)")
    
    # 计算MRR
    queries_mrr = 0
    entities_mrr = 0
    
    for result in results:
        if result['queries_index']['rank']:
            queries_mrr += 1.0 / result['queries_index']['rank']
        if result['entities_index']['rank']:
            entities_mrr += 1.0 / result['entities_index']['rank']
    
    queries_mrr /= len(results) if results else 1
    entities_mrr /= len(results) if results else 1
    
    print(f"\n【MRR (Mean Reciprocal Rank)】")
    print(f"  查询词索引 MRR: {queries_mrr:.4f}")
    print(f"  实体索引 MRR: {entities_mrr:.4f}")
    
    # 保存结果
    if args.output:
        output_file = args.output
    else:
        output_file = os.path.join(WORK_DIR, 'trainlog', 
                                  f'compare_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'queries_index': QUERIES_INDEX_NAME,
        'entities_index': FIND_INDEX_NAME,
        'top_k': args.top_k,
        'statistics': {
            'queries_index': queries_stats,
            'entities_index': entities_stats,
            'queries_mrr': queries_mrr,
            'entities_mrr': entities_mrr
        },
        'detailed_results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存到: {output_file}")
    print("=" * 70)


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Find索引管理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
功能说明:
  build-entities  从find.xlsx构建实体索引（data2_find）
  build-queries   从find.xlsx构建查询词索引（data2_find_queries）
  compare         对比两个索引的检索效果

示例:
  python find_index_manager.py build-entities
  python find_index_manager.py build-queries
  python find_index_manager.py compare --top-k 20
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # build-entities 子命令
    parser_entities = subparsers.add_parser('build-entities', help='构建实体索引')
    parser_entities.add_argument('--find-file', type=str, default=None,
                                 help='find.xlsx文件路径（默认: work_wyy/data/find.xlsx）')
    parser_entities.add_argument('--wiki-file', type=str, default=None,
                                 help='zh_wiki_v2.jsonl文件路径（默认: work_wyy/data/zh_wiki_v2.jsonl）')
    
    # build-queries 子命令
    parser_queries = subparsers.add_parser('build-queries', help='构建查询词索引')
    parser_queries.add_argument('--find-file', type=str, default=None,
                                help='find.xlsx文件路径（默认: work_wyy/data/find.xlsx）')
    
    # compare 子命令
    parser_compare = subparsers.add_parser('compare', help='对比两个索引的检索效果')
    parser_compare.add_argument('--top-k', type=int, default=10,
                                help='检索返回的top_k数量（默认: 10）')
    parser_compare.add_argument('--output', type=str, default=None,
                                help='输出结果文件路径（默认: trainlog/compare_results_*.json）')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'build-entities':
        cmd_build_entities(args)
    elif args.command == 'build-queries':
        cmd_build_queries(args)
    elif args.command == 'compare':
        cmd_compare(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
