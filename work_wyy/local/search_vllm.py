#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量检索与LLM重排序系统

支持5种检索方案对比：
1. vector_only: 纯向量检索
2. es_text_only: 纯ES文本搜索
3. llm_only: 纯LLM判断
4. vector_with_llm_always: 向量+LLM（始终重排序）
5. vector_with_llm: 向量+LLM（智能混合模式，推荐）

详细说明请参考：检索方案对比说明.md
"""

import torch
from transformers import BertTokenizer, BertModel
from zhipuai import ZhipuAI
import numpy as np
import re
from urllib.parse import unquote
# 导入配置
try:
    from config import (
        ES_INDEX_NAME,
        TRAINLOG_DIR,
        ZHIPUAI_API_KEY,
        VECTOR_SEARCH_TOP_K,
        LLM_RERANK_TOP_K,
        WORK_DIR
    )
    from es_client import es
except ImportError:
    # 如果无法导入配置，使用默认值
    ES_INDEX_NAME = 'data2'
    TRAINLOG_DIR = None
    ZHIPUAI_API_KEY = "1a2a485fe1fc4bd5aa0d965bf452c8c8.se8RZdT8cH8skEDo"
    VECTOR_SEARCH_TOP_K = 30
    LLM_RERANK_TOP_K = 30
    WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    from es_client import es

import logging
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import json
import os
import sys
from datetime import datetime

# 获取脚本所在目录（local）
script_dir_search = os.path.dirname(os.path.abspath(__file__))

# 创建 trainlog 文件夹（如果不存在）
if TRAINLOG_DIR:
    trainlog_dir = TRAINLOG_DIR
else:
    trainlog_dir = os.path.join(os.path.dirname(script_dir_search), 'trainlog')
os.makedirs(trainlog_dir, exist_ok=True)

# 创建日志文件名（保存到 trainlog 文件夹）
log_filename = os.path.join(trainlog_dir, f'vector_search_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
console_log_file = os.path.join(trainlog_dir, f'search_vllm_console_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 创建Tee类，同时输出到控制台和文件
class Tee:
    """同时将输出写入文件和控制台"""
    def __init__(self, file_path, mode='a', encoding='utf-8'):
        self.file = open(file_path, mode, encoding=encoding)
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        
    def write(self, text):
        self.file.write(text)
        self.file.flush()  # 立即刷新到文件
        self.stdout.write(text)
        self.stdout.flush()
        
    def flush(self):
        self.file.flush()
        self.stdout.flush()
        
    def close(self):
        if self.file:
            self.file.close()

# 重定向stdout和stderr到文件和控制台
tee = Tee(console_log_file, mode='w')
sys.stdout = tee
sys.stderr = tee

logger.info(f"所有控制台输出将同时保存到: {console_log_file}")
logger.info(f"日志信息保存到: {log_filename}")

# 关闭ES和urllib3的HTTP请求日志（只显示错误）
logging.getLogger('elasticsearch').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('urllib3.connectionpool').setLevel(logging.ERROR)

# 使用统一的向量生成模块（支持微调后的模型）
try:
    from vector_model import load_vector_model, generate_vector as _generate_vector_module, batch_generate_vectors
    model, tokenizer, device = load_vector_model(use_finetuned=True)
    print(f"✓ 向量生成模型加载成功 (使用微调后的模型，设备: {device})")
except Exception as e:
    print(f"警告: 向量生成模型加载失败 ({e})，尝试使用基础模型")
    try:
        model_name = './model/chinese-roberta-wwm-ext-large'
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        model.eval()
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model = model.to(device)
        print(f"✓ 基础模型加载成功 (设备: {device})")
    except Exception as e2:
        print(f"错误: 基础模型也加载失败 ({e2})，向量生成功能将不可用")
        model = None
        tokenizer = None
        device = torch.device("cpu")

# 智谱AI API客户端（使用配置文件中的API Key）
client = ZhipuAI(api_key=ZHIPUAI_API_KEY)

# 向量缓存字典
_vector_cache = {}
_cache_max_size = 1000

# 批量向量生成时的默认 batch 大小（仅用于评测加速）
_batch_size_for_eval = 32


def preprocess_query(query):
    """
    预处理查询文本
    - 清理换行符、多余空格
    - 统一格式
    """
    if not query:
        return ""

    # 转换为字符串
    query = str(query)

    # 替换换行符为空格
    query = query.replace('\n', ' ').replace('\r', ' ')

    # 清理多余空格
    query = ' '.join(query.split())

    # 清理首尾空格
    query = query.strip()

    return query


def _batch_generate_vectors_internal(texts, use_cache=True, batch_size=None):
    """
    批量生成 1024 维向量（使用统一的向量生成模块，支持微调后的模型）

    - 使用统一的向量生成模块
    - 自动处理 768/1024 维模型到 1024 维
    - 做 L2 归一化
    """
    if batch_size is None:
        batch_size = _batch_size_for_eval

    results = [None] * len(texts)
    to_compute_indices = []
    to_compute_texts = []

    for i, t in enumerate(texts):
        if not t or not str(t).strip():
            continue
        # 预处理文本
        t = preprocess_query(str(t))
        if not t:
            continue
        if use_cache and t in _vector_cache:
            results[i] = _vector_cache[t]
        else:
            to_compute_indices.append(i)
            to_compute_texts.append(t)

    if not to_compute_texts:
        return results

    # 使用统一的批量向量生成模块
    try:
        batch_vectors = batch_generate_vectors(
            to_compute_texts,
            use_finetuned=True,
            target_dim=1024,
            batch_size=batch_size
        )
        
        for j, vec in enumerate(batch_vectors):
            if vec is not None:
                global_idx = to_compute_indices[j]
                results[global_idx] = vec
                if use_cache:
                    _vector_cache[str(texts[global_idx])] = vec
    except:
        # 回退到原始方法
        if model is None or tokenizer is None:
            return results
        target_dim = 1024
        for start in range(0, len(to_compute_texts), batch_size):
            end = start + batch_size
            batch_texts = to_compute_texts[start:end]
            if not batch_texts:
                continue
            inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            for j in range(vectors.shape[0]):
                v = vectors[j]
                vec_dim = len(v)
                if vec_dim < target_dim:
                    v = np.pad(v, (0, target_dim - vec_dim), 'constant', constant_values=0)
                else:
                    v = v[:target_dim]
                norm = np.linalg.norm(v)
                if norm > 0:
                    v = v / norm
                v_list = v.astype(float).tolist()
                global_idx = to_compute_indices[start + j]
                results[global_idx] = v_list
                if use_cache:
                    _vector_cache[str(texts[global_idx])] = v_list

    return results


def generate_vector(text, use_cache=True):
    """
    生成文本向量（使用统一的向量生成模块，支持微调后的模型）

    Args:
        text: 输入文本
        use_cache: 是否使用缓存（默认True）

    Returns:
        1024维向量列表，已L2归一化
    """
    # 预处理文本
    text = preprocess_query(text)

    if not text:
        return None

    # 使用缓存
    if use_cache:
        cache_key = text
        if cache_key in _vector_cache:
            return _vector_cache[cache_key]

    # 使用统一的向量生成模块
    try:
        vector_list = _generate_vector_module(text, use_finetuned=True, target_dim=1024)
    except:
        # 如果模块不可用，回退到原始方法
        if model is None or tokenizer is None:
            return None
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        vector = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        vector_dim = len(vector)
        target_dim = 1024
        if vector_dim < target_dim:
            vector = np.pad(vector, (0, target_dim - vector_dim), 'constant', constant_values=0)
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
        elif vector_dim > target_dim:
            vector = vector[:target_dim]
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
        vector_list = vector.tolist()

    if vector_list is None:
        return None

    # 添加到缓存
    if use_cache:
        if len(_vector_cache) >= _cache_max_size:
            first_key = next(iter(_vector_cache))
            del _vector_cache[first_key]
        _vector_cache[cache_key] = vector_list

    return vector_list


def vector_search(query_text, top_k=20, query_vector=None):
    """
    单独的向量检索（同时检索所有7个向量字段）

    Args:
        query_text: 查询文本
        top_k: 返回结果数量
        query_vector: 预计算的查询向量（可选）

    Returns:
        检索结果列表
    """
    # 预处理查询
    query_text = preprocess_query(query_text)

    # 生成查询向量
    if query_vector is None and model is not None and tokenizer is not None:
        try:
            query_vector = generate_vector(query_text, use_cache=True)
        except Exception:
            query_vector = None

    if query_vector is None:
        return []

    # 确定要搜索的向量字段（同时检索所有向量字段）
    vector_fields = [
        ("descriptions_zh_vector", "zh", "desc"),
        ("descriptions_en_vector", "en", "desc"),
        ("entity_words_zh_vector", "zh", "entity"),  # 中文实体词向量
        ("entity_words_en_vector", "en", "entity"),  # 英文实体词向量
        ("label_vector", "mixed", "label"),
        ("label_zh_vector", "zh", "label"),
        ("label_en_vector", "en", "label")
    ]

    # 同时对多个向量字段做检索，然后融合结果
    index_names = [ES_INDEX_NAME]  # 使用配置文件中的索引名称
    merged_hits = {}

    for index_name in index_names:
        try:
            if not es.indices.exists(index=index_name):
                continue
        except Exception:
            continue

        # 针对当前索引，尝试所有向量字段
        for field_name, lang_tag, field_type in vector_fields:
            knn_query = {
                "field": field_name,
                "query_vector": query_vector,
                "k": top_k * 2,  # 增加候选数量
                "num_candidates": top_k * 5  # 增加候选数量
            }
            search_body = {
                "knn": knn_query,
                "size": top_k * 2  # 增加返回数量
            }
            try:
                resp = es.search(index=index_name, body=search_body)
                hits = resp.get("hits", {}).get("hits", []) or []
                for hit in hits:
                    doc_id = hit.get("_id")
                    if not doc_id:
                        continue
                    key = (index_name, doc_id)
                    score = float(hit.get("_score", 0.0) or 0.0)
                    # 如果同一个文档被多个字段命中，保留最高分
                    if key not in merged_hits or score > merged_hits[key]["score"]:
                        merged_hits[key] = {
                            "source": hit.get("_source", {}),
                            "score": score,
                            "lang": lang_tag,
                            "field_type": field_type
                        }
            except Exception:
                continue

    if not merged_hits:
        return []

    # 按得分排序，取前 top_k
    sorted_items = sorted(merged_hits.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    results = []
    for item in sorted_items:
        source = item["source"]
        result = {
            "label": source.get("label", ""),
            "aliases_zh": source.get("aliases_zh", []),
            "aliases_en": source.get("aliases_en", []),
            "descriptions_zh": source.get("descriptions_zh", ""),
            "descriptions_en": source.get("descriptions_en", ""),
            "link": source.get("link", ""),
            "_score": item["score"],
            "_lang": item["lang"],
            "_field_type": item["field_type"]
        }
        results.append(result)

    return results


def get_alias_and_definition(mention):
    """获取实体的别名、定义和详细描述（中英文各一版）"""
    # 预处理查询
    mention = preprocess_query(mention)

    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=[
            {
                "role": "user",
                "content": (
                    f"你现在是军事领域专家，需要参照以下例子给出提及对应的别名、定义和详细描述（中英文各一版）。\n"
                    f"例子：\n"
                    f"提及：Steyr HS .50\n"
                    f"中文别名：斯泰尔HS .50狙击步枪\n"
                    f"英文别名：Steyr HS .50 sniper rifle\n"
                    f"中文定义：斯泰尔HS .50（Steyr HS.50）是由奥地利斯泰尔-曼利夏公司研制的一款手动枪机式反器材狙击步枪。\n"
                    f"英文定义：The Steyr HS .50 (Steyr HS.50) is a manually operated anti-materiel sniper rifle developed by Steyr Mannlicher of Austria.\n"
                    f"中文详细描述：斯泰尔HS .50是一款大口径反器材狙击步枪，采用手动枪机操作方式，发射12.7×99毫米（.50 BMG）弹药。该枪具有出色的远距离精确射击能力，主要用于反器材作战和远程狙击任务。\n"
                    f"英文详细描述：The Steyr HS .50 is a large-caliber anti-materiel sniper rifle with manual bolt action, chambered for 12.7×99mm (.50 BMG) ammunition. It features excellent long-range precision shooting capabilities and is primarily used for anti-materiel operations and long-range sniper missions.\n\n"
                    f"输入提及：{mention}\n\n"
                    f"请按照以下格式直接返回所需内容，不要解释或附加内容：\n"
                    f"标签：{mention}\n"
                    f"中文别名：\n"
                    f"英文别名：\n"
                    f"中文定义：\n"
                    f"英文定义：\n"
                    f"中文详细描述：\n"
                    f"英文详细描述："
                )
            }
        ],
    )
    response_content = response.choices[0].message.content.strip()

    if not response_content:
        raise ValueError(f"No response content for mention '{mention}'")

    return response_content


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
                    try:
                        decoded_title = unquote(title)
                    except:
                        decoded_title = title
                    return decoded_title
        except Exception:
            pass

    return url


def clean_link(link):
    """清理链接，移除空白字符和常见前缀"""
    if not link:
        return ""
    link = str(link).strip()
    link = re.sub(r'^\d+[\.\)]\s*', '', link)
    link = re.sub(r'^选项\d+[：:]\s*', '', link)
    link = re.sub(r'^link[：:]\s*', '', link, flags=re.IGNORECASE)
    return link.strip()


def ensure_links_match(sorted_links, original_links):
    """确保排序后的链接与原始链接一致，支持模糊匹配"""
    cleaned_sorted = [clean_link(link) for link in sorted_links]
    original_links_set = set(original_links)

    cleaned_to_original = {}
    for orig_link in original_links:
        cleaned = clean_link(orig_link)
        cleaned_to_original[cleaned] = orig_link

    matched_links = []
    used_original_links = set()

    for cleaned_link in cleaned_sorted:
        matched = False
        # 精确匹配
        if cleaned_link in cleaned_to_original:
            orig_link = cleaned_to_original[cleaned_link]
            if orig_link not in used_original_links:
                matched_links.append(orig_link)
                used_original_links.add(orig_link)
                matched = True

        # URL归一化匹配
        if not matched:
            normalized_link = normalize_url(cleaned_link)
            for orig_link in original_links:
                if orig_link not in used_original_links:
                    orig_cleaned = clean_link(orig_link)
                    orig_normalized = normalize_url(orig_cleaned)
                    if normalized_link == orig_normalized:
                        matched_links.append(orig_link)
                        used_original_links.add(orig_link)
                        matched = True
                        break

        # 模糊匹配
        if not matched:
            for orig_link in original_links:
                if orig_link not in used_original_links:
                    orig_cleaned = clean_link(orig_link)
                    if cleaned_link in orig_cleaned or orig_cleaned in cleaned_link:
                        matched_links.append(orig_link)
                        used_original_links.add(orig_link)
                        matched = True
                        break

        # 归一化后的模糊匹配
        if not matched:
            normalized_link = normalize_url(cleaned_link)
            for orig_link in original_links:
                if orig_link not in used_original_links:
                    orig_normalized = normalize_url(clean_link(orig_link))
                    if normalized_link in orig_normalized or orig_normalized in normalized_link:
                        matched_links.append(orig_link)
                        used_original_links.add(orig_link)
                        matched = True
                        break

    # 添加未匹配的原始链接
    for orig_link in original_links:
        if orig_link not in used_original_links:
            matched_links.append(orig_link)

    return matched_links


def llm_search_direct_from_es(query, top_k=None):
    """
    直接从ES获取候选，使用LLM判断匹配度（不依赖向量检索结果）
    
    Args:
        query: 查询文本
        top_k: 从ES获取的候选数量
    
    Returns:
        排序后的链接列表
    """
    # 预处理查询
    query = preprocess_query(query)
    
    if not query:
        return []
    
    # 如果top_k为None，使用配置文件中的默认值
    if top_k is None:
        top_k = LLM_RERANK_TOP_K
    
    # 从ES获取候选（使用文本搜索，不依赖向量）
    try:
        # 使用match查询获取候选
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["label^3", "aliases_zh^2", "aliases_en^2", "descriptions_zh", "descriptions_en"],
                    "type": "best_fields",
                    "operator": "or"
                }
            },
            "size": top_k
        }
        
        resp = es.search(index=ES_INDEX_NAME, body=search_body)
        hits = resp.get("hits", {}).get("hits", []) or []
        
        if not hits:
            logger.warning(f"ES文本搜索未找到候选 for query '{query}'")
            return []
        
        # 构建结果列表
        results = []
        for hit in hits:
            source = hit.get("_source", {})
            result = {
                "label": source.get("label", ""),
                "aliases_zh": source.get("aliases_zh", []),
                "aliases_en": source.get("aliases_en", []),
                "descriptions_zh": source.get("descriptions_zh", ""),
                "descriptions_en": source.get("descriptions_en", ""),
                "link": source.get("link", ""),
                "_score": hit.get("_score", 0.0)
            }
            results.append(result)
        
        logger.info(f"从ES获取 {len(results)} 个候选（文本搜索）")
        
    except Exception as e:
        logger.error(f"ES文本搜索失败: {e}")
        return []
    
    # 使用LLM判断匹配度
    return generate_prompt_and_sort_with_description(query, results, use_vector_results=False)


def generate_prompt_and_sort_with_description(mention, results, use_vector_results=True):
    """
    使用LLM重排序，重点使用完整的描述信息进行匹配

    Args:
        mention: 查询提及
        results: 向量检索结果列表（如果use_vector_results=False，则results来自ES文本搜索）
        use_vector_results: 是否使用向量检索结果（False时表示直接从ES获取的候选）

    Returns:
        排序后的链接列表
    """
    # 预处理查询
    mention = preprocess_query(mention)

    input_label = mention
    response_content = ""

    try:
        response_content = get_alias_and_definition(mention)

        # 安全提取字段内容
        def safe_extract(content, field_name, default=""):
            # 尝试中文冒号
            if f"{field_name}：" in content:
                parts = content.split(f"{field_name}：", 1)
                if len(parts) > 1:
                    # 找到下一个字段标记作为结束位置
                    next_markers = ["英文别名", "中文别名", "中文定义", "英文定义", "中文详细描述", "英文详细描述", "标签", "\n\n"]
                    end_pos = len(parts[1])
                    for marker in next_markers:
                        marker_idx = parts[1].find(marker)
                        if marker_idx != -1 and marker_idx < end_pos:
                            end_pos = marker_idx
                    value = parts[1][:end_pos].strip()
                    return value if value else default
            # 尝试英文冒号
            elif f"{field_name}:" in content:
                parts = content.split(f"{field_name}:", 1)
                if len(parts) > 1:
                    next_markers = ["英文别名", "中文别名", "中文定义", "英文定义", "中文详细描述", "英文详细描述", "标签", "\n\n"]
                    end_pos = len(parts[1])
                    for marker in next_markers:
                        marker_idx = parts[1].find(marker)
                        if marker_idx != -1 and marker_idx < end_pos:
                            end_pos = marker_idx
                    value = parts[1][:end_pos].strip()
                    return value if value else default
            return default

        input_aliases_zh = safe_extract(response_content, "中文别名", "")
        input_aliases_en = safe_extract(response_content, "英文别名", "")
        input_definition_zh = safe_extract(response_content, "中文定义", "")
        input_definition_en = safe_extract(response_content, "英文定义", "")
        input_description_zh = safe_extract(response_content, "中文详细描述", "")
        input_description_en = safe_extract(response_content, "英文详细描述", "")

        if not input_aliases_zh and not input_aliases_en and not input_definition_zh and not input_definition_en and not input_description_zh and not input_description_en:
            raise ValueError("无法从LLM响应中提取任何有效字段")

    except (ValueError, IndexError, Exception) as e:
        logger.warning(f"LLM解析失败 for mention '{mention}': {e}")
        return [result['link'] for result in results]

    # 构建选项列表，确保包含完整的描述信息（中英文）
    options = []
    original_links = []

    for idx, result in enumerate(results, start=1):
        # 获取完整的描述信息（中英文）
        descriptions_zh = result.get('descriptions_zh', '')
        if not descriptions_zh:
            descriptions_zh = "（无描述信息）"
        descriptions_en = result.get('descriptions_en', '')
        if not descriptions_en:
            descriptions_en = "（无描述信息）"

        # 构建选项，重点展示描述信息（中英文）
        option = (
            f"选项{idx}：\n"
            f"标签(label): {result.get('label', '')}\n"
            f"中文别名(aliases_zh): {', '.join(result.get('aliases_zh', [])) if result.get('aliases_zh') else '无'}\n"
            f"英文别名(aliases_en): {', '.join(result.get('aliases_en', [])) if result.get('aliases_en') else '无'}\n"
            f"中文完整描述(descriptions_zh): {descriptions_zh}\n"
            f"英文完整描述(descriptions_en): {descriptions_en}\n"
            f"链接(link): {result.get('link', '')}\n"
        )
        options.append(option)
        original_links.append(result.get('link', ''))

    # 构建prompt，根据是否使用向量检索结果调整提示
    if use_vector_results:
        context_note = "（这些候选来自向量相似度检索）"
        match_instruction = "请根据输入信息与选项的匹配度（特别关注中英文描述信息的匹配度），从高到低严格返回所有候选的link值。"
    else:
        context_note = "（这些候选来自ES文本搜索，请直接根据查询词和描述的语义匹配度进行判断，不考虑向量相似度）"
        match_instruction = "请直接比较查询词与每个选项的描述信息的语义匹配度，从高到低严格返回所有候选的link值。不要考虑向量相似度分数，只根据文本语义进行判断。"
    
    # 构建prompt，明确强调要使用描述信息进行匹配（中英文）
    prompt = (
        f"现在你是军事领域专家，需要根据输入查询词与选项列表中每个候选的描述信息进行匹配度判断，然后从高到低排序。{context_note}\n\n"
        f"【重要提示】请重点参考每个选项的完整描述信息（包括中文描述descriptions_zh和英文描述descriptions_en）进行匹配度判断，"
        f"描述信息包含了实体的详细特征和定义，比标签和别名更能准确反映实体的本质特征。在判断匹配度时，描述信息的权重应该高于标签和别名。\n\n"
        f"【匹配原则】\n"
        f"1. 直接比较查询词（{input_label}）与每个选项的描述信息（descriptions_zh和descriptions_en）的语义匹配度\n"
        f"2. 如果查询词中的关键词在描述中出现，或者描述中提到的实体特征与查询词相关，则匹配度较高\n"
        f"3. 不要考虑向量相似度分数，只根据文本语义进行判断\n\n"
        f"输入查询词：{input_label}\n"
        f"查询词相关信息：\n"
        f"  中文别名：{input_aliases_zh if input_aliases_zh else '无'}\n"
        f"  英文别名：{input_aliases_en if input_aliases_en else '无'}\n"
        f"  中文定义：{input_definition_zh if input_definition_zh else '无'}\n"
        f"  英文定义：{input_definition_en if input_definition_en else '无'}\n"
        f"  中文详细描述：{input_description_zh if input_description_zh else '无'}\n"
        f"  英文详细描述：{input_description_en if input_description_en else '无'}\n\n"
        f"选项列表：\n"
        f"{''.join(options)}\n\n"
        f"{match_instruction}\n"
        f"【重要要求】\n"
        f"1. 必须返回所有{len(options)}个选项的link值，不能有缺失\n"
        f"2. 每个link值只能出现一次，不能有重复\n"
        f"3. 只返回link值，每行一个，不要解释或附加内容\n"
        f"4. 确保返回的link值完全匹配选项列表中的link值"
    )

    try:
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = response.choices[0].message.content.strip()
        sorted_links_raw = [line.strip() for line in response_text.split("\n") if line.strip()]

        # 去重：保留第一次出现的链接
        seen = set()
        sorted_links_dedup = []
        for link in sorted_links_raw:
            link_normalized = normalize_url(clean_link(link))
            if link_normalized not in seen:
                seen.add(link_normalized)
                sorted_links_dedup.append(link)

        sorted_links = ensure_links_match(sorted_links_dedup, original_links)
        return sorted_links
    except Exception as e:
        logger.error(f"LLM排序失败 for mention '{mention}': {e}")
        return original_links


def test_vector_search_only(query, top_k=10):
    """
    测试单独的向量检索（不使用LLM重排序）

    Args:
        query: 查询文本
        top_k: 返回结果数量
    """
    print(f"\n{'=' * 60}")
    print(f"测试向量检索（无LLM重排序）")
    print(f"{'=' * 60}")
    print(f"查询: '{query}'")
    print(f"-" * 60)

    try:
        results = vector_search(query, top_k=top_k)
        print(f"找到 {len(results)} 个候选实体\n")

        for i, result in enumerate(results, 1):
            print(f"{i}. {result.get('label', 'N/A')}")
            print(f"   相似度分数: {result.get('_score', 'N/A'):.4f}")
            print(f"   链接: {result.get('link', 'N/A')}")
            print(f"   描述: {result.get('descriptions_zh', '')[:100]}...")
            print()

        return results
    except Exception as e:
        print(f"向量检索失败: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_vector_search_with_llm(query, top_k=10):
    """
    测试向量检索 + LLM重排序（使用完整描述信息）

    Args:
        query: 查询文本
        top_k: 返回结果数量
    """
    print(f"\n{'=' * 60}")
    print(f"测试向量检索 + LLM重排序（使用完整描述信息）")
    print(f"{'=' * 60}")
    print(f"查询: '{query}'")
    print(f"-" * 60)

    try:
        # 1. 向量检索
        print("步骤1: 执行向量检索...")
        results = vector_search(query, top_k=top_k)
        print(f"找到 {len(results)} 个候选实体\n")

        # 显示原始向量检索结果
        print("向量检索原始结果（按相似度排序）:")
        for i, result in enumerate(results[:5], 1):
            print(f"  {i}. {result.get('label', 'N/A')} (分数: {result.get('_score', 0):.4f})")
        print()

        # 2. LLM重排序
        print("步骤2: 使用LLM重排序（重点参考描述信息）...")
        sorted_links = generate_prompt_and_sort_with_description(query, results)

        # 显示重排序后的结果
        print(f"\nLLM重排序后的结果:")
        for i, link in enumerate(sorted_links[:5], 1):
            # 找到对应的结果信息
            result_info = next((r for r in results if r.get('link') == link), None)
            if result_info:
                print(f"  {i}. {result_info.get('label', 'N/A')}")
                print(f"     链接: {link}")
                print(f"     描述: {result_info.get('descriptions_zh', '')[:100]}...")
            else:
                print(f"  {i}. {link}")
        print()

        return sorted_links
    except Exception as e:
        print(f"向量检索+LLM重排序失败: {e}")
        import traceback
        traceback.print_exc()
        return []


def read_excel(file_path):
    """读取Excel测试集文件"""
    df = pd.read_excel(file_path, header=None)
    queries = df[0].tolist()
    correct_links = df[1].tolist()
    return queries, correct_links


def calculate_metrics(queries, correct_links, search_mode="vector_only"):
    """
    计算评估指标

    Args:
        queries: 查询列表
        correct_links: 正确链接列表
        search_mode: 检索模式，支持以下模式：
            - "vector_only": 方案1 - 纯向量检索（无LLM）
            - "es_text_only": 方案2 - 纯ES文本搜索（不使用向量，不使用LLM）
            - "llm_only": 方案3 - 纯LLM判断（直接从ES获取候选，用LLM判断）
            - "vector_with_llm_always": 方案4 - 始终使用LLM重排序向量结果
            - "vector_with_llm": 方案5 - 向量检索+LLM（智能混合模式）
    """
    mrr = 0
    hit_at_1 = 0
    hit_at_5 = 0
    hit_at_10 = 0
    total_processed = 0

    # 用于存储详细结果的列表
    detailed_results = []

    # 预先批量生成向量，加速 GPU 计算
    precomputed_vectors = None
    precomputed_map = {}
    if model is not None and tokenizer is not None:
        try:
            precomputed_vectors = _batch_generate_vectors_internal(queries, use_cache=True)
            precomputed_map = {
                str(q): vec for q, vec in zip(queries, precomputed_vectors) if vec is not None
            }
            logger.info(f"[{search_mode}] 批量预生成向量完成，可用向量数: {len(precomputed_map)}")
        except Exception as e:
            logger.warning(f"[{search_mode}] 批量预生成向量失败，回退到单条生成: {e}")
            precomputed_map = {}

    def process_query(query, correct_link):
        try:
            # 根据不同模式选择不同的搜索函数
            if search_mode == "vector_only":
                # 方案1: 纯向量检索，不使用 LLM 重排序，直接按 ES 返回顺序（按相似度分数排序）
                pre_vec = precomputed_map.get(str(query))
                results = vector_search(query, top_k=30, query_vector=pre_vec)  # 增加top_k到30
                # 直接按 ES 返回的顺序（已经按相似度分数排序）
                sorted_links = [r.get("link", "") for r in results]
                
            elif search_mode == "es_text_only":
                # 方案2: 纯ES文本搜索（不使用向量，不使用LLM）
                try:
                    search_body = {
                        "query": {
                            "multi_match": {
                                "query": query,
                                "fields": ["label^3", "aliases_zh^2", "aliases_en^2", "descriptions_zh", "descriptions_en"],
                                "type": "best_fields",
                                "operator": "or"
                            }
                        },
                        "size": 30
                    }
                    resp = es.search(index=ES_INDEX_NAME, body=search_body)
                    hits = resp.get("hits", {}).get("hits", []) or []
                    sorted_links = [hit.get("_source", {}).get("link", "") for hit in hits]
                except Exception as e:
                    logger.error(f"ES文本搜索失败: {e}")
                    sorted_links = []
                    
            elif search_mode == "llm_only":
                # 方案3: 纯LLM判断（直接从ES获取候选，用LLM判断，不进行向量检索）
                sorted_links = llm_search_direct_from_es(query, top_k=30)
                
            elif search_mode == "vector_with_llm_always":
                # 方案4: 始终使用LLM重排序向量结果（不判断前10是否有hit）
                pre_vec = precomputed_map.get(str(query))
                results = vector_search(query, top_k=30, query_vector=pre_vec)  # 获取30个候选
                sorted_links = generate_prompt_and_sort_with_description(query, results, use_vector_results=True)
                
            else:  # vector_with_llm (智能混合模式)
                # 方案5: 向量检索 + LLM（智能混合模式）
                pre_vec = precomputed_map.get(str(query))
                results = vector_search(query, top_k=10, query_vector=pre_vec)  # 先获取前10个结果
                
                # 检查前10个结果中是否有正确链接
                top10_links = [r.get("link", "") for r in results[:10]]
                correct_link_cleaned = clean_link(str(correct_link))
                correct_link_normalized = normalize_url(correct_link_cleaned)
                
                has_hit_in_top10 = False
                for link in top10_links:
                    link_cleaned = clean_link(str(link))
                    link_normalized = normalize_url(link_cleaned)
                    if (correct_link_normalized == link_normalized or
                        correct_link_cleaned == link_cleaned or
                        correct_link_cleaned in link_cleaned or
                        link_cleaned in correct_link_cleaned):
                        has_hit_in_top10 = True
                        break
                
                if has_hit_in_top10:
                    # 前10有hit，使用向量检索结果进行LLM重排序
                    logger.info(f"[{search_mode}] 查询 '{query}' 前10有hit，使用向量检索结果进行LLM重排序")
                    sorted_links = generate_prompt_and_sort_with_description(query, results, use_vector_results=True)
                else:
                    # 前10没有hit，直接从ES获取候选，用LLM判断（不参考向量检索结果）
                    logger.info(f"[{search_mode}] 查询 '{query}' 前10无hit，直接从ES获取候选，用LLM判断")
                    sorted_links = llm_search_direct_from_es(query, top_k=30)

            rank = None

            # 改进的链接匹配：支持双向匹配、清理后的匹配和URL归一化
            correct_link_cleaned = clean_link(str(correct_link))
            correct_link_normalized = normalize_url(correct_link_cleaned)

            for i, link in enumerate(sorted_links):
                link_cleaned = clean_link(str(link))
                link_normalized = normalize_url(link_cleaned)

                # 多种匹配方式：
                # 1. 归一化后的URL匹配（处理URL编码问题）
                if correct_link_normalized == link_normalized:
                    rank = i + 1
                    break

                # 2. 清理后的精确匹配
                if correct_link_cleaned == link_cleaned:
                    rank = i + 1
                    break

                # 3. 双向子字符串匹配
                if (correct_link_cleaned in link_cleaned or
                        link_cleaned in correct_link_cleaned):
                    rank = i + 1
                    break

                # 4. 归一化后的双向匹配
                if (correct_link_normalized in link_normalized or
                        link_normalized in correct_link_normalized):
                    rank = i + 1
                    break

            # 记录详细结果
            result_entry = {
                "query": query,
                "correct_link": correct_link,
                "rank": rank,
                "sorted_links": sorted_links[:10],  # 只记录前10个结果
                "mrr": 1 / rank if rank is not None else 0,
                "hit@1": 1 if rank is not None and rank <= 1 else 0,
                "hit@5": 1 if rank is not None and rank <= 5 else 0,
                "hit@10": 1 if rank is not None and rank <= 10 else 0
            }

            detailed_results.append(result_entry)

            if rank is not None:
                if os.getenv("DEBUG_SEARCH", "0") == "1":
                    logger.info(f"[{search_mode}] 查询 '{query}' 的正确结果排名: {rank}")
                return 1 / rank, 1 if rank <= 1 else 0, 1 if rank <= 5 else 0, 1 if rank <= 10 else 0
            else:
                if os.getenv("DEBUG_SEARCH", "0") == "1":
                    logger.warning(f"[{search_mode}] 查询 '{query}' 未找到正确结果")
                return 0, 0, 0, 0
        except Exception as e:
            logger.error(f"[{search_mode}] 处理查询 '{query}' 时出错: {e}")
            result_entry = {
                "query": query,
                "correct_link": correct_link,
                "error": str(e),
                "mrr": 0,
                "hit@1": 0,
                "hit@5": 0,
                "hit@10": 0
            }
            detailed_results.append(result_entry)
            return 0, 0, 0, 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(process_query, query, correct_link) for query, correct_link in
                   zip(queries, correct_links)]

        # 使用tqdm显示进度，并实时更新指标
        progress_bar = tqdm(concurrent.futures.as_completed(futures), total=len(queries),
                            desc=f"Processing {search_mode} queries",
                            ncols=100,
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        for future in progress_bar:
            result = future.result()
            mrr += result[0]
            hit_at_1 += result[1]
            hit_at_5 += result[2]
            hit_at_10 += result[3]
            total_processed += 1

            # 实时计算并更新进度条显示的指标
            current_mrr = mrr / total_processed if total_processed > 0 else 0
            current_hit1 = hit_at_1 / total_processed if total_processed > 0 else 0
            current_hit5 = hit_at_5 / total_processed if total_processed > 0 else 0
            current_hit10 = hit_at_10 / total_processed if total_processed > 0 else 0

            # 更新进度条描述，显示当前指标
            progress_bar.set_postfix({
                'MRR': f'{current_mrr:.4f}',
                'Hit@1': f'{current_hit1:.4f}',
                'Hit@5': f'{current_hit5:.4f}',
                'Hit@10': f'{current_hit10:.4f}'
            })

    if total_processed > 0:
        mrr /= total_processed
        hit_at_1 /= total_processed
        hit_at_5 /= total_processed
        hit_at_10 /= total_processed
    else:
        print("No queries were processed successfully.")
        return 0, 0, 0, 0

    # 生成完整报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "search_mode": search_mode,
        "total_queries": len(queries),
        "processed_queries": total_processed,
        "metrics": {
            "mrr": mrr,
            "hit@1": hit_at_1,
            "hit@5": hit_at_5,
            "hit@10": hit_at_10
        },
        "detailed_results": detailed_results
    }

    # 保存报告到文件（保存到 trainlog 文件夹）
    report_filename = os.path.join(trainlog_dir, f'evaluation_report_{search_mode}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(report_filename, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"[{search_mode}] 完整评估报告已保存到文件")

    return mrr, hit_at_1, hit_at_5, hit_at_10


def evaluate_all_modes(queries, correct_links):
    """评估所有检索模式"""
    modes = [
        "vector_only",           # 方案1: 纯向量检索
        "es_text_only",          # 方案2: 纯ES文本搜索
        "llm_only",              # 方案3: 纯LLM判断
        "vector_with_llm_always", # 方案4: 始终使用LLM重排序向量结果
        "vector_with_llm"        # 方案5: 向量检索+LLM（智能混合模式）
    ]
    results = {}

    for mode in modes:
        print(f"\n开始评测 {mode} 检索模式...")
        logger.info(f"开始评测 {mode} 检索模式，共 {len(queries)} 个查询")

        mrr, hit_at_1, hit_at_5, hit_at_10 = calculate_metrics(queries, correct_links, mode)

        results[mode] = {
            "mrr": mrr,
            "hit@1": hit_at_1,
            "hit@5": hit_at_5,
            "hit@10": hit_at_10
        }

        print(f"\n{mode} 检索模式评测结果:")
        print("=" * 50)
        print(f"MRR: {mrr:.4f}")
        print(f"Hit@1: {hit_at_1:.4f}")
        print(f"Hit@5: {hit_at_5:.4f}")
        print(f"Hit@10: {hit_at_10:.4f}")

    # 输出汇总报告
    print(f"\n{'=' * 70}")
    print("所有检索模式评测结果汇总:")
    print(f"{'=' * 70}")
    print(f"{'模式':<20} {'MRR':<10} {'Hit@1':<10} {'Hit@5':<10} {'Hit@10':<10}")
    print("-" * 70)
    for mode in modes:
        metrics = results[mode]
        mode_display = {
            "vector_only": "方案1: 纯向量检索",
            "es_text_only": "方案2: 纯ES文本搜索",
            "llm_only": "方案3: 纯LLM判断",
            "vector_with_llm_always": "方案4: 向量+LLM(始终)",
            "vector_with_llm": "方案5: 向量+LLM(智能)"
        }.get(mode, mode)
        print(
            f"{mode_display:<20} {metrics['mrr']:<10.4f} {metrics['hit@1']:<10.4f} {metrics['hit@5']:<10.4f} {metrics['hit@10']:<10.4f}")

    # 保存汇总报告
    summary_report = {
        "timestamp": datetime.now().isoformat(),
        "summary": results
    }

    # 保存汇总报告到 trainlog 文件夹
    summary_filename = os.path.join(trainlog_dir, f'evaluation_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(summary_filename, 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, ensure_ascii=False, indent=2)

    logger.info("所有检索模式评测完成，汇总报告已保存")


def main():
    """主函数：从Excel文件读取测试集并计算评估指标"""
    import sys

    file_path = "data/find.xlsx"

    # 如果评测文件不存在，运行简单测试
    if not os.path.exists(file_path):
        print(f"未找到评测文件: {file_path}")
        print("运行简单测试模式...\n")
        print("支持的检索方案：")
        print("  1. vector_only - 纯向量检索")
        print("  2. es_text_only - 纯ES文本搜索")
        print("  3. llm_only - 纯LLM判断")
        print("  4. vector_with_llm_always - 向量+LLM（始终重排序）")
        print("  5. vector_with_llm - 向量+LLM（智能混合模式，推荐）")
        print("\n详细说明请参考：检索方案对比说明.md\n")

        # 测试查询列表
        test_queries = ["AK47", "F-16战斗机", "步枪"]

        for query in test_queries:
            test_vector_search_only(query, top_k=10)
            test_vector_search_with_llm(query, top_k=10)
            print("\n" + "=" * 60 + "\n")
        return

    try:
        queries, correct_links = read_excel(file_path)
        print(f"读取了 {len(queries)} 个查询")

        # 检查是否有指定评测模式的命令行参数
        mode_map = {
            "--vector-only": "vector_only",
            "--es-text-only": "es_text_only",
            "--llm-only": "llm_only",
            "--vector-llm-always": "vector_with_llm_always",
            "--vector-llm": "vector_with_llm"
        }
        
        if len(sys.argv) > 1 and sys.argv[1] in mode_map:
            # 显式指定单一模式
            mode = mode_map[sys.argv[1]]
            mode_names = {
                "vector_only": "方案1: 纯向量检索",
                "es_text_only": "方案2: 纯ES文本搜索",
                "llm_only": "方案3: 纯LLM判断",
                "vector_with_llm_always": "方案4: 向量+LLM（始终重排序）",
                "vector_with_llm": "方案5: 向量+LLM（智能混合模式）"
            }
            logger.info(f"开始评测 {mode} 检索模式，共 {len(queries)} 个查询")
            print(f"开始评测 {mode_names.get(mode, mode)}...")
            mrr, hit_at_1, hit_at_5, hit_at_10 = calculate_metrics(queries, correct_links, mode)
            print(f"\n{mode_names.get(mode, mode)} 评测结果:")
            print(f"{'=' * 50}")
            print(f"MRR: {mrr:.4f}")
            print(f"Hit@1: {hit_at_1:.4f}")
            print(f"Hit@5: {hit_at_5:.4f}")
            print(f"Hit@10: {hit_at_10:.4f}")
        else:
            # 默认或 --all：依次评测所有模式并输出汇总
            print("\n" + "=" * 70)
            print("开始评估所有5种检索方案")
            print("=" * 70)
            print("方案列表：")
            print("  1. vector_only - 纯向量检索")
            print("  2. es_text_only - 纯ES文本搜索")
            print("  3. llm_only - 纯LLM判断")
            print("  4. vector_with_llm_always - 向量+LLM（始终重排序）")
            print("  5. vector_with_llm - 向量+LLM（智能混合模式，推荐）")
            print("\n详细说明请参考：检索方案对比说明.md\n")
            evaluate_all_modes(queries, correct_links)
    except Exception as e:
        print(f"评测失败: {e}")
        logger.error(f"评测失败: {e}")
        print("\n运行简单测试模式...\n")

        # 测试查询列表
        test_queries = ["AK47", "F-16战斗机", "步枪"]

        for query in test_queries:
            test_vector_search_only(query, top_k=10)
            test_vector_search_with_llm(query, top_k=10)
            print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n用户中断执行")
    except Exception as e:
        logger.error(f"\n执行异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 恢复stdout和stderr，并关闭文件
        if 'tee' in globals():
            original_stdout = tee.stdout
            original_stderr = tee.stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            tee.close()
            print(f"\n✅ 控制台输出已保存到: {console_log_file}")
            print(f"✅ 日志信息已保存到: {log_filename}")

