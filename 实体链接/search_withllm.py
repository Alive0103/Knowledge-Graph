import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
from tqdm import tqdm
from zhipuai import ZhipuAI
import concurrent.futures
import re
import numpy as np
from functools import lru_cache
from urllib.parse import quote, unquote
from 实体链接.es_client import es
import logging
import json
import os
from datetime import datetime

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('search_evaluation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 模型加载（用于向量生成，支持 GPU 加速）
# 支持两种模型：
# - chinese-roberta-wwm-ext: 768维
# - chinese-roberta-wwm-ext-large: 1024维（推荐，与ES向量字段维度匹配）
model_name = './model/chinese-roberta-wwm-ext-large'
# model_name = 'D:/model/chinese-roberta-wwm-ext-large'
# model_name = './model/chinese-roberta-wwm-ext'  # 768维模型
model = None
tokenizer = None
model_dimension = None  # 模型向量维度
device = torch.device("cpu")
try:
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()

    # GPU / CPU 选择（与向量入库、debug 脚本保持一致）
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"✅ 检测到 GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = torch.device("cpu")
        print("⚠️ 未检测到 GPU，使用 CPU 进行在线向量检索（性能会略慢）")

    model = model.to(device)

    # 获取模型隐藏层维度
    model_dimension = model.config.hidden_size
    print(f"✓ Chinese-RoBERTa模型加载成功 (维度: {model_dimension}, 设备: {device})")
    if model_dimension == 1024:
        print("  ✓ 模型维度与ES向量字段匹配（1024维）")
    elif model_dimension == 768:
        print("  ⚠ 模型维度为768维，将自动扩展到1024维以匹配ES字段")
except Exception as e:
    print(f"警告: 模型加载失败 ({e})，向量生成功能将不可用")
    print("提示: 模型是可选的，如果不使用向量检索，可以跳过")

# 智谱AI API客户端（必需，用于LLM重排序功能）
# API密钥获取：https://open.bigmodel.cn/
client = ZhipuAI(api_key="1a2a485fe1fc4bd5aa0d965bf452c8c8.se8RZdT8cH8skEDo")

# 向量缓存字典（用于缓存生成的向量）
_vector_cache = {}
_cache_max_size = 1000

# 批量向量生成时的默认 batch 大小（仅用于评测加速）
_batch_size_for_eval = 32

def _generate_vector_internal(text):
    """
    内部向量生成函数（实际生成向量）
    
    自动处理不同维度的模型：
    - 1024维模型：直接使用，无需转换
    - 768维模型：零填充到1024维
    - 其他维度：自动调整到1024维
    """
    if model is None or tokenizer is None:
        return None
    
    # 与向量入库 / debug 向量搜索保持一致：支持 GPU，结果转回 CPU 再做 numpy
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    vector = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    
    # L2归一化
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    
    # 处理维度问题：ES需要1024维向量
    # 如果模型是1024维（large模型），直接使用
    # 如果模型是768维（base模型），需要扩展到1024维
    vector_dim = len(vector)
    target_dim = 1024  # ES向量字段维度
    
    if vector_dim == target_dim:
        # 维度匹配，直接使用（large模型的情况）
        pass
    elif vector_dim == 768:
        # 768维模型，零填充到1024维
        vector = np.pad(vector, (0, target_dim - vector_dim), 'constant', constant_values=0)
        # 重新归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
    else:
        # 其他维度，自动调整
        if vector_dim < target_dim:
            vector = np.pad(vector, (0, target_dim - vector_dim), 'constant', constant_values=0)
        else:
            vector = vector[:target_dim]
        # 重新归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
    
    return vector.tolist()


def _batch_generate_vectors_internal(texts, use_cache=True, batch_size=None):
    """
    批量生成 1024 维向量（仅用于评测加速）
    
    - 复用当前已加载的 model / tokenizer / device
    - 自动处理 768/1024 维模型到 1024 维
    - 做 L2 归一化
    """
    if model is None or tokenizer is None:
        return [None] * len(texts)
    
    if batch_size is None:
        batch_size = _batch_size_for_eval
    
    results = [None] * len(texts)
    to_compute_indices = []
    to_compute_texts = []
    
    for i, t in enumerate(texts):
        if not t or not str(t).strip():
            continue
        t = str(t)
        if use_cache and t in _vector_cache:
            results[i] = _vector_cache[t]
        else:
            to_compute_indices.append(i)
            to_compute_texts.append(t)
    
    if not to_compute_texts:
        return results
    
    target_dim = 1024
    
    for start in range(0, len(to_compute_texts), batch_size):
        end = start + batch_size
        batch_texts = to_compute_texts[start:end]
        if not batch_texts:
            continue
        
        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        # [B, H]
        vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        for j in range(vectors.shape[0]):
            v = vectors[j]
            # 处理维度为 1024
            vec_dim = len(v)
            if vec_dim == target_dim:
                pass
            elif vec_dim < target_dim:
                v = np.pad(v, (0, target_dim - vec_dim), 'constant', constant_values=0)
            else:
                v = v[:target_dim]
            
            # L2 归一化
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
    生成文本向量（需要模型已加载）
    
    Args:
        text: 输入文本
        use_cache: 是否使用缓存（默认True）
    
    Returns:
        1024维向量列表，已L2归一化
    """
    if not text or model is None or tokenizer is None:
        return None
    
    # 使用缓存
    if use_cache:
        cache_key = str(text)
        if cache_key in _vector_cache:
            return _vector_cache[cache_key]
        
        # 生成向量
        vector = _generate_vector_internal(cache_key)
        
        # 添加到缓存（如果缓存已满，删除最旧的）
        if len(_vector_cache) >= _cache_max_size:
            # 删除第一个（FIFO）
            first_key = next(iter(_vector_cache))
            del _vector_cache[first_key]
        
        _vector_cache[cache_key] = vector
        return vector
    else:
        # 不使用缓存，直接生成
        return _generate_vector_internal(str(text))

def text_search(query_text, top_k=20, text_boost=1.0):
    """
    单独的文本检索
    
    Args:
        query_text: 查询文本
        top_k: 返回结果数量
        text_boost: 文本检索的boost权重（默认1.0）
    
    Returns:
        检索结果列表
    """
    # 构建文本检索查询
    text_query = {
        "bool": {
            "should": [
                {
                    "match": {
                        "label": {
                            "query": query_text,
                            "boost": text_boost
                        }
                    }
                },
                {
                    "match": {
                        "aliases_zh": {
                            "query": query_text,
                            "boost": text_boost
                        }
                    }
                },
            ]
        }
    }
    
    # 尝试多个索引名称
    index_names = ["data2", "data1"]
    response = None
    for index_name in index_names:
        try:
            if es.indices.exists(index=index_name):
                search_query = {
                    "query": text_query,
                    "size": top_k
                }
                response = es.search(index=index_name, body=search_query)
                break
        except Exception as e:
            # 如果所有尝试都失败，继续尝试下一个索引
            continue
    
    if response is None:
        raise Exception(f"未找到可用的索引，尝试过的索引: {index_names}")
    
    hits = response["hits"]["hits"]
    results = [
        {
            "label": hit["_source"].get("label", ""),
            "aliases_zh": hit["_source"].get("aliases_zh", []),
            "aliases_en": hit["_source"].get("aliases_en", []),
            "descriptions_zh": hit["_source"].get("descriptions_zh", ""),
            "link": hit["_source"].get("link", "")
        } 
        for hit in hits
    ]
    return results

def vector_search(query_text, top_k=20, vector_boost=0.8, query_vector=None, use_label_vector=False, use_llm_definition=False):
    """
    单独的向量检索
    
    Args:
        query_text: 查询文本
        top_k: 返回结果数量
        vector_boost: 向量检索的boost权重（默认0.8）
        query_vector: 预计算的查询向量（可选）
        use_label_vector: 是否同时使用 label_vector 字段（默认False）
        use_llm_definition: 是否使用 LLM 生成定义后生成向量（对齐方案，默认False）
    
    Returns:
        检索结果列表
    """
    # 生成查询向量
    if query_vector is None and model is not None and tokenizer is not None:
        try:
            if use_llm_definition:
                # 方案A：使用 LLM 生成规范化定义，然后生成向量（对齐方案）
                try:
                    response_content = get_alias_and_definition(query_text)
                    # 提取定义部分
                    if "定义：" in response_content:
                        definition = response_content.split("定义：")[1].strip()
                        definition = definition.split("\n")[0].split("标签：")[0].strip()
                    elif "定义" in response_content:
                        parts = response_content.split("定义")
                        if len(parts) > 1:
                            definition = parts[1].strip().lstrip("：").lstrip(":").strip()
                            definition = definition.split("\n")[0].split("标签：")[0].strip()
                        else:
                            definition = query_text  # 回退到原始查询
                    else:
                        definition = query_text  # 回退到原始查询
                    
                    query_vector = generate_vector(definition, use_cache=True)
                except Exception as e:
                    # LLM 失败，回退到直接使用查询文本
                    query_vector = generate_vector(query_text, use_cache=True)
            else:
                # 直接使用查询文本生成向量
                query_vector = generate_vector(query_text, use_cache=True)
        except Exception:
            query_vector = None
    
    if query_vector is None:
        return []  # 如果没有向量，返回空结果
    
    # 确定要搜索的向量字段
    if use_label_vector:
        # 方案B：同时对 descriptions_vector 和 label_vector 进行搜索
        vector_fields = [
            ("descriptions_zh_vector", "zh", "desc"),
            ("descriptions_en_vector", "en", "desc"),
            ("label_zh_vector", "zh", "label"),
            ("label_en_vector", "en", "label")
        ]
    else:
        # 默认：只搜索 descriptions_vector
        vector_fields = [
            ("descriptions_zh_vector", "zh", "desc"),
            ("descriptions_en_vector", "en", "desc")
        ]
    
    # 同时对多个向量字段做检索，然后融合结果
    index_names = ["data2", "data1"]
    merged_hits = {}  # key: (index, _id), value: {'source': ..., 'score': ..., 'lang': 'zh'/'en', 'field_type': 'desc'/'label'}
    
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
                "k": top_k,
                "num_candidates": top_k * 3
            }
            search_body = {
                "knn": knn_query,
                "size": top_k
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
                            "field_type": field_type  # 'desc' 或 'label'
                        }
            except Exception:
                # 某个字段失败不影响整体，继续尝试其他字段 / 索引
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
            "link": source.get("link", ""),
            "_score": item["score"],
            "_lang": item["lang"],  # 调试用：来自中文/英文向量
            "_field_type": item["field_type"]  # 调试用：来自描述向量还是标签向量
        }
        results.append(result)
        
    return results

def hybrid_search(query_text, top_k=20, text_boost=1.0, vector_boost=1.5, use_vector=True, use_llm_definition=True):
    """
    混合检索：文本检索 + 向量检索
    
    Args:
        query_text: 查询文本
        top_k: 返回结果数量
        text_boost: 文本检索的boost权重（默认1.0）
        vector_boost: 向量检索的boost权重（默认1.5，提高向量权重）
        use_vector: 是否使用向量检索（默认True）
    
    Returns:
        检索结果列表
    """
    # 构建文本检索查询
    text_query = {
        "bool": {
            "should": [
                {
                    "match": {
                        "label": {
                            "query": query_text,
                            "boost": text_boost
                        }
                    }
                },
                {
                    "match": {
                        "aliases_zh": {
                            "query": query_text,
                            "boost": text_boost
                        }
                    }
                },
            ]
        }
    }
    
    # 尝试生成查询向量
    query_vector = None
    knn_query = None
    
    if use_vector and model is not None and tokenizer is not None:
        if use_llm_definition:
            # 使用 LLM 生成规范化定义，再生成向量（原有逻辑）
            try:
                response_content = get_alias_and_definition(query_text)
                # 更健壮的解析：支持多种格式
                if "定义：" in response_content:
                    input_definition = response_content.split("定义：")[1].strip()
                    # 移除可能的后续内容（如换行、其他标签等）
                    input_definition = input_definition.split("\n")[0].split("标签：")[0].strip()
                elif "定义" in response_content:
                    # 尝试没有冒号的格式
                    parts = response_content.split("定义")
                    if len(parts) > 1:
                        input_definition = parts[1].strip().lstrip("：").lstrip(":").strip()
                        input_definition = input_definition.split("\n")[0].split("标签：")[0].strip()
                    else:
                        raise ValueError("无法从LLM响应中提取定义")
                else:
                    raise ValueError("LLM响应中未找到定义字段")
                
                if not input_definition:
                    raise ValueError("提取的定义为空")
                
                query_vector = generate_vector(input_definition, use_cache=True)
            except (ValueError, IndexError, Exception):
                # 如果LLM失败，使用原始查询文本生成向量（静默失败，不打印错误）
                try:
                    query_vector = generate_vector(query_text, use_cache=True)
                except Exception:
                    query_vector = None
        else:
            # 不使用 LLM，直接对原始 query 文本生成向量（用于 hybrid_no_llm ablation）
            try:
                query_vector = generate_vector(query_text, use_cache=True)
            except Exception:
                query_vector = None
        
        # 如果成功生成向量，构建KNN查询
        if query_vector is not None:
            # 优先使用descriptions_zh_vector（因为content_vector字段可能不存在）
            # 如果descriptions_zh_vector查询返回空结果，可以尝试content_vector作为备选
            knn_query = {
                "field": "descriptions_zh_vector",  # 直接使用descriptions_zh_vector（99%文档有此字段）
                "query_vector": query_vector,
                "k": 10,
                "num_candidates": 20,
                "boost": vector_boost
            }
    
    # 构建混合查询
    if knn_query:
        # 使用hybrid query（文本 + 向量）
        search_query = {
            "query": text_query,
            "knn": knn_query,
            "size": top_k
        }
    else:
        # 仅使用文本检索
        search_query = {
            "query": text_query,
            "size": top_k
        }
    # 尝试多个索引名称
    index_names = ["data2", "data1"]
    response = None
    for index_name in index_names:
        try:
            if es.indices.exists(index=index_name):
                # 执行混合查询
                try:
                    response = es.search(index=index_name, body=search_query)
                    # 检查返回结果数量，如果为0且使用的是descriptions_zh_vector，可以尝试content_vector
                    if response and response.get("hits", {}).get("total", {}).get("value", 0) == 0:
                        # 如果descriptions_zh_vector返回空结果，尝试content_vector（如果存在）
                        if knn_query and knn_query.get("field") == "descriptions_zh_vector":
                            try:
                                # 尝试使用content_vector
                                search_query["knn"]["field"] = "content_vector"
                                response_content = es.search(index=index_name, body=search_query)
                                # 如果content_vector有结果，使用它；否则保持原结果
                                if response_content and response_content.get("hits", {}).get("total", {}).get("value", 0) > 0:
                                    response = response_content
                            except Exception:
                                # content_vector查询失败，保持使用descriptions_zh_vector的结果（空结果）
                                pass
                except Exception as e:
                    # 如果descriptions_zh_vector查询失败，尝试content_vector作为备选
                    if knn_query and knn_query.get("field") == "descriptions_zh_vector" and ("field" in str(e).lower() or "descriptions_zh_vector" in str(e).lower()):
                        try:
                            search_query["knn"]["field"] = "content_vector"
                            response = es.search(index=index_name, body=search_query)
                        except Exception as e2:
                            # 如果content_vector也失败，移除knn查询，仅使用文本检索
                            search_query = {
                                "query": text_query,
                                "size": top_k
                            }
                            response = es.search(index=index_name, body=search_query)
                    else:
                        raise e
                break
        except Exception as e:
            # 如果所有尝试都失败，继续尝试下一个索引
            continue
    
    if response is None:
        raise Exception(f"未找到可用的索引，尝试过的索引: {index_names}")
    hits = response["hits"]["hits"]
    # results = [hit["_source"] for hit in hits]
    results = [
        {
            "label": hit["_source"].get("label", ""),
            "aliases_zh": hit["_source"].get("aliases_zh", []),
            "aliases_en": hit["_source"].get("aliases_en", []),
            "descriptions_zh": hit["_source"].get("descriptions_zh", ""),
            "link": hit["_source"].get("link", "")
        } 
        for hit in hits
    ]
    return results

def get_alias_and_definition(mention):
    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=[
            {
                "role": "user",
                "content": (
                    f"你现在是军事领域专家，需要参照以下例子给出提及对应的别名和定义。\n"
                    f"例子：提及：Steyr HS .50、别名：斯泰尔HS .50狙击步枪、定义：斯泰尔HS .50（Steyr HS.50）是由奥地利斯泰尔-曼利夏公司研制的一款手动枪机式反器材狙击步枪。\n\n"
                    f"输入提及：{mention}\n\n"
                    f"请按照标签：{mention}、中文别名：、英文别名：、定义：的格式直接返回所需内容，不要解释或附加内容。"
                )
            }
        ],
    )
    response_content = response.choices[0].message.content.strip()
    
    if not response_content:
        raise ValueError(f"No response content for mention '{mention}'")
    
    return response_content

# 全局错误回调函数（用于详细评测）
_error_callback = None

def set_error_callback(callback):
    """设置错误回调函数，用于详细评测时记录错误"""
    global _error_callback
    _error_callback = callback

def generate_prompt_and_sort(mention, results):
    input_label = mention
    response_content = ""
    try:
        response_content = get_alias_and_definition(mention)
        
        # 更健壮的解析：支持多种格式和缺失字段
        def safe_extract(content, field_name, default=""):
            """安全提取字段内容"""
            # 尝试带冒号的格式
            if f"{field_name}：" in content:
                parts = content.split(f"{field_name}：", 1)
                if len(parts) > 1:
                    value = parts[1].split("英文别名")[0].split("定义")[0].split("\n")[0].strip()
                    return value if value else default
            # 尝试不带冒号的格式
            elif field_name in content:
                # 简单匹配，提取字段后的内容
                idx = content.find(field_name)
                if idx != -1:
                    start = idx + len(field_name)
                    # 跳过可能的冒号或空格
                    while start < len(content) and content[start] in [":", "：", " ", "\t"]:
                        start += 1
                    # 提取到下一个字段或换行
                    end = len(content)
                    for marker in ["英文别名", "定义", "\n", "标签"]:
                        marker_idx = content.find(marker, start)
                        if marker_idx != -1 and marker_idx < end:
                            end = marker_idx
                    value = content[start:end].strip()
                    return value if value else default
            return default
        
        input_aliases_zh = safe_extract(response_content, "中文别名", "")
        input_aliases_en = safe_extract(response_content, "英文别名", "")
        input_definition = safe_extract(response_content, "定义", "")
        
        # 如果所有字段都为空，说明解析失败
        if not input_aliases_zh and not input_aliases_en and not input_definition:
            raise ValueError("无法从LLM响应中提取任何有效字段")
            
    except (ValueError, IndexError, Exception) as e:
        # 记录详细错误信息
        error_info = {
            "type": "llm_parse_error",
            "mention": mention,
            "error": str(e),
            "error_type": type(e).__name__,
            "response_content": response_content[:500] if 'response_content' in locals() and response_content else None
        }
        
        # 调用错误回调（如果设置了）
        if _error_callback:
            try:
                _error_callback("llm_parse_error", mention, str(e), error_info)
            except:
                pass
        
        # 只在调试模式下打印详细错误，避免输出过多
        import os
        if os.getenv("DEBUG_LLM", "0") == "1":
            print(f"LLM failed to generate valid response for mention '{mention}'. Error: {e}")
            print(f"  Response content: {response_content[:200] if 'response_content' in locals() else 'N/A'}")
        
        return [result['link'] for result in results]  

    options = []
    original_links = []  
    for idx, result in enumerate(results, start=1):
        option = (
            f"选项{idx}：\n"
            f"label: {result['label']}\n"
            f"aliases_zh: {', '.join(result['aliases_zh'])}\n"
            f"aliases_en: {', '.join(result['aliases_en'])}\n"
            f"descriptions_zh: {result['descriptions_zh']}\n"
            f"link: {result['link']}\n"
        )
        options.append(option)
        original_links.append(result['link'])  

    prompt = (
        f"现在你是军事领域专家，需要根据输入信息与选项列表的候选的匹配度进行从高到低排序\n"
        f"输入标签名：{input_label}\n"
        f"输入中文别名：{input_aliases_zh}\n"
        f"输入英文别名：{input_aliases_en}\n"
        f"输入定义：{input_definition}\n\n"
        f"选项列表：\n"
        f"{''.join(options)}\n\n"
        f"请根据输入信息与选项的匹配度，从高到低严格返回所有候选的link值，确保返回的link值是原始选项列表中的link值的排序，不能有缺失或重复，不要解释或附加内容。"
    )

    try:
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": prompt}]
        )
        # 解析LLM返回的链接，支持多种格式
        response_text = response.choices[0].message.content.strip()
        # 按行分割，并清理每行
        sorted_links_raw = [line.strip() for line in response_text.split("\n") if line.strip()]
        # 使用ensure_links_match进行匹配和清理
        sorted_links = ensure_links_match(sorted_links_raw, original_links)
        return sorted_links
    except Exception as e:
        # 记录详细错误信息
        error_info = {
            "type": "llm_sort_error",
            "mention": mention,
            "error": str(e),
            "error_type": type(e).__name__,
            "results_count": len(results)
        }
        
        # 调用错误回调（如果设置了）
        if _error_callback:
            try:
                _error_callback("llm_sort_error", mention, str(e), error_info)
            except:
                pass
        
        import os
        if os.getenv("DEBUG_LLM", "0") == "1":
            print(f"LLM failed to sort links for mention '{mention}'. Error: {e}")
        
        return original_links  
    
def normalize_url(url):
    """
    归一化URL，处理URL编码问题
    将URL编码和解码的版本都归一化到同一个格式进行比较
    返回解码后的标题部分（用于比较）
    """
    if not url:
        return ""
    
    url = str(url).strip()
    
    # 如果是维基百科链接，提取标题部分进行归一化
    if "wikipedia.org/wiki/" in url:
        try:
            # 提取wiki标题部分
            if "/wiki/" in url:
                parts = url.split("/wiki/", 1)
                if len(parts) == 2:
                    title = parts[1]
                    
                    # 解码URL编码（如果有）
                    try:
                        decoded_title = unquote(title)
                    except:
                        decoded_title = title
                    
                    # 返回解码后的标题（用于比较）
                    # 这样无论输入是URL编码还是中文直接编码，都能正确匹配
                    return decoded_title
        except Exception as e:
            # 如果处理失败，返回原始URL的标题部分
            pass
    
    # 如果不是维基百科链接，返回原始URL
    return url

def clean_link(link):
    """清理链接，移除空白字符和常见前缀"""
    if not link:
        return ""
    # 移除首尾空白
    link = str(link).strip()
    # 移除常见的编号前缀（如 "1. ", "选项1: " 等）
    link = re.sub(r'^\d+[\.\)]\s*', '', link)  # 移除 "1. " 或 "1) "
    link = re.sub(r'^选项\d+[：:]\s*', '', link)  # 移除 "选项1: "
    link = re.sub(r'^link[：:]\s*', '', link, flags=re.IGNORECASE)  # 移除 "link: "
    return link.strip()

def ensure_links_match(sorted_links, original_links):
    """
    确保排序后的链接与原始链接一致，替换不匹配的链接。
    支持模糊匹配（子字符串匹配）。
    """
    # 清理所有链接
    cleaned_sorted = [clean_link(link) for link in sorted_links]
    original_links_set = set(original_links)
    
    # 创建映射：清理后的链接 -> 原始链接
    cleaned_to_original = {}
    for orig_link in original_links:
        cleaned = clean_link(orig_link)
        cleaned_to_original[cleaned] = orig_link
    
    # 匹配和重建排序列表
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
        
        # 2. URL归一化匹配（处理URL编码问题）
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
        
        # 3. 如果精确匹配失败，尝试模糊匹配（子字符串匹配）
        if not matched:
            for orig_link in original_links:
                if orig_link not in used_original_links:
                    orig_cleaned = clean_link(orig_link)
                    # 双向子字符串匹配
                    if cleaned_link in orig_cleaned or orig_cleaned in cleaned_link:
                        matched_links.append(orig_link)
                        used_original_links.add(orig_link)
                        matched = True
                        break
        
        # 4. 归一化后的模糊匹配
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

def read_excel(file_path):
    df = pd.read_excel(file_path, header=None)
    queries = df[0].tolist()
    correct_links = df[1].tolist()
    return queries, correct_links

def calculate_metrics(queries, correct_links, search_mode="hybrid"):
    """
    计算评估指标
    
    Args:
        queries: 查询列表
        correct_links: 正确链接列表
        search_mode: 检索模式 ("text", "vector", "vector_no_llm", "vector_aligned", "vector_dual", "hybrid", "hybrid_no_llm")
    """
    mrr = 0
    hit_at_1 = 0
    hit_at_5 = 0
    hit_at_10 = 0
    total_processed = 0
    
    # 用于存储详细结果的列表
    detailed_results = []

    # 仅在向量检索评测模式预先批量生成向量，加速 GPU 计算
    precomputed_vectors = None
    precomputed_map = {}
    if search_mode in ["vector", "vector_no_llm", "vector_dual"] and model is not None and tokenizer is not None:
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
            if search_mode == "text":
                results = text_search(query)
                sorted_links = generate_prompt_and_sort(query, results)
            elif search_mode == "vector":
                # 如果有预先计算好的向量，就直接复用，避免重复跑 BERT
                pre_vec = precomputed_map.get(str(query))
                results = vector_search(query, query_vector=pre_vec)
                sorted_links = generate_prompt_and_sort(query, results)
            elif search_mode == "vector_no_llm":
                # 纯向量检索，不使用 LLM 重排序，直接按 ES 返回顺序（按相似度分数排序）
                pre_vec = precomputed_map.get(str(query))
                results = vector_search(query, query_vector=pre_vec)
                # 直接按 ES 返回的顺序（已经按相似度分数排序）
                sorted_links = [r.get("link", "") for r in results]
            elif search_mode == "vector_aligned":
                # 方案A：使用 LLM 生成定义后生成向量（对齐方案），不使用 LLM 重排序
                results = vector_search(query, use_llm_definition=True)
                # 直接按 ES 返回的顺序（已经按相似度分数排序）
                sorted_links = [r.get("link", "") for r in results]
            elif search_mode == "vector_dual":
                # 方案B：同时对 descriptions_vector 和 label_vector 进行搜索并融合，不使用 LLM 重排序
                pre_vec = precomputed_map.get(str(query))
                results = vector_search(query, query_vector=pre_vec, use_label_vector=True)
                # 直接按 ES 返回的顺序（已经按相似度分数排序）
                sorted_links = [r.get("link", "") for r in results]
            elif search_mode == "hybrid_no_llm":
                # ES 混合检索，不使用 LLM，按 ES 返回顺序评测
                results = hybrid_search(query, use_vector=True, use_llm_definition=False)
                sorted_links = [r.get("link", "") for r in results]
            else:  # hybrid（文本 + 向量 + LLM 重排序）
                results = hybrid_search(query)
                sorted_links = generate_prompt_and_sort(query, results)
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
            
            # 正式检索时，减少详细日志输出，只记录关键信息
            # 详细信息已通过进度条显示
            if rank is not None:
                # 只在调试模式下输出详细日志
                if os.getenv("DEBUG_SEARCH", "0") == "1":
                    logger.info(f"[{search_mode}] 查询 '{query}' 的正确结果排名: {rank}")
                return 1 / rank, 1 if rank <= 1 else 0, 1 if rank <= 5 else 0, 1 if rank <= 10 else 0
            else:
                # 只在调试模式下输出详细日志
                if os.getenv("DEBUG_SEARCH", "0") == "1":
                    logger.warning(f"[{search_mode}] 查询 '{query}' 未找到正确结果")
                return 0, 0, 0, 0
        except Exception as e:
            # 错误仍然记录，但减少详细输出
            logger.error(f"[{search_mode}] 处理查询 '{query}' 时出错: {e}")
            # 记录出错的查询
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
        futures = [executor.submit(process_query, query, correct_link) for query, correct_link in zip(queries, correct_links)]
        
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
    
    # 保存报告到文件
    with open(f'evaluation_report_{search_mode}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"[{search_mode}] 完整评估报告已保存到文件")
    
    return mrr, hit_at_1, hit_at_5, hit_at_10

def evaluate_all_modes(queries, correct_links):
    """评估所有检索模式"""
    modes = ["text", "vector", "vector_no_llm", "vector_aligned", "vector_dual", "hybrid_no_llm", "hybrid"]
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
    print(f"\n{'='*70}")
    print("所有检索模式评测结果汇总:")
    print(f"{'='*70}")
    print(f"{'模式':<18} {'MRR':<10} {'Hit@1':<10} {'Hit@5':<10} {'Hit@10':<10}")
    print("-" * 70)
    for mode in modes:
        metrics = results[mode]
        # 美化模式名称显示
        mode_display = {
            "text": "文本检索+LLM",
            "vector": "向量检索+LLM",
            "vector_no_llm": "向量检索(无LLM)",
            "vector_aligned": "向量检索(对齐)",
            "vector_dual": "向量检索(双字段)",
            "hybrid_no_llm": "混合检索(无LLM)",
            "hybrid": "混合检索+LLM"
        }.get(mode, mode)
        print(f"{mode_display:<18} {metrics['mrr']:<10.4f} {metrics['hit@1']:<10.4f} {metrics['hit@5']:<10.4f} {metrics['hit@10']:<10.4f}")
    
    # 保存汇总报告
    summary_report = {
        "timestamp": datetime.now().isoformat(),
        "summary": results
    }
    
    with open(f'evaluation_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, ensure_ascii=False, indent=2)
    
    logger.info("所有检索模式评测完成，汇总报告已保存")

def test_individual_search_modes():
    """测试单独的搜索模式：文本检索、向量检索、混合检索"""
    print("=" * 60)
    print("测试单独的搜索模式")
    print("=" * 60)
    
    test_queries = ["AK47", "F-16战斗机", "步枪", "航空母舰"]
    
    for query in test_queries:
        print(f"\n查询: '{query}'")
        print("-" * 40)
        
        # 1. 文本检索
        try:
            print("1. 文本检索结果:")
            text_results = text_search(query, top_k=3)
            print(f"   找到 {len(text_results)} 个候选实体")
            for i, result in enumerate(text_results, 1):
                print(f"   {i}. {result.get('label', 'N/A')}")
                print(f"      链接: {result.get('link', 'N/A')}")
        except Exception as e:
            print(f"   文本检索失败: {e}")
        
        # 2. 向量检索
        try:
            print("\n2. 向量检索结果:")
            vector_results = vector_search(query, top_k=3)
            print(f"   找到 {len(vector_results)} 个候选实体")
            for i, result in enumerate(vector_results, 1):
                print(f"   {i}. {result.get('label', 'N/A')} (相似度: {result.get('_score', 'N/A')})")
                print(f"      链接: {result.get('link', 'N/A')}")
        except Exception as e:
            print(f"   向量检索失败: {e}")
        
        # 3. 混合检索
        try:
            print("\n3. 混合检索结果:")
            hybrid_results = hybrid_search(query, top_k=3)
            print(f"   找到 {len(hybrid_results)} 个候选实体")
            for i, result in enumerate(hybrid_results, 1):
                print(f"   {i}. {result.get('label', 'N/A')}")
                print(f"      链接: {result.get('link', 'N/A')}")
        except Exception as e:
            print(f"   混合检索失败: {e}")

def test_vector_search():
    """测试向量搜索功能"""
    print("=" * 50)
    print("测试向量搜索功能")
    print("=" * 50)
    
    test_queries = ["AK47", "F-16战斗机", "步枪"]
    
    for query in test_queries:
        try:
            print(f"\n查询: '{query}'")
            results = vector_search(query, top_k=5)
            print(f"找到 {len(results)} 个候选实体")
            
            # 显示结果
            for i, result in enumerate(results[:3], 1):
                print(f"  {i}. {result.get('label', 'N/A')}")
                print(f"     链接: {result.get('link', 'N/A')}")
                print(f"     描述: {result.get('descriptions_zh', '')[:100]}...")
                
        except Exception as e:
            print(f"查询 '{query}' 失败: {e}")
            import traceback
            traceback.print_exc()

def test_search():
    """测试搜索功能（不需要评测文件）"""
    print("=" * 50)
    print("测试LLM增强搜索功能")
    print("=" * 50)
    
    test_queries = ["AK47", "F-16战斗机", "步枪"]
    
    for query in test_queries:
        try:
            print(f"\n查询: '{query}'")
            results = hybrid_search(query, top_k=5)
            print(f"找到 {len(results)} 个候选实体")
            
            # 显示前3个结果
            for i, result in enumerate(results[:3], 1):
                print(f"  {i}. {result.get('label', 'N/A')}")
                print(f"     链接: {result.get('link', 'N/A')}")
            
            # 使用LLM重排序（需要API密钥）
            print(f"\n使用LLM重排序...")
            sorted_links = generate_prompt_and_sort(query, results)
            print(f"重排序后的前3个结果:")
            for i, link in enumerate(sorted_links[:3], 1):
                print(f"  {i}. {link}")
        except Exception as e:
            print(f"查询 '{query}' 失败: {e}")
            import traceback
            traceback.print_exc()

def main():
    import os
    import sys
    
    # 检查是否有命令行参数用于测试向量搜索
    if len(sys.argv) > 1 and sys.argv[1] == "--test-vector":
        test_vector_search()
        return
    
    # 检查是否有命令行参数用于测试单独的搜索模式
    if len(sys.argv) > 1 and sys.argv[1] == "--test-modes":
        test_individual_search_modes()
        return
    
    file_path = "find.xlsx"
    
    # 如果评测文件不存在，运行测试搜索
    if not os.path.exists(file_path):
        print(f"未找到评测文件: {file_path}")
        print("运行测试搜索模式...\n")
        test_search()
        return
    
    try:
        queries, correct_links = read_excel(file_path)
        print(f"读取了 {len(queries)} 个查询")
        
        # 检查是否有指定评测模式的命令行参数
        if len(sys.argv) > 1 and sys.argv[1] in ["--text", "--vector", "--vector-no-llm", "--vector-aligned", "--vector-dual", "--hybrid", "--hybrid-no-llm"]:
            # 显式指定单一模式
            arg = sys.argv[1]
            if arg == "--hybrid-no-llm":
                mode = "hybrid_no_llm"
            elif arg == "--vector-no-llm":
                mode = "vector_no_llm"
            elif arg == "--vector-aligned":
                mode = "vector_aligned"
            elif arg == "--vector-dual":
                mode = "vector_dual"
            else:
                mode = arg[2:]  # 移除 "--" 前缀
            logger.info(f"开始评测 {mode} 检索模式，共 {len(queries)} 个查询")
            print(f"开始评测 {mode} 检索模式...")
            mrr, hit_at_1, hit_at_5, hit_at_10 = calculate_metrics(queries, correct_links, mode)
            print(f"\n{mode} 检索模式评测结果:")
            print(f"{'='*50}")
            print(f"MRR: {mrr:.4f}")
            print(f"Hit@1: {hit_at_1:.4f}")
            print(f"Hit@5: {hit_at_5:.4f}")
            print(f"Hit@10: {hit_at_10:.4f}")
        else:
            # 默认或 --all：依次评测所有模式并输出汇总
            # （即：python search_withllm.py 与 python search_withllm.py --all 等价）
            evaluate_all_modes(queries, correct_links)
    except Exception as e:
        print(f"评测失败: {e}")
        logger.error(f"评测失败: {e}")
        print("\n运行测试搜索模式...\n")
        test_search()

if __name__ == "__main__":
    main()