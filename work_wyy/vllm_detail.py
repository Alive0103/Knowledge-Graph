import torch
from transformers import BertTokenizer, BertModel
from zhipuai import ZhipuAI
import numpy as np
import re
from urllib.parse import unquote
from es_client import es
import logging
import pandas as pd
import json
import os
from datetime import datetime

# 配置日志记录（详细模式）
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vector_search_debug.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 创建输出目录
output_dir = "debug_output"
os.makedirs(output_dir, exist_ok=True)

# 模型加载（用于向量生成，支持 GPU 加速）
model_name = './model/chinese-roberta-wwm-ext-large'
model = None
tokenizer = None
device = torch.device("cpu")
try:
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"✅ 检测到 GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = torch.device("cpu")
        print("⚠️ 未检测到 GPU，使用 CPU 进行在线向量检索（性能会略慢）")

    model = model.to(device)
    model_dimension = model.config.hidden_size
    print(f"✓ Chinese-RoBERTa模型加载成功 (维度: {model_dimension}, 设备: {device})")
except Exception as e:
    print(f"警告: 模型加载失败 ({e})，向量生成功能将不可用")

# 智谱AI API客户端
client = ZhipuAI(api_key="1a2a485fe1fc4bd5aa0d965bf452c8c8.se8RZdT8cH8skEDo")

# 向量缓存字典
_vector_cache = {}
_cache_max_size = 1000


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


def generate_vector(text, use_cache=True, verbose=True):
    """
    生成文本向量（需要模型已加载）

    Args:
        text: 输入文本
        use_cache: 是否使用缓存（默认True）
        verbose: 是否打印详细信息
    """
    # 预处理文本
    text = preprocess_query(text)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"🔹 生成向量: '{text}'")
        print(f"{'=' * 60}")

    if not text or model is None or tokenizer is None:
        if verbose:
            print("❌ 模型未加载，无法生成向量")
        return None

    # 使用缓存
    if use_cache:
        cache_key = text
        if cache_key in _vector_cache:
            if verbose:
                print(f"✅ 使用缓存向量（已存在）")
            return _vector_cache[cache_key]

    if verbose:
        print(f"🔄 开始生成向量...")

    # 生成向量
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    if verbose:
        print(f"   输入token数量: {inputs['input_ids'].shape[1]}")

    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    vector = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

    if verbose:
        print(f"   原始向量维度: {len(vector)}")

    # L2归一化
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    if verbose:
        print(f"   L2归一化后范数: {np.linalg.norm(vector):.6f}")

    # 处理维度问题：ES需要1024维向量
    vector_dim = len(vector)
    target_dim = 1024

    if vector_dim == target_dim:
        if verbose:
            print(f"   ✅ 向量维度匹配（{vector_dim}维）")
    elif vector_dim < target_dim:
        if verbose:
            print(f"   ⚠️  向量维度不足（{vector_dim}维），填充到{target_dim}维")
        vector = np.pad(vector, (0, target_dim - vector_dim), 'constant', constant_values=0)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
    else:
        if verbose:
            print(f"   ⚠️  向量维度超出（{vector_dim}维），截断到{target_dim}维")
        vector = vector[:target_dim]
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

    vector_list = vector.tolist()

    if verbose:
        print(f"   ✅ 最终向量维度: {len(vector_list)}")
        print(f"   ✅ 向量前5个值: {vector_list[:5]}")

    # 添加到缓存
    if use_cache:
        if len(_vector_cache) >= _cache_max_size:
            first_key = next(iter(_vector_cache))
            del _vector_cache[first_key]
        _vector_cache[cache_key] = vector_list

    return vector_list


def vector_search(query_text, top_k=20, query_vector=None, verbose=True):
    """
    单独的向量检索（同时检索所有7个向量字段）

    Args:
        query_text: 查询文本
        top_k: 返回结果数量
        query_vector: 预计算的查询向量（可选）
        verbose: 是否打印详细信息
    """
    # 预处理查询
    query_text = preprocess_query(query_text)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"🔍 向量检索: '{query_text}'")
        print(f"{'=' * 60}")

    # 生成查询向量
    if query_vector is None and model is not None and tokenizer is not None:
        if verbose:
            print(f"📝 生成查询向量...")
        try:
            query_vector = generate_vector(query_text, use_cache=True, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"❌ 向量生成失败: {e}")
            query_vector = None

    if query_vector is None:
        if verbose:
            print(f"❌ 无法生成查询向量，返回空结果")
        return []

    if verbose:
        print(f"✅ 查询向量已准备（维度: {len(query_vector)}）")

    # 确定要搜索的向量字段（同时检索所有向量字段）
    vector_fields = [
        ("descriptions_zh_vector", "zh", "desc"),
        ("descriptions_en_vector", "en", "desc"),
        ("high_freq_words_zh_vector", "zh", "high_freq"),
        ("high_freq_words_en_vector", "en", "high_freq"),
        ("label_vector", "mixed", "label"),
        ("label_zh_vector", "zh", "label"),
        ("label_en_vector", "en", "label")
    ]

    if verbose:
        print(f"\n📋 搜索所有向量字段（共{len(vector_fields)}个）:")
        for field_name, lang_tag, field_type in vector_fields:
            print(f"   - {field_name} ({lang_tag}, {field_type})")

    # 同时对多个向量字段做检索，然后融合结果
    index_names = ["data2"]  # 只使用data2索引
    merged_hits = {}

    for index_name in index_names:
        if verbose:
            print(f"\n🔎 尝试索引: {index_name}")
        try:
            if not es.indices.exists(index=index_name):
                if verbose:
                    print(f"   ⚠️  索引不存在，跳过")
                continue
            if verbose:
                print(f"   ✅ 索引存在")
        except Exception as e:
            if verbose:
                print(f"   ❌ 检查索引失败: {e}")
            continue

        # 针对当前索引，尝试所有向量字段
        for field_name, lang_tag, field_type in vector_fields:
            if verbose:
                print(f"\n   🔍 搜索字段: {field_name} (语言: {lang_tag}, 类型: {field_type})")

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

                if verbose:
                    print(f"      ✅ 找到 {len(hits)} 个结果")

                for hit in hits:
                    doc_id = hit.get("_id")
                    if not doc_id:
                        continue
                    key = (index_name, doc_id)
                    score = float(hit.get("_score", 0.0) or 0.0)

                    if verbose and len(merged_hits) < 3:  # 只打印前3个
                        source = hit.get("_source", {})
                        label = source.get("label", "N/A")
                        print(f"         - 文档ID: {doc_id}, 分数: {score:.4f}, 标签: {label}")

                    # 如果同一个文档被多个字段命中，保留最高分
                    if key not in merged_hits or score > merged_hits[key]["score"]:
                        merged_hits[key] = {
                            "source": hit.get("_source", {}),
                            "score": score,
                            "lang": lang_tag,
                            "field_type": field_type
                        }
            except Exception as e:
                if verbose:
                    print(f"      ❌ 搜索失败: {e}")
                continue

    if not merged_hits:
        if verbose:
            print(f"\n❌ 未找到任何结果")
        return []

    if verbose:
        print(f"\n✅ 合并后共找到 {len(merged_hits)} 个唯一文档")

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

    if verbose:
        print(f"\n📊 向量检索结果（前{min(5, len(results))}个）:")
        for i, result in enumerate(results[:5], 1):
            print(f"   {i}. {result.get('label', 'N/A')}")
            print(f"      分数: {result.get('_score', 0):.4f}")
            print(f"      链接: {result.get('link', 'N/A')}")
            print(f"      描述: {result.get('descriptions_zh', '')[:80]}...")
            print()

    return results


def get_alias_and_definition(mention, verbose=True):
    """获取实体的别名、定义和详细描述（中英文各一版）"""
    # 预处理查询
    mention = preprocess_query(mention)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"🤖 LLM调用: 获取别名、定义和详细描述（中英文）")
        print(f"{'=' * 60}")
        print(f"📝 输入提及: '{mention}'")

    prompt = (
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

    if verbose:
        print(f"\n📤 发送Prompt:")
        print(f"{'-' * 60}")
        print(prompt)
        print(f"{'-' * 60}")

    try:
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )
        response_content = response.choices[0].message.content.strip()

        if verbose:
            print(f"\n📥 LLM响应:")
            print(f"{'-' * 60}")
            print(response_content)
            print(f"{'-' * 60}")

        if not response_content:
            raise ValueError(f"No response content for mention '{mention}'")

        return response_content
    except Exception as e:
        if verbose:
            print(f"\n❌ LLM调用失败: {e}")
        raise


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


def semantic_entity_match(query, description, verbose=True):
    """
    使用LLM进行精确的语义实体匹配
    
    Args:
        query: 查询文本
        description: 条目描述文本
        verbose: 是否打印详细信息
    
    Returns:
        bool: 是否匹配
    """
    prompt = f"""请判断以下描述是否属于查询实体的类别。只需回答"是"或"否"。

查询实体: "{query}"
描述文本: "{description[:500]}"

判断标准:
- 如果描述明确提到属于查询实体类别，回答"是"
- 如果描述是关于查询实体类别的具体实例，回答"是"  
- 如果描述与查询实体类别相关但不属于，回答"否"
- 如果描述不相关，回答"否"

答案: """

    try:
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10
        )
        answer = response.choices[0].message.content.strip().lower()

        if verbose:
            print(f"   🤖 LLM判断: '{answer}'")

        return "是" in answer or "yes" in answer or "true" in answer
    except Exception as e:
        if verbose:
            print(f"   ❌ LLM匹配失败: {e}")
        return False


def is_entity_match(query_text, entry_description, verbose=True):
    """
    判断条目是否属于查询实体类别（语义匹配）
    
    使用多级匹配策略：
    1. 关键词匹配：提取查询中的核心实体词，检查是否在描述中出现
    2. 向量相似度匹配：快速近似匹配
    3. LLM语义匹配：精确但较慢的匹配
    
    Args:
        query_text: 查询文本（如"阿利·伯克Flight Ⅲ"）
        entry_description: 条目描述文本
        verbose: 是否打印详细信息
    
    Returns:
        bool: 是否匹配
    """
    # 预处理文本
    query_text = preprocess_query(query_text)
    entry_description = preprocess_query(entry_description)

    if verbose:
        print(f"\n🔍 语义匹配检查:")
        print(f"   查询: '{query_text}'")
        print(f"   条目描述: '{entry_description[:200]}...'")

    query_lower = query_text.lower()
    desc_lower = entry_description.lower()

    # 1. 直接关键词匹配：提取查询中的核心实体词
    # 移除常见修饰词，提取核心实体名称
    import re
    
    # 提取可能的实体关键词（中文和英文）
    # 匹配中文字符、英文单词、数字、连字符等
    entity_patterns = [
        r'[\u4e00-\u9fa5]+',  # 中文字符
        r'[A-Z][a-zA-Z\s-]+',  # 英文专有名词（首字母大写）
        r'[A-Z]+[0-9]+',  # 型号（如P226, OH-58D）
    ]
    
    extracted_terms = []
    for pattern in entity_patterns:
        matches = re.findall(pattern, query_text)
        for match in matches:
            match_clean = match.strip()
            # 过滤太短的词（少于2个字符）和常见修饰词
            if len(match_clean) >= 2 and match_clean.lower() not in ['级', '型', '号', '的', 'the', 'a', 'an']:
                extracted_terms.append(match_clean)
    
    # 检查提取的关键词是否在描述中出现
    for term in extracted_terms:
        term_lower = term.lower()
        # 如果关键词在查询和描述中都出现，且长度>=3（避免太短的词误匹配）
        if len(term) >= 3 and term_lower in query_lower and term_lower in desc_lower:
            if verbose:
                print(f"   ✅ 关键词匹配: '{term}' (在查询和描述中都出现)")
            return True
    
    # 2. 基于向量相似度的匹配（快速近似，优先使用）
    if model is not None and tokenizer is not None:
        try:
            query_vec = generate_vector(query_text, use_cache=True, verbose=False)
            desc_vec = generate_vector(entry_description[:500], use_cache=True, verbose=False)

            if query_vec and desc_vec:
                similarity = np.dot(query_vec, desc_vec)
                if verbose:
                    print(f"   📊 向量相似度: {similarity:.4f}")

                if similarity > 0.75:  # 提高阈值到0.75，更严格
                    if verbose:
                        print(f"   ✅ 向量相似度匹配 (>{0.75})")
                    return True
        except Exception as e:
            if verbose:
                print(f"   ⚠️  向量匹配失败: {e}")

    # 3. 使用LLM进行语义匹配（更精确但较慢，作为最后手段）
    if len(entry_description) > 50:  # 只有描述足够长时才使用LLM
        try:
            return semantic_entity_match(query_text, entry_description, verbose)
        except Exception as e:
            if verbose:
                print(f"   ⚠️  LLM语义匹配失败: {e}")

    if verbose:
        print(f"   ❌ 未匹配")
    return False


def check_query_hit(query_text, entry_descriptions, verbose=True):
    """
    检查查询是否命中条目（支持语义匹配）
    
    Args:
        query_text: 查询文本
        entry_descriptions: 条目描述列表（可以是字符串或字典列表）
        verbose: 是否打印详细信息
    
    Returns:
        bool: 是否命中
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"🎯 语义命中检查: '{query_text}'")
        print(f"{'=' * 60}")

    # 统一处理条目描述
    descriptions = []
    if isinstance(entry_descriptions, str):
        descriptions = [entry_descriptions]
    elif isinstance(entry_descriptions, list):
        if entry_descriptions and isinstance(entry_descriptions[0], dict):
            # 从字典中提取描述文本
            for entry in entry_descriptions:
                desc = entry.get('descriptions_zh', '') or entry.get('descriptions_en', '') or entry.get('label', '')
                if desc:
                    descriptions.append(desc)
        else:
            descriptions = entry_descriptions

    if not descriptions:
        if verbose:
            print("   ❌ 无有效描述可检查")
        return False

    # 检查每个描述（只检查前10个，避免太慢）
    hit_count = 0
    check_count = min(10, len(descriptions))
    for i, desc in enumerate(descriptions[:check_count]):
        if is_entity_match(query_text, desc, verbose=verbose and i < 3):  # 只详细打印前3个
            hit_count += 1
            if verbose:
                print(f"   ✅ 命中条目 {i+1}")

    is_hit = hit_count > 0

    if verbose:
        print(f"\n📊 命中统计: {hit_count}/{check_count} (检查前{check_count}个)")
        print(f"🎯 最终结果: {'✅ 命中' if is_hit else '❌ 未命中'}")

    return is_hit


def ensure_links_match(sorted_links, original_links, verbose=True):
    """确保排序后的链接与原始链接一致，支持模糊匹配"""
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"🔗 链接匹配和验证")
        print(f"{'=' * 60}")
        print(f"原始链接数量: {len(original_links)}")
        print(f"LLM返回链接数量: {len(sorted_links)}")

    cleaned_sorted = [clean_link(link) for link in sorted_links]
    original_links_set = set(original_links)

    cleaned_to_original = {}
    for orig_link in original_links:
        cleaned = clean_link(orig_link)
        cleaned_to_original[cleaned] = orig_link

    matched_links = []
    used_original_links = set()
    match_info = []

    for i, cleaned_link in enumerate(cleaned_sorted):
        matched = False
        match_type = None

        # 精确匹配
        if cleaned_link in cleaned_to_original:
            orig_link = cleaned_to_original[cleaned_link]
            if orig_link not in used_original_links:
                matched_links.append(orig_link)
                used_original_links.add(orig_link)
                matched = True
                match_type = "精确匹配"

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
                        match_type = "URL归一化匹配"
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
                        match_type = "模糊匹配"
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
                        match_type = "归一化模糊匹配"
                        break

        if verbose and i < 5:  # 只打印前5个匹配信息
            if matched:
                print(f"   {i + 1}. ✅ {match_type}: '{cleaned_link[:50]}...' -> 已匹配")
            else:
                print(f"   {i + 1}. ❌ 未匹配: '{cleaned_link[:50]}...'")

    # 添加未匹配的原始链接
    unmatched_count = 0
    for orig_link in original_links:
        if orig_link not in used_original_links:
            matched_links.append(orig_link)
            unmatched_count += 1

    if verbose:
        print(f"\n✅ 匹配完成: {len(matched_links)} 个链接（其中 {unmatched_count} 个未匹配的原始链接）")

    return matched_links


def generate_prompt_and_sort_with_description(mention, results, verbose=True):
    """
    使用LLM重排序，重点使用完整的描述信息进行匹配

    Args:
        mention: 查询提及
        results: 向量检索结果列表
        verbose: 是否打印详细信息
    """
    # 预处理查询
    mention = preprocess_query(mention)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"🤖 LLM重排序: '{mention}'")
        print(f"{'=' * 60}")
        print(f"📊 输入结果数量: {len(results)}")

    input_label = mention
    response_content = ""

    try:
        response_content = get_alias_and_definition(mention, verbose=verbose)

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

        if verbose:
            print(f"\n📋 解析结果:")
            print(f"   标签: {input_label}")
            print(f"   中文别名: {input_aliases_zh if input_aliases_zh else '无'}")
            print(f"   英文别名: {input_aliases_en if input_aliases_en else '无'}")
            print(f"   中文定义: {input_definition_zh if input_definition_zh else '无'}")
            print(f"   英文定义: {input_definition_en if input_definition_en else '无'}")
            print(f"   中文详细描述: {input_description_zh[:100] if input_description_zh else '无'}...")
            print(f"   英文详细描述: {input_description_en[:100] if input_description_en else '无'}...")

        if not input_aliases_zh and not input_aliases_en and not input_definition_zh and not input_definition_en and not input_description_zh and not input_description_en:
            raise ValueError("无法从LLM响应中提取任何有效字段")

    except (ValueError, IndexError, Exception) as e:
        if verbose:
            print(f"\n❌ LLM解析失败: {e}")
            print(f"   回退到原始顺序")
        return [result['link'] for result in results]

    # 构建选项列表，确保包含完整的描述信息
    options = []
    original_links = []

    if verbose:
        print(f"\n📝 构建选项列表...")

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

        if verbose and idx <= 3:  # 只打印前3个选项
            print(f"\n   选项{idx}:")
            print(f"      标签: {result.get('label', '')}")
            print(f"      中文描述: {descriptions_zh[:100]}...")
            print(f"      英文描述: {descriptions_en[:100]}...")
            print(f"      链接: {result.get('link', '')}")

    # 判断输入是类别还是实例
    is_class_query = any(keyword in input_label.lower() for keyword in ['级', 'class', '型', '系列', 'series'])
    if not is_class_query:
        # 检查是否包含具体实例标识（如DDG-88, OH-58D等）
        has_instance_id = bool(re.search(r'[A-Z]+[-_]?\d+', input_label) or 
                              re.search(r'[A-Z]{2,3}-\d+', input_label))
        is_class_query = not has_instance_id
    
    # 构建prompt，明确强调要使用描述信息进行匹配（中英文），并优先选择类别页面
    class_instruction = ""
    if is_class_query:
        class_instruction = (
            f"【关键判断】根据输入信息分析，这是一个关于**类别/级别**的查询（如'阿利·伯克级驱逐舰'、'P226手枪'等），"
            f"而不是具体某艘舰艇或某把枪的查询。\n\n"
            f"【排序优先级】请严格按照以下优先级排序：\n"
            f"1. **最高优先级**：类别/级别的总页面（描述整个级别/系列的特征、历史、技术参数、发展历程等）\n"
            f"2. **次优先级**：属于该类别的具体实例页面（如具体某艘舰艇、某把枪的页面）\n"
            f"3. **最低优先级**：相关但不完全匹配的页面\n\n"
            f"【识别类别页面的特征】类别页面通常包含以下特征：\n"
            f"- 描述中使用'级'、'class'、'系列'、'series'等词汇\n"
            f"- 描述整个级别/系列的发展历史、技术特点、生产情况\n"
            f"- 标签(label)通常是类别名称，而不是具体舰艇/武器的名称\n"
            f"- 描述中会提到'该级'、'该系列'、'该型'等词汇\n\n"
        )
    else:
        class_instruction = (
            f"【关键判断】根据输入信息分析，这是一个关于**具体实例**的查询（如'USS Preble (DDG-88)'等），"
            f"应优先选择对应的具体实例页面。\n\n"
        )
    
    prompt = (
        f"现在你是军事领域专家，需要根据输入信息与选项列表的候选的匹配度进行从高到低排序。\n\n"
        f"【重要提示1：描述信息优先】请重点参考每个选项的完整描述信息（包括中文描述descriptions_zh和英文描述descriptions_en）进行匹配度判断，"
        f"描述信息包含了实体的详细特征和定义，比标签和别名更能准确反映实体的本质特征。在判断匹配度时，描述信息的权重应该高于标签和别名。\n\n"
        f"{class_instruction}"
        f"【重要提示2：匹配度判断】在判断匹配度时，请综合考虑：\n"
        f"- 标签和别名是否与输入信息匹配\n"
        f"- 描述信息是否与输入信息的定义和详细描述匹配\n"
        f"- 如果是类别查询，描述中是否明确提到属于该类别\n\n"
        f"输入信息：\n"
        f"  标签名：{input_label}\n"
        f"  中文别名：{input_aliases_zh if input_aliases_zh else '无'}\n"
        f"  英文别名：{input_aliases_en if input_aliases_en else '无'}\n"
        f"  中文定义：{input_definition_zh if input_definition_zh else '无'}\n"
        f"  英文定义：{input_definition_en if input_definition_en else '无'}\n"
        f"  中文详细描述：{input_description_zh if input_description_zh else '无'}\n"
        f"  英文详细描述：{input_description_en if input_description_en else '无'}\n\n"
        f"选项列表：\n"
        f"{''.join(options)}\n\n"
        f"请根据输入信息与选项的匹配度（特别关注中英文描述信息的匹配度，以及类别vs实例的区分），从高到低严格返回所有候选的link值。\n"
        f"【重要要求】\n"
        f"1. 必须返回所有{len(options)}个选项的link值，不能有缺失\n"
        f"2. 每个link值只能出现一次，不能有重复\n"
        f"3. 只返回link值，每行一个，不要解释或附加内容\n"
        f"4. 确保返回的link值完全匹配选项列表中的link值\n"
        f"5. 如果输入是类别/级别，优先将类别页面排在前面；如果输入是具体实例，优先将对应实例页面排在前面"
    )

    if verbose:
        print(f"\n📤 发送重排序Prompt（长度: {len(prompt)} 字符）")
        print(f"{'-' * 60}")
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        print(f"{'-' * 60}")

    try:
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = response.choices[0].message.content.strip()

        if verbose:
            print(f"\n📥 LLM重排序响应:")
            print(f"{'-' * 60}")
            print(response_text)
            print(f"{'-' * 60}")

        sorted_links_raw = [line.strip() for line in response_text.split("\n") if line.strip()]
        
        # 去重：保留第一次出现的链接
        seen = set()
        sorted_links_dedup = []
        for link in sorted_links_raw:
            link_normalized = normalize_url(clean_link(link))
            if link_normalized not in seen:
                seen.add(link_normalized)
                sorted_links_dedup.append(link)

        if verbose:
            if len(sorted_links_raw) != len(sorted_links_dedup):
                print(f"\n⚠️  LLM返回了 {len(sorted_links_raw)} 个链接，去重后为 {len(sorted_links_dedup)} 个")
            print(f"\n📋 解析后的链接列表（前5个）:")
            for i, link in enumerate(sorted_links_dedup[:5], 1):
                print(f"   {i}. {link[:80]}...")

        sorted_links = ensure_links_match(sorted_links_dedup, original_links, verbose=verbose)

        if verbose:
            print(f"\n✅ 最终排序结果（前5个）:")
            for i, link in enumerate(sorted_links[:5], 1):
                print(f"   {i}. {link[:80]}...")

        return sorted_links
    except Exception as e:
        if verbose:
            print(f"\n❌ LLM排序失败: {e}")
            import traceback
            traceback.print_exc()
        return original_links


def find_rank(correct_link, sorted_links, verbose=True):
    """查找正确链接在排序列表中的排名"""
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"🎯 查找正确链接排名")
        print(f"{'=' * 60}")
        print(f"正确链接: {correct_link}")

    rank = None
    correct_link_cleaned = clean_link(str(correct_link))
    correct_link_normalized = normalize_url(correct_link_cleaned)

    if verbose:
        print(f"清理后: {correct_link_cleaned}")
        print(f"归一化后: {correct_link_normalized}")
        print(f"\n开始匹配检查...")

    for i, link in enumerate(sorted_links):
        link_cleaned = clean_link(str(link))
        link_normalized = normalize_url(link_cleaned)

        # 多种匹配方式
        matched = False
        match_type = None

        # 1. 归一化后的URL匹配
        if correct_link_normalized == link_normalized:
            rank = i + 1
            matched = True
            match_type = "URL归一化匹配"

        # 2. 清理后的精确匹配
        if not matched and correct_link_cleaned == link_cleaned:
            rank = i + 1
            matched = True
            match_type = "精确匹配"

        # 3. 双向子字符串匹配
        if not matched and (correct_link_cleaned in link_cleaned or link_cleaned in correct_link_cleaned):
            rank = i + 1
            matched = True
            match_type = "子字符串匹配"

        # 4. 归一化后的双向匹配
        if not matched and (correct_link_normalized in link_normalized or link_normalized in correct_link_normalized):
            rank = i + 1
            matched = True
            match_type = "归一化子字符串匹配"

        if matched:
            if verbose:
                print(f"\n✅ 找到匹配！")
                print(f"   排名: {rank}")
                print(f"   匹配类型: {match_type}")
                print(f"   匹配链接: {link}")
            break

        if verbose and i < 10:  # 打印前10个尝试
            print(f"   位置 {i + 1}: '{link[:80]}...' - 不匹配")
            print(f"      归一化后: '{link_normalized[:60]}...'")

    if rank is None:
        if verbose:
            print(f"\n❌ 未找到匹配")
            print(f"   已检查 {len(sorted_links)} 个链接")
            print(f"   正确答案归一化后: '{correct_link_normalized}'")
            print(f"\n   排序列表中的所有链接（前20个）:")
            for i, link in enumerate(sorted_links[:20], 1):
                link_norm = normalize_url(clean_link(link))
                print(f"      {i}. {link}")
                print(f"         归一化: {link_norm}")

    return rank


def process_single_query(query, correct_link, use_llm=True, verbose=True):
    """处理单个查询"""
    print(f"\n{'#' * 80}")
    print(f"# 处理查询: '{query}'")
    print(f"# 正确答案: '{correct_link}'")
    print(f"# 使用LLM重排序: {use_llm}")
    print(f"{'#' * 80}")

    # 1. 向量检索（使用所有向量字段）
    results = vector_search(query, top_k=30, verbose=verbose)  # 增加到30，同时检索所有7个向量字段，给LLM更多候选

    if not results:
        print(f"\n❌ 向量检索未找到结果")
        return None, 0, 0, 0, 0

    # 2. LLM重排序（如果启用）
    if use_llm:
        sorted_links = generate_prompt_and_sort_with_description(query, results, verbose=verbose)
    else:
        sorted_links = [r.get("link", "") for r in results]
        if verbose:
            print(f"\n📋 直接使用向量检索顺序（不使用LLM重排序）")
            print(f"   前5个链接:")
            for i, link in enumerate(sorted_links[:5], 1):
                print(f"      {i}. {link[:80]}...")

    # 3. 检查正确答案是否在ES中，并计算向量相似度（调试用）
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"🔍 检查正确答案是否在ES中并计算向量相似度")
        print(f"{'=' * 60}")
        correct_link_normalized = normalize_url(clean_link(str(correct_link)))
        found_in_es = False
        found_in_results = False
        
        # 生成查询向量（用于计算相似度）
        query_vec = None
        if model is not None and tokenizer is not None:
            try:
                query_vec = generate_vector(query, use_cache=True, verbose=False)
            except Exception as e:
                print(f"⚠️  生成查询向量失败: {e}")
        
        # 尝试在ES中直接搜索这个链接（使用多种方式匹配）
        try:
            # 方法1: 使用term查询精确匹配（如果link字段有keyword子字段）
            # 显式指定要返回的向量字段
            vector_fields_list = [
                "descriptions_zh_vector", "descriptions_en_vector",
                "high_freq_words_zh_vector", "high_freq_words_en_vector",
                "label_vector", "label_zh_vector", "label_en_vector"
            ]
            search_query = {
                "query": {
                    "term": {
                        "link.keyword": correct_link
                    }
                },
                "_source": ["label", "link"] + vector_fields_list,  # 显式指定要返回的字段
                "size": 10  # 多返回一些，用于匹配
            }
            resp = es.search(index="data2", body=search_query)
            hits = resp.get("hits", {}).get("hits", [])
            
            # 方法2: 如果没找到，尝试match_phrase查询
            if not hits:
                search_query = {
                    "query": {
                        "match_phrase": {
                            "link": correct_link
                        }
                    },
                    "_source": ["label", "link"] + vector_fields_list,  # 显式指定要返回的字段
                    "size": 10
                }
                resp = es.search(index="data2", body=search_query)
                hits = resp.get("hits", {}).get("hits", [])
            
            # 方法3: 如果还没找到，尝试使用归一化后的URL进行匹配
            if not hits:
                # 获取所有文档，然后手动匹配（因为ES可能存储的是编码后的URL）
                search_query = {
                    "query": {"match_all": {}},
                    "_source": ["label", "link"] + vector_fields_list,  # 显式指定要返回的字段
                    "size": 1000  # 限制数量，避免查询太慢
                }
                resp = es.search(index="data2", body=search_query)
                all_hits = resp.get("hits", {}).get("hits", [])
                
                # 手动匹配归一化后的URL
                for hit in all_hits:
                    source = hit.get("_source", {})
                    doc_link = source.get("link", "")
                    if doc_link:
                        doc_link_normalized = normalize_url(clean_link(str(doc_link)))
                        if correct_link_normalized == doc_link_normalized:
                            hits = [hit]
                            break
            if hits:
                found_in_es = True
                hit = hits[0]
                source = hit.get("_source", {})
                print(f"✅ 正确答案在ES中找到:")
                print(f"   标签: {source.get('label', 'N/A')}")
                print(f"   链接: {source.get('link', 'N/A')}")
                
                # 计算向量相似度（语义对齐检查）
                if query_vec:
                    print(f"\n📊 向量相似度分析（语义对齐检查）:")
                    print(f"   查询文本: '{query}'")
                    
                    # 检查各个向量字段的相似度
                    vector_fields_to_check = [
                        ("descriptions_zh_vector", "中文描述向量"),
                        ("descriptions_en_vector", "英文描述向量"),
                        ("high_freq_words_zh_vector", "中文高频词向量"),
                        ("high_freq_words_en_vector", "英文高频词向量"),
                        ("label_vector", "标签向量"),
                        ("label_zh_vector", "中文标签向量"),
                        ("label_en_vector", "英文标签向量")
                    ]
                    
                    max_similarity = 0
                    best_field = None
                    
                    for field_name, field_desc in vector_fields_to_check:
                        doc_vector = source.get(field_name)
                        if doc_vector and isinstance(doc_vector, list) and len(doc_vector) == len(query_vec):
                            # 计算余弦相似度（点积，因为都已归一化）
                            similarity = np.dot(query_vec, doc_vector)
                            if similarity > max_similarity:
                                max_similarity = similarity
                                best_field = field_name
                            print(f"   {field_desc}: {similarity:.4f}")
                        else:
                            print(f"   {field_desc}: 无向量数据")
                    
                    if best_field:
                        print(f"\n   ✅ 最高相似度: {max_similarity:.4f} (字段: {best_field})")
                        if max_similarity < 0.7:
                            print(f"   ⚠️  相似度较低（<0.7），可能导致检索效果不佳")
                            print(f"   建议：检查查询文本和索引文本的格式是否一致")
                
                # 检查是否在检索结果中
                for i, result in enumerate(results):
                    result_link = result.get("link", "")
                    result_link_normalized = normalize_url(clean_link(str(result_link)))
                    if correct_link_normalized == result_link_normalized:
                        found_in_results = True
                        print(f"\n✅ 正确答案在检索结果中（位置: {i+1}, 分数: {result.get('_score', 0):.4f}）")
                        break
                
                if not found_in_results:
                    print(f"\n⚠️  正确答案在ES中，但不在top_{len(results)}检索结果中")
                    if query_vec and max_similarity < 0.7:
                        print(f"   可能原因：向量相似度较低（{max_similarity:.4f}），正确答案的相似度分数可能低于其他文档")
            else:
                print(f"❌ 正确答案不在ES索引中")
        except Exception as e:
            print(f"⚠️  检查ES时出错: {e}")
            import traceback
            traceback.print_exc()

    # 4. 语义命中检查
    semantic_hit = False
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"🔍 语义命中检查")
        print(f"{'=' * 60}")
    
    # 使用新的语义匹配逻辑
    try:
        semantic_hit = check_query_hit(query, results, verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"⚠️  语义匹配检查失败: {e}")
        semantic_hit = False

    # 5. 查找排名
    rank = find_rank(correct_link, sorted_links, verbose=verbose)

    # 6. 计算指标
    if rank is not None:
        mrr = 1 / rank
        hit_at_1 = 1 if rank <= 1 else 0
        hit_at_5 = 1 if rank <= 5 else 0
        hit_at_10 = 1 if rank <= 10 else 0
    else:
        mrr = 0
        hit_at_1 = 0
        hit_at_5 = 0
        hit_at_10 = 0

    # 7. 显示结果
    print(f"\n{'=' * 80}")
    print(f"📊 评估结果:")
    print(f"{'=' * 80}")
    print(f"   查询: {query}")
    print(f"   正确答案: {correct_link}")
    print(f"   排名: {rank if rank else '未找到'}")
    print(f"   语义命中: {'✅ 是' if semantic_hit else '❌ 否'}")
    print(f"   MRR: {mrr:.4f}")
    print(f"   Hit@1: {hit_at_1}")
    print(f"   Hit@5: {hit_at_5}")
    print(f"   Hit@10: {hit_at_10}")
    print(f"{'=' * 80}")

    return {
        "query": query,
        "correct_link": correct_link,
        "rank": rank,
        "semantic_hit": 1 if semantic_hit else 0,  # 新增字段
        "mrr": mrr,
        "hit@1": hit_at_1,
        "hit@5": hit_at_5,
        "hit@10": hit_at_10,
        "sorted_links": sorted_links[:10]
    }, mrr, hit_at_1, hit_at_5, hit_at_10


def read_excel(file_path, max_rows=5):
    """读取Excel测试集文件（限制行数用于调试）"""
    df = pd.read_excel(file_path, header=None)
    queries = df[0].tolist()[:max_rows]
    correct_links = df[1].tolist()[:max_rows]
    print(f"📖 读取测试集: {len(queries)} 个查询（限制为前{max_rows}条）")
    return queries, correct_links


def main():
    """主函数：使用少量测试集验证流程"""
    import sys

    file_path = "data/find.xlsx"
    max_test_rows = 5  # 只测试前5条

    # 检查命令行参数
    use_llm = True
    if len(sys.argv) > 1:
        if sys.argv[1] == "--no-llm":
            use_llm = False
        elif sys.argv[1].startswith("--rows="):
            max_test_rows = int(sys.argv[1].split("=")[1])

    if not os.path.exists(file_path):
        print(f"❌ 未找到评测文件: {file_path}")
        print("请确保测试文件存在")
        return

    # 重定向stdout到文件和控制台
    class DualOutput:
        def __init__(self, filepath):
            self.terminal = sys.__stdout__  # 使用原始stdout
            self.log_file = open(filepath, "w", encoding="utf-8")
            self._closed = False

        def write(self, message):
            if not self._closed:
                try:
                    self.terminal.write(message)
                    self.log_file.write(message)
                    self.terminal.flush()
                    self.log_file.flush()
                except Exception:
                    pass  # 忽略写入错误

        def flush(self):
            if not self._closed:
                try:
                    self.terminal.flush()
                    self.log_file.flush()
                except Exception:
                    pass  # 忽略刷新错误

        def close(self):
            if not self._closed:
                try:
                    self.log_file.close()
                    self._closed = True
                except Exception:
                    pass  # 忽略关闭错误

        def is_closed(self):
            return self._closed

    # 创建输出文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(output_dir, f"debug_output_{timestamp}.md")
    
    # 创建双输出对象
    dual_output = DualOutput(output_filename)
    original_stdout = sys.stdout
    sys.stdout = dual_output

    try:
        queries, correct_links = read_excel(file_path, max_rows=max_test_rows)

        print(f"\n{'=' * 80}")
        print(f"🚀 开始调试测试")
        print(f"{'=' * 80}")
        print(f"测试查询数量: {len(queries)}")
        print(f"使用LLM重排序: {use_llm}")
        print(f"{'=' * 80}\n")

        all_results = []
        total_mrr = 0
        total_hit1 = 0
        total_hit5 = 0
        total_hit10 = 0
        total_semantic_hit = 0

        for i, (query, correct_link) in enumerate(zip(queries, correct_links), 1):
            print(f"\n\n{'=' * 80}")
            print(f"查询 {i}/{len(queries)}")
            print(f"{'=' * 80}")

            result, mrr, hit1, hit5, hit10 = process_single_query(
                query, correct_link, use_llm=use_llm, verbose=True
            )

            if result:
                all_results.append(result)
                total_mrr += mrr
                total_hit1 += hit1
                total_hit5 += hit5
                total_hit10 += hit10
                total_semantic_hit += result.get("semantic_hit", 0)

        # 计算平均指标
        if len(all_results) > 0:
            avg_mrr = total_mrr / len(all_results)
            avg_hit1 = total_hit1 / len(all_results)
            avg_hit5 = total_hit5 / len(all_results)
            avg_hit10 = total_hit10 / len(all_results)
            avg_semantic_hit = total_semantic_hit / len(all_results)

            print(f"\n\n{'=' * 80}")
            print(f"📊 总体评估结果")
            print(f"{'=' * 80}")
            print(f"测试查询数量: {len(all_results)}")
            print(f"平均 MRR: {avg_mrr:.4f}")
            print(f"平均 Hit@1: {avg_hit1:.4f}")
            print(f"平均 Hit@5: {avg_hit5:.4f}")
            print(f"平均 Hit@10: {avg_hit10:.4f}")
            print(f"平均语义命中率: {avg_semantic_hit:.4f}")
            print(f"{'=' * 80}")

            # 保存结果
            report = {
                "timestamp": datetime.now().isoformat(),
                "test_mode": "debug",
                "use_llm": use_llm,
                "total_queries": len(all_results),
                "metrics": {
                    "mrr": avg_mrr,
                    "hit@1": avg_hit1,
                    "hit@5": avg_hit5,
                    "hit@10": avg_hit10,
                    "semantic_hit_rate": avg_semantic_hit
                },
                "detailed_results": all_results
            }

            filename = os.path.join(output_dir, f'debug_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            print(f"\n✅ 详细报告已保存到: {filename}")
        else:
            print(f"\n❌ 没有成功处理的查询")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 恢复原始stdout
        sys.stdout = original_stdout
        # 关闭文件
        dual_output.close()
        print(f"\n✅ 控制台输出已保存到: {output_filename}")

if __name__ == "__main__":
    main()

