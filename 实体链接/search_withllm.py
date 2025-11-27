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

# 模型加载（用于向量生成）
# 支持两种模型：
# - chinese-roberta-wwm-ext: 768维
# - chinese-roberta-wwm-ext-large: 1024维（推荐，与ES向量字段维度匹配）
model_name = './model/chinese-roberta-wwm-ext-large'
# model_name = 'D:/model/chinese-roberta-wwm-ext-large'
# model_name = './model/chinese-roberta-wwm-ext'  # 768维模型
model = None
tokenizer = None
model_dimension = None  # 模型向量维度
try:
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()
    # 获取模型隐藏层维度
    model_dimension = model.config.hidden_size
    print(f"✓ Chinese-RoBERTa模型加载成功 (维度: {model_dimension})")
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
    
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    vector = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    
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
        cache_key = text
        if cache_key in _vector_cache:
            return _vector_cache[cache_key]
        
        # 生成向量
        vector = _generate_vector_internal(text)
        
        # 添加到缓存（如果缓存已满，删除最旧的）
        if len(_vector_cache) >= _cache_max_size:
            # 删除第一个（FIFO）
            first_key = next(iter(_vector_cache))
            del _vector_cache[first_key]
        
        _vector_cache[cache_key] = vector
        return vector
    else:
        # 不使用缓存，直接生成
        return _generate_vector_internal(text)

def hybrid_search(query_text, top_k=20, text_boost=1.0, vector_boost=0.8, use_vector=True):
    """
    混合检索：文本检索 + 向量检索
    
    Args:
        query_text: 查询文本
        top_k: 返回结果数量
        text_boost: 文本检索的boost权重（默认1.0）
        vector_boost: 向量检索的boost权重（默认0.8）
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
    
    # 尝试生成查询向量（使用LLM生成的规范化定义）
    query_vector = None
    knn_query = None
    
    if use_vector and model is not None and tokenizer is not None:
        try:
            # 使用LLM生成规范化定义
            response_content = get_alias_and_definition(query_text)
            input_definition = response_content.split("定义：")[1].strip()
            query_vector = generate_vector(input_definition, use_cache=True)
        except (ValueError, IndexError, Exception) as e:
            # 如果LLM失败，使用原始查询文本生成向量
            try:
                query_vector = generate_vector(query_text, use_cache=True)
            except Exception as e2:
                # 向量生成失败，仅使用文本检索
                query_vector = None
        
        # 如果成功生成向量，构建KNN查询
        if query_vector is not None:
            # 优先使用content_vector（包含完整页面内容）
            # 如果content_vector字段不存在或为空，ES查询会失败，需要回退到descriptions_zh_vector
            # 这里先尝试content_vector，如果失败会在ES查询时捕获异常并回退
            knn_query = {
                "field": "content_vector",  # 优先使用content_vector
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
                # 如果使用向量检索，先尝试content_vector，失败则回退到descriptions_zh_vector
                if knn_query and knn_query.get("field") == "content_vector":
                    try:
                        response = es.search(index=index_name, body=search_query)
                    except Exception as e:
                        # content_vector字段可能不存在或为空，回退到descriptions_zh_vector
                        if "content_vector" in str(e).lower() or "field" in str(e).lower():
                            # 修改knn_query使用descriptions_zh_vector
                            search_query["knn"]["field"] = "descriptions_zh_vector"
                            try:
                                response = es.search(index=index_name, body=search_query)
                            except Exception as e2:
                                # 如果descriptions_zh_vector也失败，移除knn查询，仅使用文本检索
                                search_query = {
                                    "query": text_query,
                                    "size": top_k
                                }
                                response = es.search(index=index_name, body=search_query)
                        else:
                            raise e
                else:
                    response = es.search(index=index_name, body=search_query)
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

def generate_prompt_and_sort(mention, results):
    input_label = mention
    response_content = ""
    try:
        response_content = get_alias_and_definition(mention)
        input_aliases_zh = response_content.split("中文别名：")[1].split("英文别名")[0].strip()
        input_aliases_en = response_content.split("英文别名：")[1].split("定义")[0].strip()
        input_definition = response_content.split("定义：")[1].strip()
    except (ValueError, IndexError, Exception) as e:
        print(f"LLM failed to generate valid response for mention '{mention}'. Error: {e}")
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

def calculate_metrics(queries, correct_links):
    mrr = 0
    hit_at_1 = 0
    hit_at_5 = 0
    hit_at_10 = 0
    total_processed = 0

    def process_query(query, correct_link):
        try:
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

            if rank is not None:
                return 1 / rank, 1 if rank <= 1 else 0, 1 if rank <= 5 else 0, 1 if rank <= 10 else 0
            return 0, 0, 0, 0
        except Exception as e:
            print(f"Error processing query '{query}': {e}")
            return 0, 0, 0, 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(process_query, query, correct_link) for query, correct_link in zip(queries, correct_links)]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(queries), desc="Processing queries"):
            result = future.result()
            mrr += result[0]
            hit_at_1 += result[1]
            hit_at_5 += result[2]
            hit_at_10 += result[3]
            total_processed += 1

    if total_processed > 0:
        mrr /= total_processed
        hit_at_1 /= total_processed
        hit_at_5 /= total_processed
        hit_at_10 /= total_processed
    else:
        print("No queries were processed successfully.")
        return 0, 0, 0, 0

    return mrr, hit_at_1, hit_at_5, hit_at_10

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
        print("开始评测（使用LLM重排序）...")
        mrr, hit_at_1, hit_at_5, hit_at_10 = calculate_metrics(queries, correct_links)
        print(f"\n{'='*50}")
        print("评测结果:")
        print(f"{'='*50}")
        print(f"MRR: {mrr:.4f}")
        print(f"Hit@1: {hit_at_1:.4f}")
        print(f"Hit@5: {hit_at_5:.4f}")
        print(f"Hit@10: {hit_at_10:.4f}")
    except Exception as e:
        print(f"评测失败: {e}")
        print("\n运行测试搜索模式...\n")
        test_search()

if __name__ == "__main__":
    main()