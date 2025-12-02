import torch
from transformers import BertTokenizer, BertModel
from zhipuai import ZhipuAI
import numpy as np
import re
from urllib.parse import unquote
from work_wyy.es_client import es
import logging

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vector_search_test.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
    vector_dim = len(vector)
    target_dim = 1024
    
    if vector_dim == target_dim:
        pass
    elif vector_dim < target_dim:
        vector = np.pad(vector, (0, target_dim - vector_dim), 'constant', constant_values=0)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
    else:
        vector = vector[:target_dim]
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
    
    vector_list = vector.tolist()
    
    # 添加到缓存
    if use_cache:
        if len(_vector_cache) >= _cache_max_size:
            first_key = next(iter(_vector_cache))
            del _vector_cache[first_key]
        _vector_cache[cache_key] = vector_list
    
    return vector_list

def vector_search(query_text, top_k=20, query_vector=None):
    """
    单独的向量检索
    
    Args:
        query_text: 查询文本
        top_k: 返回结果数量
        query_vector: 预计算的查询向量（可选）
    
    Returns:
        检索结果列表
    """
    # 生成查询向量
    if query_vector is None and model is not None and tokenizer is not None:
        try:
            query_vector = generate_vector(query_text, use_cache=True)
        except Exception:
            query_vector = None
    
    if query_vector is None:
        return []
    
    # 确定要搜索的向量字段
    vector_fields = [
        ("descriptions_zh_vector", "zh", "desc"),
        ("descriptions_en_vector", "en", "desc")
    ]
    
    # 同时对多个向量字段做检索，然后融合结果
    index_names = ["data2", "data1"]
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
            "link": source.get("link", ""),
            "_score": item["score"],
            "_lang": item["lang"],
            "_field_type": item["field_type"]
        }
        results.append(result)
        
    return results

def get_alias_and_definition(mention):
    """获取实体的别名和定义"""
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

def generate_prompt_and_sort_with_description(mention, results):
    """
    使用LLM重排序，重点使用完整的描述信息进行匹配
    
    Args:
        mention: 查询提及
        results: 向量检索结果列表
    
    Returns:
        排序后的链接列表
    """
    input_label = mention
    response_content = ""
    
    try:
        response_content = get_alias_and_definition(mention)
        
        # 安全提取字段内容
        def safe_extract(content, field_name, default=""):
            if f"{field_name}：" in content:
                parts = content.split(f"{field_name}：", 1)
                if len(parts) > 1:
                    value = parts[1].split("英文别名")[0].split("定义")[0].split("\n")[0].strip()
                    return value if value else default
            elif field_name in content:
                idx = content.find(field_name)
                if idx != -1:
                    start = idx + len(field_name)
                    while start < len(content) and content[start] in [":", "：", " ", "\t"]:
                        start += 1
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
        
        if not input_aliases_zh and not input_aliases_en and not input_definition:
            raise ValueError("无法从LLM响应中提取任何有效字段")
            
    except (ValueError, IndexError, Exception) as e:
        logger.warning(f"LLM解析失败 for mention '{mention}': {e}")
        return [result['link'] for result in results]

    # 构建选项列表，确保包含完整的描述信息
    options = []
    original_links = []
    
    for idx, result in enumerate(results, start=1):
        # 获取完整的描述信息
        descriptions_zh = result.get('descriptions_zh', '')
        if not descriptions_zh:
            descriptions_zh = "（无描述信息）"
        
        # 构建选项，重点展示描述信息
        option = (
            f"选项{idx}：\n"
            f"标签(label): {result.get('label', '')}\n"
            f"中文别名(aliases_zh): {', '.join(result.get('aliases_zh', [])) if result.get('aliases_zh') else '无'}\n"
            f"英文别名(aliases_en): {', '.join(result.get('aliases_en', [])) if result.get('aliases_en') else '无'}\n"
            f"完整描述(descriptions_zh): {descriptions_zh}\n"
            f"链接(link): {result.get('link', '')}\n"
        )
        options.append(option)
        original_links.append(result.get('link', ''))

    # 构建prompt，明确强调要使用描述信息进行匹配
    prompt = (
        f"现在你是军事领域专家，需要根据输入信息与选项列表的候选的匹配度进行从高到低排序。\n\n"
        f"【重要提示】请重点参考每个选项的完整描述(descriptions_zh)信息进行匹配度判断，描述信息包含了实体的详细特征和定义，"
        f"比标签和别名更能准确反映实体的本质特征。在判断匹配度时，描述信息的权重应该高于标签和别名。\n\n"
        f"输入信息：\n"
        f"  标签名：{input_label}\n"
        f"  中文别名：{input_aliases_zh if input_aliases_zh else '无'}\n"
        f"  英文别名：{input_aliases_en if input_aliases_en else '无'}\n"
        f"  定义：{input_definition if input_definition else '无'}\n\n"
        f"选项列表：\n"
        f"{''.join(options)}\n\n"
        f"请根据输入信息与选项的匹配度（特别关注描述信息的匹配度），从高到低严格返回所有候选的link值，"
        f"确保返回的link值是原始选项列表中的link值的排序，不能有缺失或重复，不要解释或附加内容。"
    )

    try:
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = response.choices[0].message.content.strip()
        sorted_links_raw = [line.strip() for line in response_text.split("\n") if line.strip()]
        sorted_links = ensure_links_match(sorted_links_raw, original_links)
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
    print(f"\n{'='*60}")
    print(f"测试向量检索（无LLM重排序）")
    print(f"{'='*60}")
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
    print(f"\n{'='*60}")
    print(f"测试向量检索 + LLM重排序（使用完整描述信息）")
    print(f"{'='*60}")
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

def main():
    """主函数：测试向量检索和向量+LLM重排序"""
    print("=" * 60)
    print("向量检索测试工具")
    print("=" * 60)
    print("\n测试模式：")
    print("1. 单独的向量检索（不使用LLM）")
    print("2. 向量检索 + LLM重排序（使用完整描述信息）")
    print()
    
    # 测试查询列表
    test_queries = ["AK47", "F-16战斗机", "步枪", "航空母舰", "狙击步枪"]
    
    for query in test_queries:
        # 测试1: 单独的向量检索
        test_vector_search_only(query, top_k=10)
        
        # 测试2: 向量检索 + LLM重排序
        test_vector_search_with_llm(query, top_k=10)
        
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()

