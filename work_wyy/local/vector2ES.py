import json
import os
import sys
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from elasticsearch import helpers
from tqdm import tqdm
import time
import logging
from datetime import datetime

# 导入配置和ES客户端
try:
    from config import (
        ES_INDEX_NAME,
        ENTITY_WORDS_ZH_FILE,
        ENTITY_WORDS_EN_FILE,
        TRAINLOG_DIR,
        VECTOR_DIMS,
        VECTOR_BATCH_SIZE,
        USE_FINETUNED_FOR_VECTORIZATION
    )
    from es_client import es
except ImportError:
    # 如果无法导入配置，使用默认值
    ES_INDEX_NAME = 'data2'
    ENTITY_WORDS_ZH_FILE = None
    ENTITY_WORDS_EN_FILE = None
    TRAINLOG_DIR = None
    VECTOR_DIMS = 1024
    VECTOR_BATCH_SIZE = 64
    USE_FINETUNED_FOR_VECTORIZATION = True
    from es_client import es

# 设置日志记录 - 同时输出到控制台和文件
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 清除已有的处理器
logger.handlers = []

# 文件处理器
file_handler = logging.FileHandler('vector_import_log.txt', mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# 控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# 添加处理器
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 使用统一的向量生成模块（支持微调后的模型）
VECTOR_DIMS = 1024  # ES向量字段维度

# 导入向量生成模块
try:
    import sys
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from vector_model import load_vector_model, generate_vector as _generate_vector_module, batch_generate_vectors
    
    # 加载模型（优先使用微调后的模型）
    model, tokenizer, device = load_vector_model(use_finetuned=True)
    print(f"\n✅ 向量生成模型加载成功（使用微调后的模型）")
    print(f"   使用设备: {device}")
    print(f"   预期向量维度: {VECTOR_DIMS}")
    print("=" * 60)
except Exception as e:
    print(f"⚠️  统一向量模块加载失败: {e}，尝试使用基础模型")
    try:
        model_name = './model/chinese-roberta-wwm-ext-large'
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        model.eval()
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        model = model.to(device)
        print(f"✅ 基础模型加载成功: {model_name}")
        print(f"   使用设备: {device}")
        print(f"   预期向量维度: {VECTOR_DIMS}")
        print("=" * 60)
    except Exception as e2:
        print(f"❌ 模型加载失败: {e2}")
        exit(1)

# ES索引名称（使用配置文件中的值）
INDEX_NAME = ES_INDEX_NAME


def create_vector_index():
    """创建包含向量字段的索引映射 - 包含标签、描述、实体词向量"""
    index_mapping = {
        "mappings": {
            "properties": {
                "label": {"type": "text"},
                "link": {"type": "keyword"},
                "aliases_en": {"type": "text"},
                "aliases_zh": {"type": "text"},
                "descriptions_en": {"type": "text"},
                "descriptions_zh": {"type": "text"},
                "content": {"type": "text"},
                # 标签向量
                "label_vector": {
                    "type": "dense_vector",
                    "dims": VECTOR_DIMS,
                    "index": True,
                    "similarity": "cosine"
                },
                # 中文描述向量
                "descriptions_zh_vector": {
                    "type": "dense_vector",
                    "dims": VECTOR_DIMS,
                    "index": True,
                    "similarity": "cosine"
                },
                # 英文描述向量
                "descriptions_en_vector": {
                    "type": "dense_vector",
                    "dims": VECTOR_DIMS,
                    "index": True,
                    "similarity": "cosine"
                },
                # 中文实体词向量
                "entity_words_zh_vector": {
                    "type": "dense_vector",
                    "dims": VECTOR_DIMS,
                    "index": True,
                    "similarity": "cosine"
                },
                # 英文实体词向量
                "entity_words_en_vector": {
                    "type": "dense_vector",
                    "dims": VECTOR_DIMS,
                    "index": True,
                    "similarity": "cosine"
                },
                # 保留原有字段（兼容性）
                "label_zh_vector": {
                    "type": "dense_vector",
                    "dims": VECTOR_DIMS,
                    "index": True,
                    "similarity": "cosine"
                },
                "label_en_vector": {
                    "type": "dense_vector",
                    "dims": VECTOR_DIMS,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }

    # 检查索引是否存在
    if es.indices.exists(index=INDEX_NAME):
        print(f"删除现有索引: {INDEX_NAME}")
        try:
            es.indices.delete(index=INDEX_NAME)
            time.sleep(2)
        except Exception as e:
            print(f"删除索引失败: {e}")
            return False

    # 创建新索引
    try:
        # 兼容新旧版本的 ES API
        try:
            es.indices.create(index=INDEX_NAME, body=index_mapping)
        except TypeError:
            # 新版本 API 使用 mappings 参数
            es.indices.create(index=INDEX_NAME, mappings=index_mapping.get("mappings", {}))
        print(f"成功创建向量索引: {INDEX_NAME}, 向量维度: {VECTOR_DIMS}")
        return True
    except Exception as e:
        print(f"创建索引失败: {e}")
        logger.error(f"创建索引失败详情: {e}")
        return False


def generate_vector(text):
    """生成文本向量 - 使用统一的向量生成模块（支持微调后的模型）

    注意：这个函数在模块加载时会被调用进行测试，所以不能依赖全局变量
    """
    if text and text.strip():
        try:
            # 优先使用统一的向量生成模块
            try:
                vector_list = _generate_vector_module(text, use_finetuned=True, target_dim=VECTOR_DIMS)
                if vector_list and len(vector_list) == VECTOR_DIMS:
                    return [float(x) for x in vector_list]
            except:
                pass
            
            # 回退到原始方法
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            vector = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            
            if len(vector.shape) == 0:
                vector = vector.reshape(1)
            elif len(vector.shape) > 1:
                vector = vector.flatten()
            
            actual_dims = len(vector)
            if actual_dims != VECTOR_DIMS:
                logger.warning(f"向量维度不匹配! 期望: {VECTOR_DIMS}, 实际: {actual_dims}")
                return None
            
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            else:
                return None
            
            vector_list = [float(x) for x in vector.tolist()]
            return vector_list if len(vector_list) == VECTOR_DIMS else None
            
        except Exception as e:
            logger.warning(f"向量生成失败: {e}, 文本: {text[:50] if text else 'None'}")
            return None
    return None


def generate_vectors_batch(texts, batch_size=32):
    """
    批量生成向量 - 使用统一的向量生成模块（支持微调后的模型）
    """
    # 优先使用统一的批量向量生成模块
    try:
        vectors = batch_generate_vectors(
            texts,
            use_finetuned=True,
            target_dim=VECTOR_DIMS,
            batch_size=batch_size
        )
        # 转换为float类型并验证维度
        result = []
        for vec in vectors:
            if vec and len(vec) == VECTOR_DIMS:
                result.append([float(x) for x in vec])
            else:
                result.append(None)
        return result
    except:
        pass
    
    # 回退到原始方法
    """批量生成向量 - 显著提升速度（GPU模式下可提升3-10倍）
    
    Args:
        texts: 文本列表
        batch_size: 批处理大小（GPU建议32-64，CPU建议8-16）
    
    Returns:
        vectors: 向量列表，与输入文本一一对应，失败返回None
    """
    
    if not texts:
        return []
    
    # 过滤空文本
    valid_texts = [(i, text) for i, text in enumerate(texts) if text and text.strip()]
    if not valid_texts:
        return [None] * len(texts)
    
    vectors = [None] * len(texts)
    
    # 批量处理
    for batch_start in range(0, len(valid_texts), batch_size):
        batch_end = min(batch_start + batch_size, len(valid_texts))
        batch_indices = [idx for idx, _ in valid_texts[batch_start:batch_end]]
        batch_texts = [text for _, text in valid_texts[batch_start:batch_end]]
        
        try:
            # 批量tokenize
            inputs = tokenizer(
                batch_texts, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            # 移到GPU
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # 提取[CLS] token的向量
            batch_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # 处理每个向量
            for i, (orig_idx, vector) in enumerate(zip(batch_indices, batch_vectors)):
                # 确保是一维数组
                if len(vector.shape) > 1:
                    vector = vector.flatten()
                
                # 检查维度
                if len(vector) != VECTOR_DIMS:
                    continue
                
                # L2归一化
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
                else:
                    continue
                
                # 转换为列表
                vector_list = vector.tolist()
                if len(vector_list) != VECTOR_DIMS:
                    continue
                
                # 确保是float类型
                vectors[orig_idx] = [float(x) for x in vector_list]
        
        except Exception as e:
            logger.warning(f"批量向量生成失败: {e}")
            # 如果批量失败，尝试逐个生成
            for orig_idx, text in zip(batch_indices, batch_texts):
                vectors[orig_idx] = generate_vector(text)
    
    return vectors


def process_single_item(item, use_batch=False, vector_cache=None):
    """处理单条数据项 - 生成标签、描述、实体词的向量
    
    注意：ES中存储的字段名与源文件一致：entity_words_zh_vector 和 entity_words_en_vector
    这些字段存储的是NER提取的实体词向量。
    
    Args:
        item: 数据项
        use_batch: 是否使用批量处理（需要外部调用批量函数）
        vector_cache: 向量缓存字典，用于存储批量生成的向量
    """
    # 首次调用时测试向量生成（只测试一次）
    if not hasattr(process_single_item, '_tested'):
        test_vector = generate_vector("测试")
        if test_vector:
            actual_dims = len(test_vector)
            if actual_dims != VECTOR_DIMS:
                logger.error(f"❌ 向量维度不匹配! 期望: {VECTOR_DIMS}, 实际: {actual_dims}")
                logger.error(f"   请检查模型配置，可能需要修改 VECTOR_DIMS 为 {actual_dims}")
            else:
                logger.info(f"✅ 向量生成测试通过: 维度 = {actual_dims}")
        else:
            logger.error(f"❌ 向量生成测试失败: generate_vector 返回 None")
        process_single_item._tested = True

    # 提取关键字段
    label = item.get("label", "")
    link = item.get("wikipedia") or item.get("wikipediaLink", "")
    aliases_en = item.get("en_aliases") or item.get("aliases_en", [])
    aliases_zh = item.get("zh_aliases") or item.get("aliases_zh", [])
    descriptions_en = item.get("en_description") or item.get("descriptions_en", "")
    descriptions_zh = item.get("zh_description") or item.get("descriptions_zh", "")
    content = item.get("content", "")
    
    # 提取实体词（优先使用新的实体词字段，兼容旧的高频词字段）
    # 注意：现在使用NER模型提取的实体词，而不是高频词统计
    entity_words_zh = item.get("_entity_words_zh", item.get("_high_freq_words_zh", []))
    entity_words_en = item.get("_entity_words_en", item.get("_high_freq_words_en", []))
    
    # 使用实体词列表（变量名保持简洁）
    entity_words_zh_list = entity_words_zh
    entity_words_en_list = entity_words_en
    
    # 优先使用 find_top_k.py 预生成的实体词向量（如果存在）
    entity_words_zh_vector = item.get("_entity_words_zh_vector")
    entity_words_en_vector = item.get("_entity_words_en_vector")

    # 构建数据对象
    new_data = {
        "label": label,
        "link": link,
        "aliases_en": aliases_en if isinstance(aliases_en, list) else [],
        "aliases_zh": aliases_zh if isinstance(aliases_zh, list) else [],
        "descriptions_en": descriptions_en,
        "descriptions_zh": descriptions_zh,
        "content": content
    }
    
    # 1. 为标签生成向量（包含所有别名）
    # 1.1 中文标签向量（label + 所有中文别名）
    label_zh_parts = []
    if label:
        label_zh_parts.append(label)
    if aliases_zh and isinstance(aliases_zh, list):
        # 包含所有中文别名，不限制数量
        label_zh_parts.extend(aliases_zh)
    
    if label_zh_parts:
        label_zh_text = " ".join(label_zh_parts)
        label_zh_vector = generate_vector(label_zh_text)
        if label_zh_vector and len(label_zh_vector) == VECTOR_DIMS:
            new_data["label_zh_vector"] = label_zh_vector
        else:
            logger.warning(f"中文标签向量生成失败，标签: {label[:30]}, 别名数: {len(aliases_zh) if aliases_zh else 0}")
    
    # 1.2 英文标签向量（label + 所有英文别名）
    label_en_parts = []
    if label:
        label_en_parts.append(label)
    if aliases_en and isinstance(aliases_en, list):
        # 包含所有英文别名，不限制数量
        label_en_parts.extend(aliases_en)
    
    if label_en_parts:
        label_en_text = " ".join(label_en_parts)
        label_en_vector = generate_vector(label_en_text)
        if label_en_vector and len(label_en_vector) == VECTOR_DIMS:
            new_data["label_en_vector"] = label_en_vector
        else:
            logger.warning(f"英文标签向量生成失败，标签: {label[:30]}, 别名数: {len(aliases_en) if aliases_en else 0}")
    
    # 1.3 通用标签向量（仅label，用于兼容性）
    if label:
        label_vector = generate_vector(label)
        if label_vector and len(label_vector) == VECTOR_DIMS:
            new_data["label_vector"] = label_vector
    
    # 2. 为中文描述生成向量
    if descriptions_zh and len(descriptions_zh.strip()) > 10:
        zh_desc_vector = generate_vector(descriptions_zh)
        if zh_desc_vector and len(zh_desc_vector) == VECTOR_DIMS:
            new_data["descriptions_zh_vector"] = zh_desc_vector
        else:
            logger.warning(f"中文描述向量生成失败，标签: {label[:30]}")
    
    # 3. 为英文描述生成向量
    if descriptions_en and len(descriptions_en.strip()) > 10:
        en_desc_vector = generate_vector(descriptions_en)
        if en_desc_vector and len(en_desc_vector) == VECTOR_DIMS:
            new_data["descriptions_en_vector"] = en_desc_vector
        else:
            logger.warning(f"英文描述向量生成失败，标签: {label[:30]}")
    
    # 4. 为中文实体词生成向量（使用NER提取的实体词）
    # 优先使用 find_top_k.py 预生成的向量（如果存在）
    if entity_words_zh_vector and isinstance(entity_words_zh_vector, list) and len(entity_words_zh_vector) == VECTOR_DIMS:
        # 直接使用预生成的向量（已使用微调后的模型向量化并合并）
        new_data["entity_words_zh_vector"] = [float(x) for x in entity_words_zh_vector]
        logger.debug(f"使用预生成的中文实体词向量，标签: {label[:30]}, 实体词数: {len(entity_words_zh_list) if entity_words_zh_list else 0}")
    elif entity_words_zh_list and isinstance(entity_words_zh_list, list) and len(entity_words_zh_list) > 0:
        # 如果没有预生成的向量，则重新向量化（使用微调后的模型）
        # 方法：对每个实体词单独向量化，然后合并（与 find_top_k.py 保持一致）
        try:
            from vector_model import batch_generate_vectors
            import numpy as np
            
            # 批量生成每个实体词的向量（使用微调后的模型）
            entity_vectors = batch_generate_vectors(
                entity_words_zh_list,
                use_finetuned=True,
                target_dim=VECTOR_DIMS,
                batch_size=32
            )
            
            # 过滤掉None值
            valid_vectors = [v for v in entity_vectors if v is not None and isinstance(v, list) and len(v) == VECTOR_DIMS]
            
            if valid_vectors:
                # 合并向量（使用平均值，与 find_top_k.py 保持一致）
                vectors_array = np.array(valid_vectors)
                merged_vector = np.mean(vectors_array, axis=0)
                
                # L2归一化
                norm = np.linalg.norm(merged_vector)
                if norm > 0:
                    merged_vector = merged_vector / norm
                    new_data["entity_words_zh_vector"] = [float(x) for x in merged_vector.tolist()]
                else:
                    logger.warning(f"中文实体词向量合并后归一化失败，标签: {label[:30]}")
            else:
                logger.warning(f"中文实体词向量化失败，标签: {label[:30]}, 实体词数: {len(entity_words_zh_list)}")
        except Exception as e:
            # 回退到原始方法（将所有实体词用空格连接，然后向量化）
            logger.warning(f"批量向量化失败，使用回退方法: {e}")
            zh_freq_text = " ".join(entity_words_zh_list)
            zh_freq_vector = generate_vector(zh_freq_text)
            if zh_freq_vector and len(zh_freq_vector) == VECTOR_DIMS:
                new_data["entity_words_zh_vector"] = zh_freq_vector
            else:
                logger.warning(f"中文实体词向量生成失败，标签: {label[:30]}, 实体词数: {len(entity_words_zh_list)}")
    
    # 5. 为英文实体词生成向量（使用NER提取的实体词）
    # 优先使用 find_top_k.py 预生成的向量（如果存在）
    if entity_words_en_vector and isinstance(entity_words_en_vector, list) and len(entity_words_en_vector) == VECTOR_DIMS:
        # 直接使用预生成的向量（已使用微调后的模型向量化并合并）
        new_data["entity_words_en_vector"] = [float(x) for x in entity_words_en_vector]
        logger.debug(f"使用预生成的英文实体词向量，标签: {label[:30]}, 实体词数: {len(entity_words_en_list) if entity_words_en_list else 0}")
    elif entity_words_en_list and isinstance(entity_words_en_list, list) and len(entity_words_en_list) > 0:
        # 如果没有预生成的向量，则重新向量化（使用微调后的模型）
        try:
            from vector_model import batch_generate_vectors
            import numpy as np
            
            # 批量生成每个实体词的向量（使用微调后的模型）
            entity_vectors = batch_generate_vectors(
                entity_words_en_list,
                use_finetuned=True,
                target_dim=VECTOR_DIMS,
                batch_size=32
            )
            
            # 过滤掉None值
            valid_vectors = [v for v in entity_vectors if v is not None and isinstance(v, list) and len(v) == VECTOR_DIMS]
            
            if valid_vectors:
                # 合并向量（使用平均值，与 find_top_k.py 保持一致）
                vectors_array = np.array(valid_vectors)
                merged_vector = np.mean(vectors_array, axis=0)
                
                # L2归一化
                norm = np.linalg.norm(merged_vector)
                if norm > 0:
                    merged_vector = merged_vector / norm
                    new_data["entity_words_en_vector"] = [float(x) for x in merged_vector.tolist()]
                else:
                    logger.warning(f"英文实体词向量合并后归一化失败，标签: {label[:30]}")
            else:
                logger.warning(f"英文实体词向量化失败，标签: {label[:30]}, 实体词数: {len(entity_words_en_list)}")
        except Exception as e:
            # 回退到原始方法（将所有实体词用空格连接，然后向量化）
            logger.warning(f"批量向量化失败，使用回退方法: {e}")
            en_freq_text = " ".join(entity_words_en_list)
            en_freq_vector = generate_vector(en_freq_text)
            if en_freq_vector and len(en_freq_vector) == VECTOR_DIMS:
                new_data["entity_words_en_vector"] = en_freq_vector
            else:
                logger.warning(f"英文实体词向量生成失败，标签: {label[:30]}, 实体词数: {len(entity_words_en_list)}")

    return new_data


def count_lines(filename):
    """快速计算文件行数"""
    with open(filename, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def process_and_import_to_es(input_path, batch_size=20, request_timeout=120, vector_batch_size=32, use_batch_vectors=True):
    """处理JSONL文件并导入到ES - 优化性能
    
    Args:
        input_path: 输入文件路径
        batch_size: ES批量导入大小
        request_timeout: ES请求超时时间
        vector_batch_size: 向量批量生成大小（GPU加速时建议32-64）
        use_batch_vectors: 是否使用批量向量生成（显著提升速度）
    """
    print("=" * 60)
    print(f"开始处理文件: {input_path}")
    print("=" * 60)
    
    # 前置检查
    print("\n[前置检查]")
    if not os.path.exists(input_path):
        print(f"❌ 错误: 文件不存在: {input_path}")
        return
    
    try:
        total_lines = count_lines(input_path)
        print(f"✅ 文件存在，总行数: {total_lines}")
    except Exception as e:
        print(f"❌ 错误: 无法读取文件: {e}")
        return
    
    # 检查ES连接
    try:
        if not es.indices.exists(index=INDEX_NAME):
            print(f"❌ 错误: 索引 {INDEX_NAME} 不存在，请先创建索引")
            return
        print(f"✅ ES索引存在: {INDEX_NAME}")
    except Exception as e:
        print(f"❌ 错误: ES连接失败: {e}")
        return
    
    # 检查设备
    device_info = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"✅ 使用设备: {device_info}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        # GPU模式下建议使用批量向量生成
        if not use_batch_vectors:
            print(f"   ⚠️  建议启用批量向量生成以提升速度（当前已禁用）")
    else:
        # CPU模式下批量处理可能较慢，但仍有提升
        if use_batch_vectors:
            print(f"   💡 CPU模式：批量向量生成可能提升有限")
    
    if use_batch_vectors:
        print(f"✅ 批量向量生成: 启用 (批量大小: {vector_batch_size})")
    else:
        print(f"⚠️  批量向量生成: 禁用 (逐个生成)")
    print("=" * 60)
    
    actions = []
    total_imported = 0
    failed_count = 0
    vector_stats = {
        'label': 0,
        'label_zh': 0,
        'label_en': 0,
        'descriptions_zh': 0,
        'descriptions_en': 0,
        'entity_words_zh': 0,
        'entity_words_en': 0
    }
    start_time = time.time()
    last_speed_time = start_time
    last_speed_count = 0

    # 记录导入前的文档数量
    try:
        doc_count_before = es.count(index=INDEX_NAME)["count"]
        print(f"\n导入前索引文档数量: {doc_count_before}")
    except Exception as e:
        print(f"⚠️  警告: 获取初始文档数失败: {e}")
        doc_count_before = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        progress_bar = tqdm(total=total_lines, desc="处理进度", unit="条")

        for line_num, line in enumerate(f, 1):
            try:
                if not line.strip():
                    progress_bar.update(1)
                    continue

                data = json.loads(line.strip())
                
                # 如果使用批量向量生成，先收集文本，稍后批量处理
                if use_batch_vectors and line_num % (batch_size * 2) == 1:
                    # 每处理一定数量后，批量生成向量
                    # 这里简化处理，仍然逐个生成，但可以优化为真正的批量
                    transformed_data = process_single_item(data)
                else:
                    transformed_data = process_single_item(data)

                # 统计向量数量
                if 'label_vector' in transformed_data:
                    vector_stats['label'] += 1
                if 'label_zh_vector' in transformed_data:
                    vector_stats['label_zh'] += 1
                if 'label_en_vector' in transformed_data:
                    vector_stats['label_en'] += 1
                if 'descriptions_zh_vector' in transformed_data:
                    vector_stats['descriptions_zh'] += 1
                if 'descriptions_en_vector' in transformed_data:
                    vector_stats['descriptions_en'] += 1
                if 'entity_words_zh_vector' in transformed_data:
                    vector_stats['entity_words_zh'] += 1
                if 'entity_words_en_vector' in transformed_data:
                    vector_stats['entity_words_en'] += 1

                # 在开始处理时显示信息
                if line_num == 1:
                    print(f"\n[第一条数据示例]")
                    label = transformed_data.get('label', 'N/A')
                    aliases_zh = transformed_data.get('aliases_zh', [])
                    aliases_en = transformed_data.get('aliases_en', [])
                    print(f"  标签: {label}")
                    print(f"  中文别名数: {len(aliases_zh) if isinstance(aliases_zh, list) else 0}")
                    print(f"  英文别名数: {len(aliases_en) if isinstance(aliases_en, list) else 0}")
                    print(f"  标签向量: {'✓' if 'label_vector' in transformed_data else '✗'}")
                    print(f"  中文标签向量(含所有别名): {'✓' if 'label_zh_vector' in transformed_data else '✗'}")
                    print(f"  英文标签向量(含所有别名): {'✓' if 'label_en_vector' in transformed_data else '✗'}")
                    print(f"  中文描述向量: {'✓' if 'descriptions_zh_vector' in transformed_data else '✗'}")
                    print(f"  英文描述向量: {'✓' if 'descriptions_en_vector' in transformed_data else '✗'}")
                    # 注意：实体词列表不会存入ES，只存储向量，所以从原始数据读取用于显示
                    entity_words_zh = data.get('_entity_words_zh', data.get('_high_freq_words_zh', []))
                    entity_words_en = data.get('_entity_words_en', data.get('_high_freq_words_en', []))
                    print(f"  中文实体词数: {len(entity_words_zh) if isinstance(entity_words_zh, list) else 0}")
                    print(f"  英文实体词数: {len(entity_words_en) if isinstance(entity_words_en, list) else 0}")
                    # ES中存储的字段名与源文件一致：entity_words_*_vector
                    print(f"  中文实体词向量(ES字段: entity_words_zh_vector): {'✓' if 'entity_words_zh_vector' in transformed_data else '✗'}")
                    print(f"  英文实体词向量(ES字段: entity_words_en_vector): {'✓' if 'entity_words_en_vector' in transformed_data else '✗'}")
                    print()

                actions.append({
                    "_index": INDEX_NAME,
                    "_source": transformed_data
                })

                # 批量导入 - 使用新的API调用方式
                if len(actions) >= batch_size:
                    try:
                        # 修正：使用新的API调用方式
                        success, failed = helpers.bulk(
                            es.options(request_timeout=request_timeout),
                            actions,
                            raise_on_error=False,
                            stats_only=False
                        )

                        if failed:
                            error_count = len(failed)
                            failed_count += error_count
                            total_imported += (len(actions) - error_count)
                            # 记录错误信息
                            if failed:
                                error_info = failed[0].get('index', {}).get('error', {})
                                logger.warning(
                                    f"批量导入部分失败: {error_count}条失败，错误: {error_info.get('reason', '未知错误')[:100]}")
                        else:
                            total_imported += len(actions)
                            logger.debug(f"成功导入 {len(actions)} 条数据到ES")

                        actions = []

                        # 每500条更新一次进度条
                        if total_imported % 500 == 0:
                            current_time = time.time()
                            elapsed = current_time - start_time
                            avg_speed = total_imported / elapsed if elapsed > 0 else 0

                            # 计算最近一段时间的速度
                            recent_elapsed = current_time - last_speed_time
                            recent_speed = (total_imported - last_speed_count) / recent_elapsed if recent_elapsed > 0 else 0
                            last_speed_time = current_time
                            last_speed_count = total_imported

                            total_vectors = sum(vector_stats.values())
                            progress_bar.set_postfix({
                                '已导入': total_imported,
                                '向量': total_vectors,
                                '失败': failed_count,
                                '速度': f'{recent_speed:.1f}条/s',
                                '设备': device_info
                            })

                    except Exception as e:
                        error_msg = str(e)[:200]
                        logger.error(f"批量导入异常: {error_msg}")
                        print(f"❌ 批量导入错误: {error_msg}")
                        failed_count += len(actions)
                        actions = []

                progress_bar.update(1)

            except Exception as e:
                failed_count += 1
                progress_bar.update(1)
                error_msg = str(e)[:200]
                if failed_count <= 10:
                    print(f"❌ 第{line_num}行处理失败: {error_msg}")
                logger.warning(f"第{line_num}行处理失败: {error_msg}")

        # 导入剩余数据
        if actions:
            try:
                success, failed = helpers.bulk(
                    es.options(request_timeout=request_timeout),
                    actions,
                    raise_on_error=False,
                    stats_only=False
                )

                if failed:
                    error_count = len(failed)
                    failed_count += error_count
                    total_imported += (len(actions) - error_count)
                else:
                    total_imported += len(actions)
            except Exception as e:
                logger.error(f"最后一批导入异常: {e}")
                failed_count += len(actions)

        progress_bar.close()

    elapsed_time = time.time() - start_time
    total_vectors = sum(vector_stats.values())

    print(f"\n{'=' * 60}")
    print(f"导入完成!")
    print(f"{'=' * 60}")
    print(f"成功导入: {total_imported}条")
    print(f"失败: {failed_count}条")
    print(f"总耗时: {elapsed_time:.1f}秒 ({elapsed_time / 60:.1f}分钟)")
    avg_speed = total_imported / elapsed_time if elapsed_time > 0 else 0
    print(f"平均速度: {avg_speed:.2f}条/秒")
    print(f"\n向量生成统计:")
    print(f"  标签向量(仅label): {vector_stats['label']}个")
    print(f"  中文标签向量(label+所有中文别名): {vector_stats['label_zh']}个")
    print(f"  英文标签向量(label+所有英文别名): {vector_stats['label_en']}个")
    print(f"  中文描述向量: {vector_stats['descriptions_zh']}个")
    print(f"  英文描述向量: {vector_stats['descriptions_en']}个")
    print(f"  中文实体词向量: {vector_stats['entity_words_zh']}个")
    print(f"  英文实体词向量: {vector_stats['entity_words_en']}个")
    print(f"  向量总数: {total_vectors}个")
    
    if total_imported > 0:
        print(f"\n向量生成率:")
        print(f"  标签(仅label): {vector_stats['label']/total_imported*100:.1f}%")
        print(f"  中文标签(label+所有别名): {vector_stats['label_zh']/total_imported*100:.1f}%")
        print(f"  英文标签(label+所有别名): {vector_stats['label_en']/total_imported*100:.1f}%")
        print(f"  中文描述: {vector_stats['descriptions_zh']/total_imported*100:.1f}%")
        print(f"  英文描述: {vector_stats['descriptions_en']/total_imported*100:.1f}%")
        print(f"  中文实体词向量: {vector_stats['entity_words_zh']/total_imported*100:.1f}%")
        print(f"  英文实体词向量: {vector_stats['entity_words_en']/total_imported*100:.1f}%")

    # 显示设备使用情况
    print(f"\n设备信息:")
    print(f"  使用设备: {device_info}")
    if torch.cuda.is_available():
        gpu_memory_used = torch.cuda.max_memory_allocated() / 1024 ** 3  # GB
        print(f"  GPU显存使用: {gpu_memory_used:.2f} GB")
    print(f"{'=' * 60}")

    # 获取最终统计
    try:
        doc_count_after = es.count(index=INDEX_NAME)["count"]
        actual_imported = doc_count_after - doc_count_before
        print(f"实际新增文档数: {actual_imported}")
        print(f"导入后索引中文档总数: {doc_count_after}")

        # 使用 exists 查询检查向量字段是否存在（更可靠）
        print("\n[向量字段存在性检查 - 使用 exists 查询]")
        
        vector_fields = [
            'label_vector',
            'label_zh_vector',
            'label_en_vector',
            'descriptions_zh_vector',
            'descriptions_en_vector',
            'entity_words_zh_vector',
            'entity_words_en_vector'
        ]
        
        field_stats = {}
        for field in vector_fields:
            try:
                # 使用 exists 查询统计有多少文档包含该字段
                exists_query = {
                    "query": {
                        "exists": {
                            "field": field
                        }
                    }
                }
                result = es.count(index=INDEX_NAME, body=exists_query)
                count = result.get('count', 0)
                field_stats[field] = count
                percentage = (count / doc_count_after * 100) if doc_count_after > 0 else 0
                print(f"  {field}: {count} 个文档 ({percentage:.1f}%)")
            except Exception as e:
                print(f"  {field}: 检查失败 - {e}")
                field_stats[field] = 0
        
        # 检查至少有一个向量字段的文档数
        try:
            any_vector_query = {
                "query": {
                    "bool": {
                        "should": [
                            {"exists": {"field": field}} for field in vector_fields
                        ],
                        "minimum_should_match": 1
                    }
                }
            }
            any_vector_result = es.count(index=INDEX_NAME, body=any_vector_query)
            any_vector_count = any_vector_result.get('count', 0)
            any_vector_percentage = (any_vector_count / doc_count_after * 100) if doc_count_after > 0 else 0
            print(f"\n至少有一个向量字段的文档数: {any_vector_count}/{doc_count_after} ({any_vector_percentage:.1f}%)")
        except Exception as e:
            print(f"\n检查至少有一个向量字段的文档数失败: {e}")
        
        # 随机采样几个文档，显示详细的向量字段情况
        print("\n[随机采样文档检查 - 显示每个文档包含的向量类型]")
        print("说明: 显示随机抽取的5个文档，列出每个文档包含哪些向量字段")
        try:
            sample_query = {
                "size": 5,
                "_source": ["label"] + vector_fields,  # 同时获取label和所有向量字段
                "query": {
                    "match_all": {}
                }
            }
            sample_result = es.search(index=INDEX_NAME, body=sample_query)
            
            for hit in sample_result['hits']['hits']:
                source = hit['_source']
                label = source.get('label', 'N/A')
                
                # 检查每个向量字段是否存在（使用 exists 查询验证，同时检查 _source）
                doc_vectors = {}
                for field in vector_fields:
                    # 先检查 _source 中是否有该字段
                    has_in_source = field in source
                    # 如果 _source 中没有，再用 exists 查询确认
                    if not has_in_source:
                        try:
                            exists_check = {
                                "query": {
                                    "bool": {
                                        "must": [
                                            {"term": {"_id": hit['_id']}},
                                            {"exists": {"field": field}}
                                        ]
                                    }
                                }
                            }
                            check_result = es.count(index=INDEX_NAME, body=exists_check)
                            has_in_source = check_result.get('count', 0) > 0
                        except:
                            pass
                    doc_vectors[field] = has_in_source
                
                # 显示向量字段状态（使用更清晰的格式）
                vector_status_parts = []
                if doc_vectors.get('label_vector'): 
                    vector_status_parts.append('✓标签')
                if doc_vectors.get('label_zh_vector'): 
                    vector_status_parts.append('✓中文标签')
                if doc_vectors.get('label_en_vector'): 
                    vector_status_parts.append('✓英文标签')
                if doc_vectors.get('descriptions_zh_vector'): 
                    vector_status_parts.append('✓中文描述')
                if doc_vectors.get('descriptions_en_vector'): 
                    vector_status_parts.append('✓英文描述')
                if doc_vectors.get('entity_words_zh_vector'): 
                    vector_status_parts.append('✓中文实体词')
                if doc_vectors.get('entity_words_en_vector'): 
                    vector_status_parts.append('✓英文实体词')
                
                # 显示缺失的字段
                missing_parts = []
                if not doc_vectors.get('label_vector'): 
                    missing_parts.append('✗标签')
                if not doc_vectors.get('label_zh_vector'): 
                    missing_parts.append('✗中文标签')
                if not doc_vectors.get('label_en_vector'): 
                    missing_parts.append('✗英文标签')
                if not doc_vectors.get('descriptions_zh_vector'): 
                    missing_parts.append('✗中文描述')
                if not doc_vectors.get('descriptions_en_vector'): 
                    missing_parts.append('✗英文描述')
                if not doc_vectors.get('entity_words_zh_vector'): 
                    missing_parts.append('✗中文实体词')
                if not doc_vectors.get('entity_words_en_vector'): 
                    missing_parts.append('✗英文实体词')
                
                # 显示结果
                if vector_status_parts:
                    status_str = " | ".join(vector_status_parts)
                    if missing_parts:
                        status_str += f" | 缺失: {', '.join(missing_parts)}"
                    print(f"  {label[:30]}: {status_str}")
                else:
                    print(f"  {label[:30]}: 无向量字段")
        except Exception as e:
            print(f"采样检查失败: {e}")
            logger.error(f"采样检查失败详情: {e}")

    except Exception as e:
        print(f"获取统计信息失败: {e}")
        logger.error(f"获取统计信息失败详情: {e}")


if __name__ == "__main__":
    import os
    import sys

    print("=" * 60)
    print("开始向量化导入流程")
    print("=" * 60)

    # 默认处理top_k_zh.jsonl和top_k_en.jsonl（预处理后的高频实体文件）
    # 可以通过命令行参数指定单个文件
    target_file = None
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--file" or sys.argv[1] == "-f":
            # 使用--file参数指定文件
            if len(sys.argv) > 2:
                target_file = sys.argv[2]
            else:
                print("错误: --file参数需要指定文件路径")
                exit(1)
        else:
            # 其他参数作为文件路径
            target_file = sys.argv[1]
    
    # 先创建正确的索引映射
    print("创建向量索引映射...")
    if not create_vector_index():
        print("索引创建失败，退出")
        exit(1)

    processed_files = []
    vector_batch = 64 if torch.cuda.is_available() else 16
    
    # 如果指定了文件路径，只处理该文件
    if target_file:
        if os.path.exists(target_file):
            print(f"\n处理指定文件: {target_file}")
            process_and_import_to_es(target_file, batch_size=20, request_timeout=180,
                                    vector_batch_size=vector_batch, use_batch_vectors=True)
            processed_files.append(target_file)
        else:
            print(f"❌ 错误: 文件不存在: {target_file}")
            exit(1)
    
    # 默认处理entity_words_zh.jsonl和entity_words_en.jsonl（预处理后的实体词文件）
    else:
        # 查找预处理后的实体词文件（由 find_top_k.py 生成）
        # 优先使用配置文件中的路径
        if ENTITY_WORDS_ZH_FILE and os.path.exists(ENTITY_WORDS_ZH_FILE):
            target_files = [ENTITY_WORDS_ZH_FILE]
        elif ENTITY_WORDS_EN_FILE and os.path.exists(ENTITY_WORDS_EN_FILE):
            target_files = [ENTITY_WORDS_EN_FILE]
        else:
            target_files = [
                "entity_words_zh.jsonl",  # 中文文件处理结果（NER提取的实体词）
                "entity_words_en.jsonl"   # 英文文件处理结果（NER提取的实体词）
            ]
        
        found_files = []
        for target_file in target_files:
            # 按优先级查找文件：当前目录 -> data目录 -> 父目录
            if os.path.exists(target_file):
                found_files.append(target_file)
                continue
            
            # 尝试在 data 目录查找（find_top_k.py 的输出目录）
            data_dir = os.path.join(parent_dir, 'data')
            data_target = os.path.join(data_dir, target_file)
            if os.path.exists(data_target):
                found_files.append(data_target)
                continue
            
            # 尝试在父目录查找
            parent_target = os.path.join(parent_dir, target_file)
            if os.path.exists(parent_target):
                found_files.append(parent_target)
        
        if found_files:
            for target_file in found_files:
                file_name = os.path.basename(target_file)
                file_type = "中文" if "zh" in file_name else "英文"
                print(f"\n找到预处理后的{file_type}实体词文件: {target_file}")
                print(f"开始处理: {target_file}")
                process_and_import_to_es(target_file, batch_size=20, request_timeout=180,
                                        vector_batch_size=vector_batch, use_batch_vectors=True)
                processed_files.append(target_file)
        else:
            print("❌ 错误: 未找到预处理后的实体词文件")
            print("   期望的文件:")
            print("     - entity_words_zh.jsonl (中文文件处理结果，包含NER提取的实体词)")
            print("     - entity_words_en.jsonl (英文文件处理结果，包含NER提取的实体词)")
            print("\n   请先运行以下命令生成预处理文件:")
            print("   cd data")
            print("   python find_top_k.py")
            print("\n   或者:")
            print("   python data/find_top_k.py")
            exit(1)

    if not processed_files:
        print("错误: 未找到任何数据文件")
        exit(1)

    print(f"\n已完成处理以下文件:")
    for f in processed_files:
        print(f"  - {f}")

    print("\n导入流程完成! 请检查阿里云控制台的向量存储用量")