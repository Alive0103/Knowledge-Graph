import json
import torch
from transformers import BertTokenizer, BertModel
from elasticsearch import helpers
from tqdm import tqdm
import time
import logging
from ..es_client import es

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

# 模型路径 - 根据实际模型确定维度
model_name = './model/chinese-roberta-wwm-ext-large'  # 使用large模型
VECTOR_DIMS = 1024  # large模型是1024维，base模型是768维

# 初始化 BERT 模型和分词器
try:
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()

    # 检查是否有GPU可用，如果有则使用GPU加速
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3  # GB
        print(f"✅ GPU检测成功!")
        print(f"   GPU设备: {gpu_name}")
        print(f"   GPU显存: {gpu_memory:.1f} GB")
        print(f"   CUDA版本: {torch.version.cuda}")
    else:
        device = torch.device('cpu')
        print(f"⚠️  未检测到GPU，使用CPU模式（速度较慢）")
        print(f"   提示: 如果有NVIDIA GPU，请安装CUDA版本的PyTorch以加速")

    model = model.to(device)
    print(f"\nBERT模型加载成功: {model_name}")
    print(f"使用设备: {device}")
    print(f"预期向量维度: {VECTOR_DIMS}")
    
    # 测试生成一个向量，验证维度是否正确（需要先定义 generate_vector，所以放在后面）
    print("\n⚠️  注意: 向量维度测试将在首次调用 generate_vector 时进行")
    print("=" * 60)
except Exception as e:
    print(f"BERT模型加载失败: {e}")
    exit(1)

# ES索引名称
INDEX_NAME = "data2"


def create_vector_index():
    """创建包含向量字段的索引映射 - 修正维度问题"""
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
                # 修正：使用正确的维度1024
                "descriptions_zh_vector": {
                    "type": "dense_vector",
                    "dims": VECTOR_DIMS,  # 使用变量而不是固定值
                    "index": True,
                    "similarity": "cosine"
                },
                "descriptions_en_vector": {
                    "type": "dense_vector",
                    "dims": VECTOR_DIMS,
                    "index": True,
                    "similarity": "cosine"
                },
                # 新增：短文本向量字段（基于 label + aliases）
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
    """生成文本向量 - 优化性能，使用GPU加速
    
    注意：这个函数在模块加载时会被调用进行测试，所以不能依赖全局变量
    """
    if text and text.strip():
        try:
            # 不再限制文本长度，让模型自己处理
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            # 将输入移到GPU（如果可用）
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
            
            vector = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

            # 确保向量是一维数组
            import numpy as np
            if len(vector.shape) == 0:
                vector = vector.reshape(1)
            elif len(vector.shape) > 1:
                # 如果是多维，展平为一维
                vector = vector.flatten()
            
            # 检查向量维度
            actual_dims = len(vector)
                
                if actual_dims != VECTOR_DIMS:
                # 记录维度不匹配的错误（改为 WARNING 以便排查问题）
                logger.warning(f"向量维度不匹配! 期望: {VECTOR_DIMS}, 实际: {actual_dims}, 文本: {text[:50] if text else 'None'}")
                    return None
                
                # 添加L2归一化
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
            else:
                return None

            vector_list = vector.tolist()

            # 验证转换后的维度
            if len(vector_list) != VECTOR_DIMS:
                return None
                
            # 确保所有元素都是 float 类型（ES dense_vector 要求）
            vector_list = [float(x) for x in vector_list]

            return vector_list

        except Exception as e:
            # 记录错误信息（改为 WARNING 级别以便排查问题）
            logger.warning(f"向量生成失败: {e}, 文本: {text[:50] if text else 'None'}")
            return None
    return None


def process_single_item(item):
    """处理单条数据项 - 确保所有数据都生成向量"""
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

    # 为中文描述生成向量
    # 优先使用描述文本，如果没有或者太短，则使用标签和别名
    zh_text_for_vector = ""
    if descriptions_zh and len(descriptions_zh.strip()) > 10:
        zh_text_for_vector = descriptions_zh
    elif label:
        # 使用标签+中文别名作为替代
        zh_text_for_vector = label
        if aliases_zh and isinstance(aliases_zh, list):
            zh_text_for_vector += " " + " ".join(aliases_zh[:5])  # 限制别名数量
    
    if zh_text_for_vector:
        vector = generate_vector(zh_text_for_vector)
        if vector:
            if len(vector) == VECTOR_DIMS:
            new_data["descriptions_zh_vector"] = vector
            else:
                logger.warning(f"中文向量维度错误: 期望{VECTOR_DIMS}, 实际{len(vector)}, 标签: {label[:30]}")
        else:
            logger.warning(f"中文向量生成失败，标签: {label[:30]}, 文本长度: {len(zh_text_for_vector)}")
    else:
        logger.warning(f"无法生成中文向量（缺少文本），标签: {label[:30]}")

    # 为英文描述生成向量
    # 优先使用描述文本，如果没有或者太短，则使用标签和别名
    en_text_for_vector = ""
    if descriptions_en and len(descriptions_en.strip()) > 10:
        en_text_for_vector = descriptions_en
    elif label:
        # 使用标签+英文别名作为替代
        en_text_for_vector = label
        if aliases_en and isinstance(aliases_en, list):
            en_text_for_vector += " " + " ".join(aliases_en[:5])  # 限制别名数量
    
    if en_text_for_vector:
        vector = generate_vector(en_text_for_vector)
        if vector:
            if len(vector) == VECTOR_DIMS:
            new_data["descriptions_en_vector"] = vector
            else:
                logger.warning(f"英文向量维度错误: 期望{VECTOR_DIMS}, 实际{len(vector)}, 标签: {label[:30]}")
        else:
            logger.warning(f"英文向量生成失败，标签: {label[:30]}, 文本长度: {len(en_text_for_vector)}")
    else:
        logger.warning(f"无法生成英文向量（缺少文本），标签: {label[:30]}")

    # 新增：为 label + aliases 生成短文本向量（用于对齐查询向量）
    # 中文短文本向量
    label_zh_text = ""
    if label:
        label_zh_text = label
        if aliases_zh and isinstance(aliases_zh, list):
            label_zh_text += " " + " ".join(aliases_zh[:5])  # 限制别名数量

    if label_zh_text:
        vector = generate_vector(label_zh_text)
        if vector:
            if len(vector) == VECTOR_DIMS:
                new_data["label_zh_vector"] = vector
            else:
                logger.warning(f"中文标签向量维度错误: 期望{VECTOR_DIMS}, 实际{len(vector)}, 标签: {label[:30]}")
        else:
            logger.warning(f"中文标签向量生成失败，标签: {label[:30]}, 文本长度: {len(label_zh_text)}")

    # 英文短文本向量
    label_en_text = ""
    if label:
        label_en_text = label
        if aliases_en and isinstance(aliases_en, list):
            label_en_text += " " + " ".join(aliases_en[:5])  # 限制别名数量

    if label_en_text:
        vector = generate_vector(label_en_text)
        if vector:
            if len(vector) == VECTOR_DIMS:
                new_data["label_en_vector"] = vector
            else:
                logger.warning(f"英文标签向量维度错误: 期望{VECTOR_DIMS}, 实际{len(vector)}, 标签: {label[:30]}")
        else:
            logger.warning(f"英文标签向量生成失败，标签: {label[:30]}, 文本长度: {len(label_en_text)}")

    return new_data


def count_lines(filename):
    """快速计算文件行数"""
    with open(filename, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def process_and_import_to_es(input_path, batch_size=20, request_timeout=120):
    """处理JSONL文件并导入到ES - 优化性能"""
    print(f"开始处理文件: {input_path}")
    
    total_lines = count_lines(input_path)
    print(f"文件总行数: {total_lines}")
    
    actions = []
    total_imported = 0
    failed_count = 0
    vector_count = 0
    start_time = time.time()
    last_speed_time = start_time
    last_speed_count = 0
    
    # 记录导入前的文档数量
    try:
        doc_count_before = es.count(index=INDEX_NAME)["count"]
        print(f"导入前索引文档数量: {doc_count_before}")
    except Exception as e:
        print(f"获取初始文档数失败: {e}")
        doc_count_before = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        progress_bar = tqdm(total=total_lines, desc="处理进度", unit="条")
        
        for line_num, line in enumerate(f, 1):
            try:
                if not line.strip():
                    progress_bar.update(1)
                    continue
                    
                data = json.loads(line.strip())
                transformed_data = process_single_item(data)
                
                # 统计向量数量（简化验证，减少性能开销）
                if 'descriptions_zh_vector' in transformed_data:
                    vector_count += 1
                if 'descriptions_en_vector' in transformed_data:
                    vector_count += 1
                if 'label_zh_vector' in transformed_data:
                    vector_count += 1
                if 'label_en_vector' in transformed_data:
                    vector_count += 1

                # 在开始处理时显示信息
                if line_num == 1:
                    logger.info(f"开始处理数据 - 第一条标签: {transformed_data.get('label', 'N/A')}")
                    logger.info(f"  中文向量: {'✓' if 'descriptions_zh_vector' in transformed_data else '✗'}, "
                                f"英文向量: {'✓' if 'descriptions_en_vector' in transformed_data else '✗'}")

                # 每处理一定数量显示一次进度
                if line_num % 100 == 0:
                    logger.info(f"已处理 {line_num} 条数据，已导入 {total_imported} 条，生成向量 {vector_count} 个")
                
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
                        
                        # 每500条显示一次进度（减少更新频率）
                        if total_imported % 500 == 0:
                            current_time = time.time()
                            elapsed = current_time - start_time
                            avg_speed = total_imported / elapsed if elapsed > 0 else 0

                            # 计算最近一段时间的速度
                            recent_elapsed = current_time - last_speed_time
                            recent_speed = (
                                                       total_imported - last_speed_count) / recent_elapsed if recent_elapsed > 0 else 0
                            last_speed_time = current_time
                            last_speed_count = total_imported

                            # 显示设备信息
                            device_info = "GPU" if torch.cuda.is_available() else "CPU"

                            progress_bar.set_postfix({
                                '已导入': total_imported, 
                                '向量数': vector_count,
                                '失败': failed_count,
                                '平均速度': f'{avg_speed:.1f}条/s',
                                '当前速度': f'{recent_speed:.1f}条/s',
                                '设备': device_info
                            })
                            
                    except Exception as e:
                        logger.error(f"批量导入异常: {e}")
                        failed_count += len(actions)
                        actions = []
                
                progress_bar.update(1)
                
            except Exception as e:
                failed_count += 1
                progress_bar.update(1)
                if failed_count <= 10:
                    print(f"第{line_num}行处理失败: {str(e)[:100]}")

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
    
    print(f"\n{'=' * 60}")
    print(f"导入完成!")
    print(f"{'=' * 60}")
    print(f"成功导入: {total_imported}条")
    print(f"失败: {failed_count}条")
    print(f"生成向量总数: {vector_count}个")
    print(f"总耗时: {elapsed_time:.1f}秒 ({elapsed_time / 60:.1f}分钟)")
    avg_speed = total_imported / elapsed_time if elapsed_time > 0 else 0
    print(f"平均速度: {avg_speed:.2f}条/秒")
    # 现在每个文档可能有4个向量（descriptions_zh/en + label_zh/en）
    print(f"向量生成率: {vector_count / (total_imported * 4) * 100:.1f}% (每个文档最多4个向量)")

    # 显示设备使用情况
    device_info = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"使用设备: {device_info}")
    if torch.cuda.is_available():
        gpu_memory_used = torch.cuda.max_memory_allocated() / 1024 ** 3  # GB
        print(f"GPU显存使用: {gpu_memory_used:.2f} GB")
    print(f"{'=' * 60}")
    
    # 获取最终统计
    try:
        doc_count_after = es.count(index=INDEX_NAME)["count"]
        actual_imported = doc_count_after - doc_count_before
        print(f"实际新增文档数: {actual_imported}")
        print(f"导入后索引中文档总数: {doc_count_after}")
        
        # 使用 exists 查询检查向量字段（dense_vector 字段不会出现在 _source 中）
        print("\n使用 exists 查询检查向量字段（dense_vector 字段默认不在 _source 中）:")
        
        vector_fields = ["descriptions_zh_vector", "descriptions_en_vector", "label_zh_vector", "label_en_vector"]
        field_stats = {}
        
        for field in vector_fields:
            try:
                query = {
                    "query": {
                        "exists": {"field": field}
                    },
                    "size": 0
                }
                result = es.search(index=INDEX_NAME, body=query)
                count = result["hits"]["total"]["value"]
                field_stats[field] = count
                percentage = (count / doc_count_after * 100) if doc_count_after > 0 else 0
                print(f"  ✓ {field}: {count} 个文档 ({percentage:.1f}%)")
            except Exception as e:
                print(f"  ✗ 查询 {field} 失败: {e}")
                field_stats[field] = 0
        
        # 检查样本文档的标签（用于显示）
        sample_query = {
            "size": 10,
            "_source": ["label"]
        }
        sample_result = es.search(index=INDEX_NAME, body=sample_query)
        print(f"\n样本文档标签（前10个）:")
        for hit in sample_result['hits']['hits']:
            label = hit['_source'].get('label', 'N/A')
            print(f"  - {label}")
        
        # 总结
        total_vectors = sum(field_stats.values())
        expected_vectors = doc_count_after * 4  # 每个文档应该有4个向量
        vector_coverage = (total_vectors / expected_vectors * 100) if expected_vectors > 0 else 0
        print(f"\n向量字段统计:")
        print(f"  总文档数: {doc_count_after}")
        print(f"  总向量字段数: {total_vectors} (期望: {expected_vectors})")
        print(f"  向量覆盖率: {vector_coverage:.1f}%")
        
        if vector_coverage >= 95:
            print(f"  ✅ 向量导入成功！")
        elif vector_coverage >= 50:
            print(f"  ⚠️  向量覆盖率较低，可能部分文档缺少向量")
        else:
            print(f"  ❌ 向量覆盖率过低，请检查导入过程")
        
    except Exception as e:
        print(f"获取统计信息失败: {e}")
        logger.error(f"获取统计信息失败详情: {e}")


if __name__ == "__main__":
    import os
    
    print("=" * 60)
    print("开始向量化导入流程")
    print("=" * 60)
    
    # 先创建正确的索引映射
    print("创建向量索引映射...")
    if not create_vector_index():
        print("索引创建失败，退出")
        exit(1)
    
    # 处理数据文件
    data_files = [
        "zh_wiki_v2.jsonl",
        "en_wiki_v3.jsonl"
    ]
    
    processed_files = []
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"\n处理数据文件: {file_path}")
            # 由于每个文档现在有4个向量字段，减小批量大小以避免超过ES限制
            process_and_import_to_es(file_path, batch_size=20, request_timeout=180)
            processed_files.append(file_path)
        else:
            print(f"警告: 数据文件 {file_path} 不存在")
    
    if not processed_files:
        print("错误: 未找到任何数据文件")
        exit(1)
    
    print(f"\n已完成处理以下文件:")
    for f in processed_files:
        print(f"  - {f}")
    
    print("\n导入流程完成! 请检查阿里云控制台的向量存储用量")