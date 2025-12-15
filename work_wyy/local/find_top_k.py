#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
实体词提取与向量化脚本
为每条数据提取其description中的实体词（使用NER模型），并对实体词进行向量化

修改说明（2025-12-08）：
1. 移除高频词统计功能，只使用NER模型提取实体词
2. 对提取的实体词使用微调后的模型进行向量化
3. 保存实体词列表和对应的向量，供后续存入ES使用
"""

import json
import os
import sys
from tqdm import tqdm
import logging
from datetime import datetime

# 导入配置
try:
    from config import (
        TRAINLOG_DIR,
        ENTITY_WORDS_ZH_FILE,
        ENTITY_WORDS_EN_FILE,
        ZH_WIKI_FILE,
        EN_WIKI_FILE,
        MIN_TEXT_LENGTH,
        MIN_ENTITY_LENGTH_ZH,
        MIN_ENTITY_LENGTH_EN,
        VECTOR_BATCH_SIZE,
        USE_FINETUNED_FOR_VECTORIZATION
    )
except ImportError:
    # 如果无法导入配置，使用默认值
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _parent_dir = os.path.dirname(_script_dir)
    TRAINLOG_DIR = os.path.join(_parent_dir, 'trainlog')
    ENTITY_WORDS_ZH_FILE = os.path.join(_parent_dir, 'data', 'entity_words_zh.jsonl')
    ENTITY_WORDS_EN_FILE = os.path.join(_parent_dir, 'data', 'entity_words_en.jsonl')
    ZH_WIKI_FILE = os.path.join(_parent_dir, 'data', 'zh_wiki_v2.jsonl')
    EN_WIKI_FILE = os.path.join(_parent_dir, 'data', 'en_wiki_v3.jsonl')
    MIN_TEXT_LENGTH = 2
    MIN_ENTITY_LENGTH_ZH = 2
    MIN_ENTITY_LENGTH_EN = 3
    VECTOR_BATCH_SIZE = 64
    USE_FINETUNED_FOR_VECTORIZATION = True

# 添加父目录到路径，以便导入模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 导入NER提取模块
try:
    from ner.ner_extract_entities import get_entity_words_from_text
except ImportError:
    print("错误: 无法导入NER模块，请确保 ner/ner_extract_entities.py 存在")
    sys.exit(1)

# 导入向量生成模块
try:
    from vector_model import batch_generate_vectors
except ImportError:
    print("错误: 无法导入向量生成模块，请确保 vector_model.py 存在")
    sys.exit(1)

# 创建 trainlog 文件夹（如果不存在）
os.makedirs(TRAINLOG_DIR, exist_ok=True)

# 设置日志（保存到 trainlog 文件夹）
log_filename = os.path.join(TRAINLOG_DIR, f'extract_entity_words_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
console_log_file = os.path.join(TRAINLOG_DIR, f'find_top_k_console_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

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

logger.info(f"日志文件: {log_filename}")
logger.info(f"控制台输出文件: {console_log_file}")
logger.info("=" * 70)
logger.info("实体词提取与向量化脚本")
logger.info("使用NER模型提取实体词，并使用微调后的模型进行向量化")
logger.info("=" * 70)


def extract_entity_words_from_text(text, lang='zh'):
    """
    从文本中提取实体词（只使用NER模型，不使用词频统计）
    
    Args:
        text: 文本内容
        lang: 语言类型 ('zh' 或 'en')
    
    Returns:
        entity_words: 实体词列表
        entity_freq: 实体词频字典（每个实体出现1次）
    """
    if not text or not isinstance(text, str) or len(text.strip()) < MIN_TEXT_LENGTH:
        return [], {}  # 使用配置文件中的最小长度要求
    
    try:
        # 使用NER模型提取实体词
        entity_words, entity_freq = get_entity_words_from_text(
            text, lang=lang, verbose=False
        )
        
        # 过滤：使用配置文件中的最小长度要求
        if lang == 'zh':
            entity_words = [e for e in entity_words if len(e.strip()) >= MIN_ENTITY_LENGTH_ZH]
        else:
            entity_words = [e for e in entity_words if len(e.strip()) >= MIN_ENTITY_LENGTH_EN]
        
        # 更新词频字典
        entity_freq = {e: 1 for e in entity_words}
        
        return entity_words, entity_freq
        
    except Exception as e:
        logger.warning(f"NER提取失败: {e}")
        return [], {}


def vectorize_entity_words_and_merge(entity_words, lang='zh', batch_size=32, merge_method='mean'):
    """
    对实体词列表进行向量化，然后合并成一个向量（使用微调后的模型）
    
    Args:
        entity_words: 实体词列表
        lang: 语言类型（用于日志）
        batch_size: 批量处理大小
        merge_method: 合并方法 ('mean' 平均, 'max' 最大值, 'sum' 求和)
    
    Returns:
        merged_vector: 合并后的向量（1024维），如果失败返回None
    """
    if not entity_words:
        return None
    
    try:
        # 使用批量向量生成（使用微调后的模型）
        vectors = batch_generate_vectors(
            entity_words,
            use_finetuned=USE_FINETUNED_FOR_VECTORIZATION,  # 使用配置文件中的设置
            target_dim=1024,
            batch_size=batch_size
        )
        
        # 过滤掉None值
        valid_vectors = [v for v in vectors if v is not None and isinstance(v, list)]
        
        if not valid_vectors:
            logger.warning(f"所有实体词向量化失败: {len(entity_words)} 个实体词")
            return None
        
        if len(valid_vectors) != len(entity_words):
            logger.debug(f"部分实体词向量化失败: {len(valid_vectors)}/{len(entity_words)}")
        
        # 转换为numpy数组进行合并
        import numpy as np
        vectors_array = np.array(valid_vectors)
        
        # 根据合并方法合并向量
        if merge_method == 'mean':
            merged_vector = np.mean(vectors_array, axis=0)
        elif merge_method == 'max':
            merged_vector = np.max(vectors_array, axis=0)
        elif merge_method == 'sum':
            merged_vector = np.sum(vectors_array, axis=0)
        else:
            merged_vector = np.mean(vectors_array, axis=0)  # 默认使用平均
        
        # L2归一化
        norm = np.linalg.norm(merged_vector)
        if norm > 0:
            merged_vector = merged_vector / norm
        else:
            return None
        
        return merged_vector.tolist()
        
    except Exception as e:
        logger.error(f"实体词向量化失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def count_lines(filename):
    """快速计算文件行数"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except Exception as e:
        logger.error(f"计算文件行数失败: {e}")
        return 0


def process_file(file_path, vectorize=True, batch_size=32):
    """
    处理单个文件，为每条数据提取实体词并进行向量化
    
    Args:
        file_path: 文件路径
        vectorize: 是否对实体词进行向量化
        batch_size: 向量化批量处理大小
    
    Returns:
        processed_entities: 处理后的实体列表
    """
    if not os.path.exists(file_path):
        logger.warning(f"文件不存在: {file_path}")
        return []
    
    logger.info(f"处理文件: {file_path}")
    total_lines = count_lines(file_path)
    logger.info(f"文件总行数: {total_lines}")
    
    processed_entities = []
    processed_count = 0
    valid_zh_count = 0
    valid_en_count = 0
    vectorized_zh_count = 0
    vectorized_en_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        progress_bar = tqdm(total=total_lines, desc="处理数据", unit="条")
        
        for line_num, line in enumerate(f, 1):
            try:
                if not line.strip():
                    progress_bar.update(1)
                    continue
                
                data = json.loads(line.strip())
                processed_count += 1
                
                # 创建新的数据对象
                entity_data = data.copy()
                
                # 处理中文描述
                zh_description = data.get("zh_description") or data.get("descriptions_zh", "")
                if zh_description and len(zh_description.strip()) >= MIN_TEXT_LENGTH:  # 使用配置文件中的最小长度要求
                    entity_words_zh, entity_freq_zh = extract_entity_words_from_text(
                        zh_description, lang='zh'
                    )
                    
                    if entity_words_zh:
                        entity_data['_entity_words_zh'] = entity_words_zh
                        entity_data['_entity_freq_zh'] = entity_freq_zh
                        entity_data['_entity_count_zh'] = len(entity_words_zh)
                        valid_zh_count += 1
                        
                        # 如果需要向量化，对实体词进行向量化并合并
                        if vectorize:
                            entity_vector_zh = vectorize_entity_words_and_merge(
                                entity_words_zh, lang='zh', batch_size=batch_size, merge_method='mean'
                            )
                            if entity_vector_zh:
                                entity_data['_entity_words_zh_vector'] = entity_vector_zh
                                vectorized_zh_count += 1
                            else:
                                entity_data['_entity_words_zh_vector'] = None
                    else:
                        entity_data['_entity_words_zh'] = []
                        entity_data['_entity_freq_zh'] = {}
                        entity_data['_entity_count_zh'] = 0
                        if vectorize:
                            entity_data['_entity_words_zh_vector'] = None
                else:
                    entity_data['_entity_words_zh'] = []
                    entity_data['_entity_freq_zh'] = {}
                    entity_data['_entity_count_zh'] = 0
                    if vectorize:
                        entity_data['_entity_words_zh_vector'] = None
                
                # 处理英文描述
                en_description = data.get("en_description") or data.get("descriptions_en", "")
                if en_description and len(en_description.strip()) >= MIN_TEXT_LENGTH:  # 使用配置文件中的最小长度要求
                    entity_words_en, entity_freq_en = extract_entity_words_from_text(
                        en_description, lang='en'
                    )
                    
                    if entity_words_en:
                        entity_data['_entity_words_en'] = entity_words_en
                        entity_data['_entity_freq_en'] = entity_freq_en
                        entity_data['_entity_count_en'] = len(entity_words_en)
                        valid_en_count += 1
                        
                        # 如果需要向量化，对实体词进行向量化并合并
                        if vectorize:
                            entity_vector_en = vectorize_entity_words_and_merge(
                                entity_words_en, lang='en', batch_size=batch_size, merge_method='mean'
                            )
                            if entity_vector_en:
                                entity_data['_entity_words_en_vector'] = entity_vector_en
                                vectorized_en_count += 1
                            else:
                                entity_data['_entity_words_en_vector'] = None
                    else:
                        entity_data['_entity_words_en'] = []
                        entity_data['_entity_freq_en'] = {}
                        entity_data['_entity_count_en'] = 0
                        if vectorize:
                            entity_data['_entity_words_en_vector'] = None
                else:
                    entity_data['_entity_words_en'] = []
                    entity_data['_entity_freq_en'] = {}
                    entity_data['_entity_count_en'] = 0
                    if vectorize:
                        entity_data['_entity_words_en_vector'] = None
                
                # 计算总的实体词数量
                entity_data['_entity_count_total'] = (
                    entity_data.get('_entity_count_zh', 0) +
                    entity_data.get('_entity_count_en', 0)
                )
                
                # 保存所有数据（不管有没有实体词都保存）
                processed_entities.append(entity_data)
                
                progress_bar.update(1)
                
                if processed_count % 10000 == 0:
                    logger.info(
                        f"已处理 {processed_count} 条，有效中文描述 {valid_zh_count} 条，有效英文描述 {valid_en_count} 条")
                    if vectorize:
                        logger.info(
                            f"  已向量化: 中文 {vectorized_zh_count} 条，英文 {vectorized_en_count} 条")
            
            except Exception as e:
                logger.warning(f"第{line_num}行处理失败: {e}")
                progress_bar.update(1)
                continue
        
        progress_bar.close()
    
    logger.info(f"处理完成: 总处理 {processed_count} 条")
    logger.info(f"  有效中文描述（有实体词）: {valid_zh_count} 条")
    logger.info(f"  有效英文描述（有实体词）: {valid_en_count} 条")
    if vectorize:
        logger.info(f"  已向量化: 中文 {vectorized_zh_count} 条，英文 {vectorized_en_count} 条")
    
    return processed_entities


def save_entities_to_file(entities, output_file):
    """保存实体到JSONL文件"""
    logger.info(f"保存 {len(entities)} 个实体到文件: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entity in entities:
            f.write(json.dumps(entity, ensure_ascii=False) + '\n')
    
    logger.info(f"保存完成: {output_file}")


def main():
    """主函数"""
    print("=" * 70)
    print("实体词提取与向量化脚本")
    print("使用NER模型提取实体词，并使用微调后的模型进行向量化")
    print("=" * 70)
    logger.info("=" * 70)
    logger.info("实体词提取与向量化脚本")
    logger.info("使用NER模型提取实体词，并使用微调后的模型进行向量化")
    logger.info("=" * 70)
    
    # 文件路径和对应的输出文件
    # 注意：输出文件保存在当前目录（work_wyy/data），与 find_top_k.py 同一目录
    file_configs = [
        {
            "input": ZH_WIKI_FILE,
            "output": ENTITY_WORDS_ZH_FILE,  # 使用配置文件中的路径
            "name": "中文"
        },
        {
            "input": EN_WIKI_FILE,
            "output": ENTITY_WORDS_EN_FILE,  # 使用配置文件中的路径
            "name": "英文"
        }
    ]
    
    # 参数设置（使用配置文件中的值）
    vectorize = True  # 是否对实体词进行向量化
    batch_size = VECTOR_BATCH_SIZE  # 使用配置文件中的批量处理大小
    
    processed_files = []
    
    # 分别处理每个文件
    for config in file_configs:
        input_file = config["input"]
        output_file = config["output"]
        file_name = config["name"]
        
        logger.info("\n" + "=" * 70)
        logger.info(f"处理{file_name}文件")
        logger.info("=" * 70)
        
        if not os.path.exists(input_file):
            logger.warning(f"文件不存在: {input_file}")
            continue
        
        # 处理文件
        entities = process_file(
            input_file,
            vectorize=vectorize,
            batch_size=batch_size
        )
        
        if not entities:
            logger.warning(f"{file_name}文件未处理到任何数据")
            continue
        
        # 保存结果
        logger.info(f"\n{file_name}文件处理完成，实体总数: {len(entities)} 条")
        save_entities_to_file(entities, output_file)
        processed_files.append(output_file)
        
        # 保存小范围数据（前5条）用于查看
        sample_output_file = output_file.replace('.jsonl', '_sample_5.jsonl')
        sample_entities = entities[:5] if len(entities) >= 5 else entities
        if sample_entities:
            save_entities_to_file(sample_entities, sample_output_file)
            logger.info(f"已保存小范围数据（{len(sample_entities)}条）到: {sample_output_file}")
        
        # 显示统计信息
        logger.info(f"\n[{file_name}文件统计信息]")
        entities_with_zh_entity = sum(1 for e in entities if e.get('_entity_count_zh', 0) > 0)
        entities_with_en_entity = sum(1 for e in entities if e.get('_entity_count_en', 0) > 0)
        entities_with_any_entity = sum(1 for e in entities if e.get('_entity_count_total', 0) > 0)
        
        avg_zh_entity = sum(e.get('_entity_count_zh', 0) for e in entities) / len(entities) if entities else 0
        avg_en_entity = sum(e.get('_entity_count_en', 0) for e in entities) / len(entities) if entities else 0
        avg_total_entity = sum(e.get('_entity_count_total', 0) for e in entities) / len(entities) if entities else 0
        
        logger.info(f"  总实体数: {len(entities)}")
        logger.info(f"  有中文实体词的实体: {entities_with_zh_entity} 条 ({entities_with_zh_entity / len(entities) * 100:.1f}%)")
        logger.info(f"  有英文实体词的实体: {entities_with_en_entity} 条 ({entities_with_en_entity / len(entities) * 100:.1f}%)")
        logger.info(f"  有任何实体词的实体: {entities_with_any_entity} 条 ({entities_with_any_entity / len(entities) * 100:.1f}%)")
        logger.info(f"  平均中文实体词数: {avg_zh_entity:.2f}")
        logger.info(f"  平均英文实体词数: {avg_en_entity:.2f}")
        logger.info(f"  平均总实体词数: {avg_total_entity:.2f}")
        
        # 显示前5个实体词最多的实体
        sorted_entities = sorted(entities, key=lambda x: x.get('_entity_count_total', 0), reverse=True)
        logger.info(f"\n  {file_name}文件前5个实体词最多的实体:")
        for i, entity in enumerate(sorted_entities[:5], 1):
            label = entity.get('label', 'N/A')
            zh_count = entity.get('_entity_count_zh', 0)
            en_count = entity.get('_entity_count_en', 0)
            total_count = entity.get('_entity_count_total', 0)
            logger.info(f"    {i}. {label}: 中文{zh_count}个, 英文{en_count}个, 总计{total_count}个")
            if zh_count > 0:
                logger.info(f"       中文实体词示例: {entity.get('_entity_words_zh', [])[:5]}")
            if en_count > 0:
                logger.info(f"       英文实体词示例: {entity.get('_entity_words_en', [])[:5]}")
    
    if not processed_files:
        logger.error("未处理任何文件")
        return
    
    print("\n" + "=" * 70)
    print("预处理完成!")
    print("=" * 70)
    print("生成的文件:")
    for output_file in processed_files:
        print(f"  - {output_file}")
    print("=" * 70)
    logger.info(f"所有日志已保存到: {log_filename}")


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
