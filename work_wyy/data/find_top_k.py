#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
预处理脚本：为每条数据提取其description内部的高频词
对于每条数据，分别统计其zh_description和en_description内部的词频，
找出该条数据的高频词集并保存
"""

import json
import os
from collections import Counter
import re
from tqdm import tqdm
import logging

# 尝试导入jieba，如果未安装则使用简单的中文分词方法
try:
    import jieba

    USE_JIEBA = True
except ImportError:
    USE_JIEBA = False
    print("警告: jieba未安装，将使用简单的中文分词方法")
    print("建议安装jieba以获得更好的分词效果: pip install jieba")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../preprocess_high_frequency.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def extract_words_zh(text):
    """提取中文文本中的词（使用jieba分词或简单方法）"""
    if not text or not isinstance(text, str):
        return []

    if USE_JIEBA:
        # 使用jieba分词
        words = jieba.cut(text, cut_all=False)
        # 过滤：只保留长度>=2的词，去除标点和纯数字
        words = [w.strip() for w in words if len(w.strip()) >= 2 and not re.match(r'^[\d\s]+$', w)]
    else:
        # 简单方法：使用正则表达式提取中文字符（2个或以上连续的中文字符）
        words = re.findall(r'[\u4e00-\u9fa5]{2,}', text)

    return words


def extract_words_en(text):
    """提取英文文本中的词（使用正则表达式）"""
    if not text or not isinstance(text, str):
        return []

    # 使用正则表达式提取英文单词（长度>=2）
    words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
    return words


def get_high_freq_words_in_text(text, lang='zh', min_freq=2, top_n=None):
    """
    获取文本内部的高频词

    Args:
        text: 文本内容
        lang: 语言类型 ('zh' 或 'en')
        min_freq: 词的最小出现频率（出现次数>=min_freq才算高频词）
        top_n: 返回前N个高频词，None表示返回所有满足条件的词

    Returns:
        high_freq_words: 高频词列表，按频率降序排列
        word_freq: 词频字典
    """
    if not text or not isinstance(text, str):
        return [], {}

    # 提取词
    if lang == 'zh':
        words = extract_words_zh(text)
    else:
        words = extract_words_en(text)

    if not words:
        return [], {}

    # 统计词频
    word_freq = Counter(words)

    # 找出高频词（出现次数>=min_freq）
    high_freq_words = [(word, freq) for word, freq in word_freq.items() if freq >= min_freq]

    # 按频率降序排序
    high_freq_words.sort(key=lambda x: x[1], reverse=True)

    # 如果指定了top_n，只返回前N个
    if top_n is not None and top_n > 0:
        high_freq_words = high_freq_words[:top_n]

    # 返回词列表和词频字典
    high_freq_word_list = [word for word, freq in high_freq_words]
    word_freq_dict = {word: freq for word, freq in high_freq_words}

    return high_freq_word_list, word_freq_dict


def count_lines(filename):
    """快速计算文件行数"""
    with open(filename, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def process_file(file_path, min_freq_zh=2, min_freq_en=2, top_n_zh=None, top_n_en=None):
    """
    处理单个文件，为每条数据提取其description内部的高频词

    Args:
        file_path: 文件路径
        min_freq_zh: 中文词的最小出现频率
        min_freq_en: 英文词的最小出现频率
        top_n_zh: 中文高频词最多返回多少个，None表示不限制
        top_n_en: 英文高频词最多返回多少个，None表示不限制

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
                if zh_description and len(zh_description.strip()) >= 10:
                    high_freq_words_zh, word_freq_zh = get_high_freq_words_in_text(
                        zh_description, lang='zh', min_freq=min_freq_zh, top_n=top_n_zh
                    )
                    if high_freq_words_zh:
                        entity_data['_high_freq_words_zh'] = high_freq_words_zh
                        entity_data['_word_freq_zh'] = word_freq_zh
                        entity_data['_high_freq_count_zh'] = len(high_freq_words_zh)
                        valid_zh_count += 1
                    else:
                        entity_data['_high_freq_words_zh'] = []
                        entity_data['_word_freq_zh'] = {}
                        entity_data['_high_freq_count_zh'] = 0
                else:
                    entity_data['_high_freq_words_zh'] = []
                    entity_data['_word_freq_zh'] = {}
                    entity_data['_high_freq_count_zh'] = 0

                # 处理英文描述
                en_description = data.get("en_description") or data.get("descriptions_en", "")
                if en_description and len(en_description.strip()) >= 10:
                    high_freq_words_en, word_freq_en = get_high_freq_words_in_text(
                        en_description, lang='en', min_freq=min_freq_en, top_n=top_n_en
                    )
                    if high_freq_words_en:
                        entity_data['_high_freq_words_en'] = high_freq_words_en
                        entity_data['_word_freq_en'] = word_freq_en
                        entity_data['_high_freq_count_en'] = len(high_freq_words_en)
                        valid_en_count += 1
                    else:
                        entity_data['_high_freq_words_en'] = []
                        entity_data['_word_freq_en'] = {}
                        entity_data['_high_freq_count_en'] = 0
                else:
                    entity_data['_high_freq_words_en'] = []
                    entity_data['_word_freq_en'] = {}
                    entity_data['_high_freq_count_en'] = 0

                # 计算总的高频词数量
                entity_data['_high_freq_count_total'] = (
                        entity_data.get('_high_freq_count_zh', 0) +
                        entity_data.get('_high_freq_count_en', 0)
                )

                # 保存所有数据（不管有没有高频词都保存）
                processed_entities.append(entity_data)

                progress_bar.update(1)

                if processed_count % 10000 == 0:
                    logger.info(
                        f"已处理 {processed_count} 条，有效中文描述 {valid_zh_count} 条，有效英文描述 {valid_en_count} 条")

            except Exception as e:
                logger.warning(f"第{line_num}行处理失败: {e}")
                progress_bar.update(1)
                continue

        progress_bar.close()

    logger.info(f"处理完成: 总处理 {processed_count} 条")
    logger.info(f"  有效中文描述（有高频词）: {valid_zh_count} 条")
    logger.info(f"  有效英文描述（有高频词）: {valid_en_count} 条")

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
    print("高频词预处理脚本")
    print("为每条数据提取其description内部的高频词")
    print("=" * 70)

    # 文件路径和对应的输出文件
    file_configs = [
        {
            "input": "./data/zh_wiki_v2.jsonl",
            "output": "top_k_zh.jsonl",
            "name": "中文"
        },
        {
            "input": "./data/en_wiki_v3.jsonl",
            "output": "top_k_en.jsonl",
            "name": "英文"
        }
    ]

    # 参数设置
    min_freq_zh = 2  # 中文词至少出现2次才算高频词
    min_freq_en = 2  # 英文词至少出现2次才算高频词
    top_n_zh = None  # None表示不限制中文高频词数量
    top_n_en = None  # None表示不限制英文高频词数量

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
            min_freq_zh=min_freq_zh,
            min_freq_en=min_freq_en,
            top_n_zh=top_n_zh,
            top_n_en=top_n_en
        )
        
        if not entities:
            logger.warning(f"{file_name}文件未处理到任何数据")
            continue
        
        # 保存结果
        logger.info(f"\n{file_name}文件处理完成，实体总数: {len(entities)} 条")
        save_entities_to_file(entities, output_file)
        processed_files.append(output_file)
        
        # 显示统计信息
        logger.info(f"\n[{file_name}文件统计信息]")
        entities_with_zh_freq = sum(1 for e in entities if e.get('_high_freq_count_zh', 0) > 0)
        entities_with_en_freq = sum(1 for e in entities if e.get('_high_freq_count_en', 0) > 0)
        entities_with_any_freq = sum(1 for e in entities if e.get('_high_freq_count_total', 0) > 0)
        
        avg_zh_freq = sum(e.get('_high_freq_count_zh', 0) for e in entities) / len(entities)
        avg_en_freq = sum(e.get('_high_freq_count_en', 0) for e in entities) / len(entities)
        avg_total_freq = sum(e.get('_high_freq_count_total', 0) for e in entities) / len(entities)
        
        logger.info(f"  总实体数: {len(entities)}")
        logger.info(f"  有中文高频词的实体: {entities_with_zh_freq} 条 ({entities_with_zh_freq / len(entities) * 100:.1f}%)")
        logger.info(f"  有英文高频词的实体: {entities_with_en_freq} 条 ({entities_with_en_freq / len(entities) * 100:.1f}%)")
        logger.info(f"  有任何高频词的实体: {entities_with_any_freq} 条 ({entities_with_any_freq / len(entities) * 100:.1f}%)")
        logger.info(f"  平均中文高频词数: {avg_zh_freq:.2f}")
        logger.info(f"  平均英文高频词数: {avg_en_freq:.2f}")
        logger.info(f"  平均总高频词数: {avg_total_freq:.2f}")
        
        # 显示前5个高频词最多的实体
        sorted_entities = sorted(entities, key=lambda x: x.get('_high_freq_count_total', 0), reverse=True)
        logger.info(f"\n  {file_name}文件前5个高频词最多的实体:")
        for i, entity in enumerate(sorted_entities[:5], 1):
            label = entity.get('label', 'N/A')
            zh_count = entity.get('_high_freq_count_zh', 0)
            en_count = entity.get('_high_freq_count_en', 0)
            total_count = entity.get('_high_freq_count_total', 0)
            logger.info(f"    {i}. {label}: 中文{zh_count}个, 英文{en_count}个, 总计{total_count}个")

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


if __name__ == "__main__":
    main()
