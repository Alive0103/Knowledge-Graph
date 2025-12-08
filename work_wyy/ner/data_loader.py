#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据加载模块
统一处理多种格式的NER数据加载
"""

import json
import os
import glob
import logging

logger = logging.getLogger(__name__)


def detect_file_encoding(file_path):
    """检测文件编码格式"""
    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'utf-16', 'latin1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                f.read()
            return encoding
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception:
            continue
    
    return 'utf-8'


def load_ner_data_from_json(file_path):
    """从JSON数组格式加载NER训练数据"""
    examples = []
    entity_types_set = set()
    
    if not os.path.exists(file_path):
        logger.warning(f"文件不存在: {file_path}")
        return examples, entity_types_set
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            if not isinstance(data, list):
                logger.warning(f"文件 {file_path} 不是JSON数组格式")
                return examples, entity_types_set
            
            for item_num, item in enumerate(data, 1):
                try:
                    text = item.get('text', '')
                    entities = item.get('entities', [])
                    
                    if not text:
                        continue
                    
                    labels = ['O'] * len(text)
                    
                    for entity in entities:
                        start = entity.get('start', -1)
                        end = entity.get('end', -1)
                        entity_text = entity.get('text', '')
                        entity_type = entity.get('type', '')
                        
                        if entity_type:
                            entity_types_set.add(entity_type)
                        
                        if start >= 0 and end > start and end <= len(text):
                            if text[start:end] != entity_text:
                                logger.warning(f"实体文本不匹配: 期望 '{entity_text}', 实际 '{text[start:end]}'")
                            
                            labels[start] = f'B-{entity_type}' if entity_type else 'B-ENTITY'
                            for i in range(start + 1, end):
                                if i < len(labels):
                                    labels[i] = f'I-{entity_type}' if entity_type else 'I-ENTITY'
                    
                    examples.append({
                        'text': text,
                        'labels': labels
                    })
                    
                except Exception as e:
                    logger.warning(f"第{item_num}条数据解析失败: {e}")
                    continue
        
        logger.debug(f"成功加载 {len(examples)} 条数据，发现 {len(entity_types_set)} 种实体类型")
        if entity_types_set:
            logger.debug(f"实体类型: {', '.join(sorted(entity_types_set))}")
        
    except Exception as e:
        logger.error(f"加载文件 {file_path} 失败: {e}")
    
    return examples, entity_types_set


def load_ccks_json_format(file_path):
    """加载CCKS格式的JSON数据"""
    examples = []
    entity_types_set = set()
    
    if not os.path.exists(file_path):
        logger.warning(f"文件不存在: {file_path}")
        return examples, entity_types_set
    
    # 尝试多种编码加载文件
    encodings_to_try = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'utf-16', 'latin1', 'cp1252']
    data = None
    used_encoding = None
    
    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                data = json.load(f)
            used_encoding = encoding
            break
        except UnicodeDecodeError:
            continue
        except json.JSONDecodeError:
            continue
        except Exception:
            continue
    
    if data is None:
        logger.error(f"无法加载文件 {file_path}，所有编码尝试都失败")
        return examples, entity_types_set
    
    # 处理加载的数据
    try:
        # 格式1：字典格式（key为文件名，value为文本）
        if isinstance(data, dict) and not any(key in data for key in ['text', 'originalText', 'entities']):
            for key, text in data.items():
                if not text:
                    continue
                examples.append({
                    'text': text,
                    'labels': ['O'] * len(text)
                })
        
        # 格式2：CCKS训练数据格式（originalText + entities）
        elif isinstance(data, dict) and 'originalText' in data:
            text = data.get('originalText', '')
            entities = data.get('entities', [])
            
            if text:
                labels = ['O'] * len(text)
                
                for entity in entities:
                    start = entity.get('start_pos', entity.get('start', -1))
                    end = entity.get('end_pos', entity.get('end', -1))
                    entity_type = entity.get('label_type', entity.get('type', ''))
                    
                    if entity_type:
                        entity_types_set.add(entity_type)
                    
                    if start >= 0 and end > start and end <= len(text):
                        labels[start] = f'B-{entity_type}' if entity_type else 'B-ENTITY'
                        for i in range(start + 1, end):
                            if i < len(labels):
                                labels[i] = f'I-{entity_type}' if entity_type else 'I-ENTITY'
                
                examples.append({
                    'text': text,
                    'labels': labels
                })
        
        # 格式3：标准格式（text + entities）
        elif isinstance(data, dict) and 'text' in data:
            text = data.get('text', '')
            entities = data.get('entities', [])
            
            if text:
                labels = ['O'] * len(text)
                
                for entity in entities:
                    start = entity.get('start', -1)
                    end = entity.get('end', -1)
                    entity_type = entity.get('type', '')
                    
                    if entity_type:
                        entity_types_set.add(entity_type)
                    
                    if start >= 0 and end > start and end <= len(text):
                        labels[start] = f'B-{entity_type}' if entity_type else 'B-ENTITY'
                        for i in range(start + 1, end):
                            if i < len(labels):
                                labels[i] = f'I-{entity_type}' if entity_type else 'I-ENTITY'
                
                examples.append({
                    'text': text,
                    'labels': labels
                })
        
        # 格式4：列表格式
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    text = item.get('originalText', item.get('text', ''))
                    entities = item.get('entities', [])
                    
                    if not text:
                        continue
                    
                    labels = ['O'] * len(text)
                    
                    for entity in entities:
                        start = entity.get('start_pos', entity.get('start', -1))
                        end = entity.get('end_pos', entity.get('end', -1))
                        entity_type = entity.get('label_type', entity.get('type', ''))
                        
                        if entity_type:
                            entity_types_set.add(entity_type)
                        
                        if start >= 0 and end > start and end <= len(text):
                            labels[start] = f'B-{entity_type}' if entity_type else 'B-ENTITY'
                            for i in range(start + 1, end):
                                if i < len(labels):
                                    labels[i] = f'I-{entity_type}' if entity_type else 'I-ENTITY'
                    
                    examples.append({
                        'text': text,
                        'labels': labels
                    })
        
        if examples:
            logger.debug(f"成功加载 {len(examples)} 条数据（编码: {used_encoding}），发现 {len(entity_types_set)} 种实体类型")
            if entity_types_set:
                logger.debug(f"实体类型: {', '.join(sorted(entity_types_set))}")
    
    except Exception as e:
        logger.error(f"处理文件 {file_path} 的数据时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    return examples, entity_types_set


def load_bio_format(file_path):
    """加载BIO格式的数据（MSRA格式：每行"字符\t标签"）"""
    examples = []
    entity_types_set = set()
    
    if not os.path.exists(file_path):
        logger.warning(f"文件不存在: {file_path}")
        return examples, entity_types_set
    
    logger.info(f"加载BIO格式数据: {file_path}")
    
    current_text = []
    current_labels = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                if not line:
                    if current_text:
                        examples.append({
                            'text': ''.join(current_text),
                            'labels': current_labels
                        })
                        current_text = []
                        current_labels = []
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 2:
                    char = parts[0]
                    label = parts[1]
                    
                    current_text.append(char)
                    current_labels.append(label)
                    
                    if label.startswith('B-') or label.startswith('I-'):
                        entity_type = label[2:]
                        if entity_type:
                            entity_types_set.add(entity_type)
                elif len(parts) == 1:
                    current_text.append(parts[0])
                    current_labels.append('O')
            
            if current_text:
                examples.append({
                    'text': ''.join(current_text),
                    'labels': current_labels
                })
        
        logger.info(f"成功加载 {len(examples)} 条数据，发现 {len(entity_types_set)} 种实体类型")
        if entity_types_set:
            logger.info(f"实体类型: {', '.join(sorted(entity_types_set))}")
    
    except Exception as e:
        logger.error(f"加载文件 {file_path} 失败: {e}")
    
    return examples, entity_types_set


def load_ccks_bio_format(sentences_file, tags_file):
    """加载CCKS的BIO格式数据（sentences.txt + tags.txt分开存储）"""
    examples = []
    entity_types_set = set()
    
    if not os.path.exists(sentences_file) or not os.path.exists(tags_file):
        logger.warning(f"文件不存在: {sentences_file} 或 {tags_file}")
        return examples, entity_types_set
    
    logger.info(f"加载CCKS BIO格式数据: {sentences_file}, {tags_file}")
    
    try:
        with open(sentences_file, 'r', encoding='utf-8') as f_sent, \
             open(tags_file, 'r', encoding='utf-8') as f_tags:
            
            for line_num, (sent_line, tag_line) in enumerate(zip(f_sent, f_tags), 1):
                sent_line = sent_line.strip()
                tag_line = tag_line.strip()
                
                if not sent_line or not tag_line:
                    continue
                
                chars = sent_line.split()
                tags = tag_line.split()
                
                if len(chars) != len(tags):
                    logger.warning(f"第{line_num}行：字符数({len(chars)})与标签数({len(tags)})不匹配，跳过")
                    continue
                
                text = ''.join(chars)
                labels = tags
                
                for label in labels:
                    if label.startswith('B-') or label.startswith('I-'):
                        entity_type = label[2:]
                        if entity_type:
                            entity_types_set.add(entity_type)
                
                examples.append({
                    'text': text,
                    'labels': labels
                })
        
        logger.info(f"成功加载 {len(examples)} 条数据，发现 {len(entity_types_set)} 种实体类型")
        if entity_types_set:
            logger.info(f"实体类型: {', '.join(sorted(entity_types_set))}")
    
    except Exception as e:
        logger.error(f"加载文件失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    return examples, entity_types_set


def load_all_traindata(data_dir, file_pattern="*_ner_train.json"):
    """加载traindata目录下的所有训练文件"""
    all_examples = []
    all_entity_types = set()
    
    pattern = os.path.join(data_dir, file_pattern)
    files = sorted(glob.glob(pattern))
    
    if not files:
        logger.warning(f"未找到匹配的文件: {pattern}")
        return all_examples, all_entity_types
    
    logger.info(f"找到 {len(files)} 个训练文件")
    
    for file_path in files:
        examples, entity_types = load_ner_data_from_json(file_path)
        all_examples.extend(examples)
        all_entity_types.update(entity_types)
    
    logger.info(f"总共加载 {len(all_examples)} 条训练数据，{len(all_entity_types)} 种实体类型")
    return all_examples, all_entity_types


def load_all_devdata(data_dir, file_pattern="*_ner_dev.json"):
    """加载traindata目录下的所有验证文件"""
    all_examples = []
    all_entity_types = set()
    
    pattern = os.path.join(data_dir, file_pattern)
    files = sorted(glob.glob(pattern))
    
    if not files:
        logger.warning(f"未找到匹配的文件: {pattern}")
        return all_examples, all_entity_types
    
    logger.info(f"找到 {len(files)} 个验证文件")
    
    for file_path in files:
        examples, entity_types = load_ner_data_from_json(file_path)
        all_examples.extend(examples)
        all_entity_types.update(entity_types)
    
    logger.info(f"总共加载 {len(all_examples)} 条验证数据，{len(all_entity_types)} 种实体类型")
    return all_examples, all_entity_types


def load_all_data_from_directories(base_dir):
    """从多个数据目录加载所有数据"""
    
    all_train_examples = []
    all_dev_examples = []
    all_entity_types = set()
    
    # 1. 从traindata目录加载
    traindata_dir = os.path.join(base_dir, "traindata")
    if os.path.exists(traindata_dir):
        logger.info("=" * 50)
        logger.info("从traindata目录加载数据...")
        train_examples, train_types = load_all_traindata(traindata_dir, "*_ner_train.json")
        dev_examples, dev_types = load_all_devdata(traindata_dir, "*_ner_dev.json")
        all_train_examples.extend(train_examples)
        all_dev_examples.extend(dev_examples)
        all_entity_types.update(train_types)
        all_entity_types.update(dev_types)
    
    # 2. 从ccks_ner目录加载
    ccks_dir = os.path.join(base_dir, "ccks_ner", "militray", "PreModel_Encoder_CRF")
    ccks_data_dir = os.path.join(ccks_dir, "ccks_8_data_v2")
    
    train_json_dir = os.path.join(ccks_data_dir, "train")
    if os.path.exists(train_json_dir):
        logger.info("=" * 50)
        logger.info("从ccks_ner/militray加载训练数据（JSON格式）...")
        json_files = glob.glob(os.path.join(train_json_dir, "*.json"))
        for json_file in sorted(json_files):
            examples, types = load_ccks_json_format(json_file)
            all_train_examples.extend(examples)
            all_entity_types.update(types)
        logger.info(f"从ccks_ner加载了 {len(json_files)} 个训练文件")
    
    validate_file = os.path.join(ccks_data_dir, "validate_data.json")
    if os.path.exists(validate_file):
        logger.info("从ccks_ner加载验证数据（validate_data.json）...")
        examples, types = load_ccks_json_format(validate_file)
        if types or any(any(label != 'O' for label in ex['labels']) for ex in examples):
            all_dev_examples.extend(examples)
            all_entity_types.update(types)
        else:
            logger.warning("validate_data.json没有实体标注，跳过")
    
    fold0_dir = os.path.join(ccks_dir, "data", "fold0", "train")
    fold0_sentences = os.path.join(fold0_dir, "sentences.txt")
    fold0_tags = os.path.join(fold0_dir, "tags.txt")
    if os.path.exists(fold0_sentences) and os.path.exists(fold0_tags):
        logger.info("从ccks_ner加载fold0训练数据（BIO格式）...")
        examples, types = load_ccks_bio_format(fold0_sentences, fold0_tags)
        all_train_examples.extend(examples)
        all_entity_types.update(types)
    
    # 3. 从nlp_datasets目录加载MSRA数据（可选）
    USE_MSRA_DATA = False
    
    if USE_MSRA_DATA:
        msra_dir = os.path.join(base_dir, "nlp_datasets", "ner", "msra")
        msra_train_file = os.path.join(msra_dir, "msra_train_bio.txt")
        msra_test_file = os.path.join(msra_dir, "msra_test_bio.txt")
        
        if os.path.exists(msra_train_file):
            logger.info("=" * 50)
            logger.warning("从nlp_datasets/ner/msra加载训练数据（通用NER，实体类型不匹配）...")
            examples, types = load_bio_format(msra_train_file)
            all_train_examples.extend(examples)
            all_entity_types.update(types)
        
        if os.path.exists(msra_test_file):
            logger.warning("从nlp_datasets/ner/msra加载测试数据（通用NER，实体类型不匹配）...")
            examples, types = load_bio_format(msra_test_file)
            all_dev_examples.extend(examples)
            all_entity_types.update(types)
    else:
        logger.info("=" * 50)
        logger.info("MSRA数据已禁用（通用NER，实体类型与军事领域不匹配）")
    
    logger.info("=" * 50)
    logger.info(f"数据加载完成:")
    logger.info(f"  训练集: {len(all_train_examples)} 条")
    logger.info(f"  验证集: {len(all_dev_examples)} 条")
    logger.info(f"  实体类型: {len(all_entity_types)} 种")
    if all_entity_types:
        logger.info(f"  类型列表: {', '.join(sorted(all_entity_types))}")
    
    return all_train_examples, all_dev_examples, all_entity_types

