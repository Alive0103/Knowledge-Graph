#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
验证标签对齐准确性的脚本
用于检查字符级标签到token级标签的映射是否正确
"""

import json
import os
from transformers import BertTokenizer

# 模型路径（使用相对路径）
BASE_MODEL_PATH = './../model/chinese-roberta-wwm-ext-large'

LABEL_TO_ID = {'O': 0, 'B-ENTITY': 1, 'I-ENTITY': 2}
ID_TO_LABEL = {0: 'O', 1: 'B-ENTITY', 2: 'I-ENTITY'}


def align_labels_improved(char_labels, text, tokens, input_ids):
    """改进的标签对齐方法（与训练时使用的方法一致）"""
    token_labels = []
    char_idx = 0
    
    for i, token in enumerate(tokens):
        if token in ['[CLS]', '[SEP]', '[PAD]', '<pad>', '<s>', '</s>', '[UNK]']:
            token_labels.append(LABEL_TO_ID['O'])
            continue
        
        is_subword = token.startswith('##')
        clean_token = token.replace('##', '').replace('▁', '').strip()
        
        if not clean_token:
            token_labels.append(LABEL_TO_ID['O'])
            continue
        
        found = False
        best_match_pos = -1
        
        search_start = max(0, char_idx - 20)
        search_end = min(len(text), char_idx + 50)
        
        for pos in range(search_start, search_end):
            if pos + len(clean_token) <= len(text):
                if text[pos:pos + len(clean_token)] == clean_token:
                    best_match_pos = pos
                    found = True
                    if abs(pos - char_idx) < 10:
                        break
        
        if found and best_match_pos >= 0:
            if best_match_pos < len(char_labels):
                char_label = char_labels[best_match_pos]
                token_labels.append(LABEL_TO_ID.get(char_label, LABEL_TO_ID['O']))
                if not is_subword:
                    char_idx = best_match_pos + len(clean_token)
            else:
                token_labels.append(LABEL_TO_ID['O'])
        else:
            if char_idx < len(char_labels):
                char_label = char_labels[char_idx]
                token_labels.append(LABEL_TO_ID.get(char_label, LABEL_TO_ID['O']))
                if not is_subword:
                    char_idx = min(char_idx + 1, len(text))
            else:
                token_labels.append(LABEL_TO_ID['O'])
    
    while len(token_labels) < len(input_ids):
        token_labels.append(LABEL_TO_ID['O'])
    if len(token_labels) > len(input_ids):
        token_labels = token_labels[:len(input_ids)]
    
    return token_labels


def verify_alignment():
    """验证标签对齐"""
    train_file = "./../data/nerdata/train.txt"
    
    if not os.path.exists(train_file):
        print(f"训练数据文件不存在: {train_file}")
        return
    
    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL_PATH)
    
    print("=" * 70)
    print("验证标签对齐准确性")
    print("=" * 70)
    
    correct_count = 0
    total_count = 0
    error_examples = []
    
    with open(train_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line.strip())
                content = data.get('content', '')
                result_list = data.get('result_list', [])
                
                if not content or not result_list:
                    continue
                
                # 创建字符级标签
                char_labels = ['O'] * len(content)
                entity_positions = []
                for entity in result_list:
                    start = entity.get('start', -1)
                    end = entity.get('end', -1)
                    text = entity.get('text', '')
                    
                    if start >= 0 and end > start and end <= len(content):
                        entity_positions.append((start, end, text))
                        char_labels[start] = 'B-ENTITY'
                        for i in range(start + 1, end):
                            if i < len(char_labels):
                                char_labels[i] = 'I-ENTITY'
                
                # Tokenize
                encoding = tokenizer(
                    content,
                    truncation=True,
                    padding='max_length',
                    max_length=512,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'][0].tolist()
                tokens = tokenizer.convert_ids_to_tokens(input_ids)
                
                # 对齐标签
                token_labels = align_labels_improved(char_labels, content, tokens, input_ids)
                
                # 验证：检查实体位置的token是否被正确标记
                errors = []
                for start, end, entity_text in entity_positions:
                    # 找到实体对应的token范围
                    entity_tokens = []
                    char_pos = 0
                    entity_start_token_idx = -1
                    entity_end_token_idx = -1
                    
                    for i, token in enumerate(tokens):
                        if token in ['[CLS]', '[SEP]', '[PAD]']:
                            continue
                        
                        clean_token = token.replace('##', '').replace('▁', '').strip()
                        if not clean_token:
                            continue
                        
                        # 检查token是否在实体范围内
                        found_pos = -1
                        for pos in range(max(0, char_pos - 20), min(len(content), char_pos + 50)):
                            if pos + len(clean_token) <= len(content):
                                if content[pos:pos + len(clean_token)] == clean_token:
                                    found_pos = pos
                                    break
                        
                        if found_pos >= 0:
                            if start <= found_pos < end:
                                if entity_start_token_idx == -1:
                                    entity_start_token_idx = i
                                entity_end_token_idx = i
                            char_pos = found_pos + len(clean_token)
                    
                    # 检查实体token是否被正确标记
                    if entity_start_token_idx >= 0:
                        # 第一个token应该是B-ENTITY
                        if entity_start_token_idx < len(token_labels):
                            if token_labels[entity_start_token_idx] != LABEL_TO_ID['B-ENTITY']:
                                errors.append(f"实体'{entity_text}'的起始token应该是B-ENTITY，但得到{ID_TO_LABEL.get(token_labels[entity_start_token_idx], 'UNK')}")
                        
                        # 后续token应该是I-ENTITY
                        for i in range(entity_start_token_idx + 1, min(entity_end_token_idx + 1, len(token_labels))):
                            if token_labels[i] != LABEL_TO_ID['I-ENTITY'] and token_labels[i] != LABEL_TO_ID['B-ENTITY']:
                                errors.append(f"实体'{entity_text}'的token {i}应该是I-ENTITY，但得到{ID_TO_LABEL.get(token_labels[i], 'UNK')}")
                
                if errors:
                    error_examples.append({
                        'line': line_num,
                        'content': content[:100],
                        'entities': [e[2] for e in entity_positions],
                        'errors': errors
                    })
                else:
                    correct_count += 1
                
                total_count += 1
                
                # 只检查前50条
                if total_count >= 50:
                    break
                    
            except Exception as e:
                print(f"处理第{line_num}行时出错: {e}")
                continue
    
    print(f"\n验证结果:")
    print(f"  总样本数: {total_count}")
    print(f"  正确对齐: {correct_count}")
    print(f"  错误对齐: {len(error_examples)}")
    print(f"  准确率: {correct_count / total_count * 100:.2f}%")
    
    if error_examples:
        print(f"\n前5个错误示例:")
        for i, example in enumerate(error_examples[:5], 1):
            print(f"\n示例 {i} (第{example['line']}行):")
            print(f"  文本: {example['content']}...")
            print(f"  实体: {example['entities']}")
            print(f"  错误: {example['errors'][0] if example['errors'] else 'N/A'}")


if __name__ == "__main__":
    verify_alignment()

