#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NER模型微调脚本
使用Chinese-RoBERTa作为基础模型，在军事领域NER数据上微调
支持多种实体类型的识别（军工企业、军事组织、火炮、导弹、军用航空器等）
"""

import json
import os
import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer, 
    BertForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import logging

# 导入数据加载模块
from data_loader import load_all_data_from_directories

# 配置日志
log_file = os.path.join(os.path.dirname(__file__), 'ner_finetune.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 模型路径（相对于当前文件，使用相对路径）
MODEL_NAME = './../model/chinese-roberta-wwm-ext-large'
OUTPUT_DIR = './../model/ner_finetuned'
MAX_LENGTH = 512
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5  # 增加到5轮，确保充分训练

# 预定义的实体类型（根据用户需求）
ENTITY_TYPES = [
    "火炮", "军工企业", "军用舰艇", "军事组织", "枪械",
    "军用航空器", "军用车辆", "武器系统", "导弹", "信息系统",
    "地缘政治实体", "军事系统", "无人机", "弹药", "军事地点",
    "装甲车辆"  # 数据中已有的类型
]

# 动态构建标签映射（BIO格式）
def build_label_mapping(entity_types):
    """构建BIO格式的标签映射"""
    label_to_id = {'O': 0}
    label_id = 1
    
    # 为每个实体类型创建B和I标签
    for entity_type in entity_types:
        label_to_id[f'B-{entity_type}'] = label_id
        label_id += 1
        label_to_id[f'I-{entity_type}'] = label_id
        label_id += 1
    
    return label_to_id

# 初始化标签映射（将在加载数据后更新）
LABEL_TO_ID = build_label_mapping(ENTITY_TYPES)
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
NUM_LABELS = len(LABEL_TO_ID)


def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    # 展平
    true_labels = labels.flatten()
    pred_labels = predictions.flatten()
    
    # 移除padding标签（-100）
    mask = true_labels != -100
    true_labels = true_labels[mask]
    pred_labels = pred_labels[mask]
    
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='weighted', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


class NERDataset(Dataset):
    """NER数据集类"""
    
    def __init__(self, examples, tokenizer, max_length=512, label_to_id=None):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_id = label_to_id or LABEL_TO_ID
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        text = example['text']
        labels = example['labels']
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 将字符级标签转换为token级标签
        input_ids = encoding['input_ids'][0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        token_labels = self.align_labels_with_tokens(
            labels,
            text,
            tokens,
            input_ids
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(token_labels, dtype=torch.long)
        }
    
    def align_labels_with_tokens(self, char_labels, text, tokens, input_ids):
        """
        将字符级标签对齐到token级标签
        
        Args:
            char_labels: 字符级标签列表
            text: 原始文本
            tokens: token列表
            input_ids: token ids
        
        Returns:
            token_labels: token级标签列表
        """
        token_labels = []
        char_idx = 0
        
        for i, token in enumerate(tokens):
            # 跳过特殊token
            if token in ['[CLS]', '[SEP]', '[PAD]', '<pad>', '<s>', '</s>', '[UNK]']:
                token_labels.append(self.label_to_id.get('O', 0))
                continue
            
            # 处理BERT子词标记
            is_subword = token.startswith('##')
            clean_token = token.replace('##', '').replace('▁', '').strip()
            
            if not clean_token:
                token_labels.append(self.label_to_id.get('O', 0))
                continue
            
            # 尝试找到token在文本中的位置
            found = False
            best_match_pos = -1
            search_start = max(0, char_idx - 20)
            search_end = min(len(text), char_idx + 50)
            
            # 精确字符串匹配
            for pos in range(search_start, search_end):
                if pos + len(clean_token) <= len(text):
                    if text[pos:pos + len(clean_token)] == clean_token:
                        best_match_pos = pos
                        found = True
                        if abs(pos - char_idx) < 10:
                            break
            
            # 字符级匹配（处理中文）
            if not found and len(clean_token) > 0:
                for pos in range(search_start, min(search_end, len(text) - len(clean_token) + 1)):
                    match = True
                    for j, char in enumerate(clean_token):
                        if pos + j >= len(text) or text[pos + j] != char:
                            match = False
                            break
                    if match:
                        best_match_pos = pos
                        found = True
                        if abs(pos - char_idx) < 10:
                            break
            
            if found and best_match_pos >= 0:
                if best_match_pos < len(char_labels):
                    char_label = char_labels[best_match_pos]
                    token_labels.append(self.label_to_id.get(char_label, self.label_to_id.get('O', 0)))
                    if not is_subword:
                        char_idx = best_match_pos + len(clean_token)
                else:
                    token_labels.append(self.label_to_id.get('O', 0))
            else:
                if char_idx < len(char_labels):
                    char_label = char_labels[char_idx]
                    token_labels.append(self.label_to_id.get(char_label, self.label_to_id.get('O', 0)))
                    if not is_subword:
                        char_idx = min(char_idx + 1, len(text))
                else:
                    token_labels.append(self.label_to_id.get('O', 0))
        
        # 确保长度匹配
        while len(token_labels) < len(input_ids):
            token_labels.append(self.label_to_id.get('O', 0))
        if len(token_labels) > len(input_ids):
            token_labels = token_labels[:len(input_ids)]
        
        return token_labels


def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    # 展平
    true_labels = labels.flatten()
    pred_labels = predictions.flatten()
    
    # 移除padding标签（-100）
    mask = true_labels != -100
    true_labels = true_labels[mask]
    pred_labels = pred_labels[mask]
    
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='weighted', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    """主函数"""
    logger.info("=" * 70)
    logger.info("NER模型微调")
    logger.info("=" * 70)
    
    # 数据目录（从多个目录加载数据）
    base_data_dir = "./../data"
    
    # 从所有数据目录加载数据（使用data_loader模块）
    logger.info("从多个数据目录加载数据...")
    train_examples, dev_examples, all_entity_types = load_all_data_from_directories(base_data_dir)
    
    if not train_examples:
        logger.error("未加载到训练数据")
        return
    
    # 动态更新标签映射
    if all_entity_types:
        logger.info("更新标签映射以包含所有实体类型...")
        # 确保包含预定义的实体类型
        all_entity_types.update(ENTITY_TYPES)
        global LABEL_TO_ID, ID_TO_LABEL, NUM_LABELS
        LABEL_TO_ID = build_label_mapping(sorted(all_entity_types))
        ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
        NUM_LABELS = len(LABEL_TO_ID)
        logger.info(f"标签映射已更新: {NUM_LABELS} 个标签")
        logger.info(f"实体类型: {', '.join(sorted(all_entity_types))}")
    
    # 如果没有验证集，从训练集中划分（80/20）
    if not dev_examples and len(train_examples) > 10:
        split_idx = int(0.8 * len(train_examples))
        dev_examples = train_examples[split_idx:]
        train_examples = train_examples[:split_idx]
        logger.info(f"从训练集中划分验证集: 训练集 {len(train_examples)} 条, 验证集 {len(dev_examples)} 条")
    
    logger.info(f"训练集: {len(train_examples)} 条")
    logger.info(f"验证集: {len(dev_examples)} 条")
    
    # 检查模型路径
    if not os.path.exists(MODEL_NAME):
        logger.error(f"模型路径不存在: {MODEL_NAME}")
        logger.error("请确保Chinese-RoBERTa模型已下载到指定路径")
        return
    
    # 加载tokenizer和模型
    logger.info(f"加载模型: {MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID
    )
    
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    model.to(device)
    
    # 创建数据集（传入更新后的标签映射）
    train_dataset = NERDataset(train_examples, tokenizer, MAX_LENGTH, LABEL_TO_ID)
    dev_dataset = NERDataset(dev_examples, tokenizer, MAX_LENGTH, LABEL_TO_ID) if dev_examples else None
    
    # 数据整理器
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        logging_dir=f'{OUTPUT_DIR}/logs',
        logging_steps=50,  # 更频繁地记录日志
        eval_strategy="epoch" if dev_examples else "no",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True if dev_examples else False,
        metric_for_best_model="f1" if dev_examples else None,
        greater_is_better=True,
        warmup_steps=min(100, len(train_examples) // BATCH_SIZE),  # 根据数据量调整warmup步数
        fp16=torch.cuda.is_available(),  # 如果支持，使用混合精度
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics if dev_examples else None,
    )
    
    # 开始训练
    logger.info("开始训练...")
    trainer.train()
    
    # 保存模型
    logger.info(f"保存模型到: {OUTPUT_DIR}")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # 保存标签映射信息
    label_info = {
        'label_to_id': LABEL_TO_ID,
        'id_to_label': ID_TO_LABEL,
        'entity_types': sorted(all_entity_types),
        'num_labels': NUM_LABELS
    }
    label_info_path = os.path.join(OUTPUT_DIR, 'label_mapping.json')
    with open(label_info_path, 'w', encoding='utf-8') as f:
        json.dump(label_info, f, ensure_ascii=False, indent=2)
    logger.info(f"标签映射信息已保存到: {label_info_path}")
    
    logger.info("=" * 70)
    logger.info("微调完成!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

