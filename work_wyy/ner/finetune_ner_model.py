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
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, 
    BertForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from tqdm import tqdm
import logging
from collections import Counter

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

# 实体类型标签映射（BIO格式）
LABEL_TO_ID = {
    'O': 0,  # Outside
    'B-ENTITY': 1,  # Begin
    'I-ENTITY': 2,  # Inside
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
NUM_LABELS = len(LABEL_TO_ID)


def load_ner_data(file_path):
    """
    加载NER训练数据
    
    Args:
        file_path: 训练数据文件路径
    
    Returns:
        examples: 数据列表，每个元素包含text和labels
    """
    examples = []
    
    if not os.path.exists(file_path):
        logger.warning(f"文件不存在: {file_path}")
        return examples
    
    logger.info(f"加载数据文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                if not line.strip():
                    continue
                
                data = json.loads(line.strip())
                content = data.get('content', '')
                result_list = data.get('result_list', [])
                
                if not content or not result_list:
                    continue
                
                # 创建标签序列（初始化为O）
                labels = ['O'] * len(content)
                
                # 标记实体
                for entity in result_list:
                    start = entity.get('start', -1)
                    end = entity.get('end', -1)
                    text = entity.get('text', '')
                    
                    if start >= 0 and end > start and end <= len(content):
                        # 标记B-ENTITY
                        labels[start] = 'B-ENTITY'
                        # 标记I-ENTITY（如果有多个字符）
                        for i in range(start + 1, end):
                            if i < len(labels):
                                labels[i] = 'I-ENTITY'
                
                examples.append({
                    'text': content,
                    'labels': labels
                })
                
            except Exception as e:
                logger.warning(f"第{line_num}行解析失败: {e}")
                continue
    
    logger.info(f"成功加载 {len(examples)} 条数据")
    return examples


class NERDataset(Dataset):
    """NER数据集类"""
    
    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        text = example['text']
        labels = example['labels']
        
        # Tokenize（不使用offset_mapping，因为Python tokenizer不支持）
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 将字符级标签转换为token级标签（不使用offset_mapping）
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
        将字符级标签对齐到token级标签（改进版，更精确）
        
        使用更精确的方法：
        1. 通过tokenizer的decode重建文本片段
        2. 使用字符级别的精确匹配
        3. 处理中文和英文的差异
        
        Args:
            char_labels: 字符级标签列表
            text: 原始文本
            tokens: token列表
            input_ids: token ids
        
        Returns:
            token_labels: token级标签列表
        """
        token_labels = []
        text_chars = list(text)  # 转换为字符列表，便于处理中文
        char_idx = 0  # 当前字符索引
        
        for i, token in enumerate(tokens):
            # 跳过特殊token
            if token in ['[CLS]', '[SEP]', '[PAD]', '<pad>', '<s>', '</s>', '[UNK]']:
                token_labels.append(LABEL_TO_ID['O'])
                continue
            
            # 处理BERT子词标记
            is_subword = token.startswith('##')
            clean_token = token.replace('##', '').replace('▁', '').strip()
            
            if not clean_token:
                token_labels.append(LABEL_TO_ID['O'])
                continue
            
            # 尝试找到token在文本中的位置
            found = False
            best_match_pos = -1
            
            # 搜索范围：从当前位置向前向后搜索
            search_start = max(0, char_idx - 20)
            search_end = min(len(text), char_idx + 50)
            
            # 方法1：精确字符串匹配
            for pos in range(search_start, search_end):
                if pos + len(clean_token) <= len(text):
                    if text[pos:pos + len(clean_token)] == clean_token:
                        best_match_pos = pos
                        found = True
                        # 如果找到的匹配位置在当前位置附近，优先使用
                        if abs(pos - char_idx) < 10:
                            break
            
            # 方法2：如果精确匹配失败，尝试字符级匹配（处理中文）
            if not found and len(clean_token) > 0:
                # 对于中文，尝试逐字符匹配
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
                # 使用找到位置的标签
                if best_match_pos < len(char_labels):
                    char_label = char_labels[best_match_pos]
                    token_labels.append(LABEL_TO_ID.get(char_label, LABEL_TO_ID['O']))
                    # 更新字符索引：完整词移动到匹配位置之后，子词不移动
                    if not is_subword:
                        char_idx = best_match_pos + len(clean_token)
                else:
                    token_labels.append(LABEL_TO_ID['O'])
            else:
                # 如果找不到匹配，使用当前位置的标签
                if char_idx < len(char_labels):
                    char_label = char_labels[char_idx]
                    token_labels.append(LABEL_TO_ID.get(char_label, LABEL_TO_ID['O']))
                    # 保守地向前移动
                    if not is_subword:
                        char_idx = min(char_idx + 1, len(text))
                else:
                    token_labels.append(LABEL_TO_ID['O'])
        
        # 确保长度匹配
        while len(token_labels) < len(input_ids):
            token_labels.append(LABEL_TO_ID['O'])
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
    
    # 加载数据（使用相对路径）
    train_file = "./../data/nerdata/train.txt"
    dev_file = "./../data/nerdata/dev.txt"
    test_file = "./../data/nerdata/test.txt"
    
    train_examples = load_ner_data(train_file)
    dev_examples = load_ner_data(dev_file)
    test_examples = load_ner_data(test_file)
    
    if not train_examples:
        logger.error("未加载到训练数据")
        return
    
    # 如果没有验证集，从训练集中划分（80/20）
    if not dev_examples and len(train_examples) > 10:
        split_idx = int(0.8 * len(train_examples))
        dev_examples = train_examples[split_idx:]
        train_examples = train_examples[:split_idx]
        logger.info(f"从训练集中划分验证集: 训练集 {len(train_examples)} 条, 验证集 {len(dev_examples)} 条")
    
    logger.info(f"训练集: {len(train_examples)} 条")
    logger.info(f"验证集: {len(dev_examples)} 条")
    logger.info(f"测试集: {len(test_examples)} 条")
    
    # 创建数据集
    train_dataset = NERDataset(train_examples, tokenizer, MAX_LENGTH)
    dev_dataset = NERDataset(dev_examples, tokenizer, MAX_LENGTH) if dev_examples else None
    test_dataset = NERDataset(test_examples, tokenizer, MAX_LENGTH) if test_examples else None
    
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
    
    # 评估测试集
    if test_examples:
        logger.info("评估测试集...")
        test_results = trainer.evaluate(test_dataset)
        logger.info(f"测试集结果: {test_results}")
    
    logger.info("=" * 70)
    logger.info("微调完成!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

