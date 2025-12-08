#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NER实体提取脚本
使用微调后的NER模型从文本中提取实体词（如"宙斯盾"、"AN/SPY-13D相控阵雷达"等）
"""

import torch
from transformers import BertTokenizer, BertForTokenClassification
import os
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 模型路径（使用相对路径）
NER_MODEL_PATH = './../model/ner_finetuned'
BASE_MODEL_PATH = './../model/chinese-roberta-wwm-ext-large'

# 验证基础模型路径
if not os.path.exists(BASE_MODEL_PATH):
    logger.warning(f"⚠️  基础模型路径不存在: {BASE_MODEL_PATH}")
    logger.warning(f"   请确保模型位于: {BASE_MODEL_PATH}")
else:
    logger.info(f"✅ 基础模型路径正确: {BASE_MODEL_PATH}")

MAX_LENGTH = 512

# 默认标签映射（如果无法加载label_mapping.json则使用）
LABEL_TO_ID = {'O': 0, 'B-ENTITY': 1, 'I-ENTITY': 2}
ID_TO_LABEL = {0: 'O', 1: 'B-ENTITY', 2: 'I-ENTITY'}

# 全局变量
ner_model = None
ner_tokenizer = None
ner_device = None


def load_label_mapping(model_path):
    """从label_mapping.json加载标签映射"""
    global LABEL_TO_ID, ID_TO_LABEL
    
    label_mapping_file = os.path.join(model_path, 'label_mapping.json')
    if os.path.exists(label_mapping_file):
        try:
            import json
            with open(label_mapping_file, 'r', encoding='utf-8') as f:
                label_info = json.load(f)
                LABEL_TO_ID = label_info.get('label_to_id', LABEL_TO_ID)
                # 确保ID_TO_LABEL是整数键
                id_to_label_raw = label_info.get('id_to_label', {})
                ID_TO_LABEL = {int(k): v for k, v in id_to_label_raw.items()}
                logger.info(f"✅ 成功加载标签映射: {len(LABEL_TO_ID)} 个标签")
                return True
        except Exception as e:
            logger.warning(f"加载标签映射失败: {e}，使用默认映射")
            return False
    else:
        logger.warning(f"标签映射文件不存在: {label_mapping_file}，使用默认映射")
        return False


def load_ner_model(model_path=None):
    """
    加载NER模型
    
    Args:
        model_path: 模型路径，如果为None则使用默认路径
    """
    global ner_model, ner_tokenizer, ner_device, LABEL_TO_ID, ID_TO_LABEL
    
    if model_path is None:
        model_path = NER_MODEL_PATH
    
    # 如果微调模型不存在，使用基础模型
    if not os.path.exists(model_path):
        logger.warning(f"微调模型不存在: {model_path}，使用基础模型: {BASE_MODEL_PATH}")
        model_path = BASE_MODEL_PATH
    
    try:
        # 验证模型路径
        if not os.path.exists(model_path):
            logger.error(f"❌ 模型路径不存在: {model_path}")
            logger.error(f"   请确保模型位于: {model_path}")
            return False
        
        logger.info(f"加载NER模型: {model_path}")
        logger.info(f"   模型路径验证: ✅ 存在")
        ner_tokenizer = BertTokenizer.from_pretrained(model_path)
        
        # 如果是指定的微调模型路径，尝试加载微调后的模型和标签映射
        if model_path == NER_MODEL_PATH and os.path.exists(model_path):
            # 先加载标签映射
            load_label_mapping(model_path)
            try:
                ner_model = BertForTokenClassification.from_pretrained(model_path)
                logger.info("✅ 成功加载微调后的NER模型")
            except Exception as e:
                logger.warning(f"加载微调模型失败: {e}，使用基础模型")
                model_path = BASE_MODEL_PATH
                ner_model = BertForTokenClassification.from_pretrained(
                    model_path,
                    num_labels=len(LABEL_TO_ID),
                    id2label=ID_TO_LABEL,
                    label2id=LABEL_TO_ID
                )
        else:
            # 使用基础模型（未微调）
            ner_model = BertForTokenClassification.from_pretrained(
                model_path,
                num_labels=len(LABEL_TO_ID),
                id2label=ID_TO_LABEL,
                label2id=LABEL_TO_ID
            )
            logger.info("✅ 使用基础模型（未微调，效果可能较差）")
        
        # 设置设备
        ner_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ner_model.to(ner_device)
        ner_model.eval()
        
        logger.info(f"NER模型加载成功，设备: {ner_device}")
        return True
        
    except Exception as e:
        logger.error(f"NER模型加载失败: {e}")
        return False


def _tokens_to_text(tokens):
    """
    将token列表转换为文本
    
    Args:
        tokens: token列表（已移除##前缀）
    
    Returns:
        text: 拼接后的文本
    """
    if not tokens:
        return ""
    
    # 使用tokenizer的decode方法更准确地还原文本
    # 但需要先转换回token ids
    try:
        # 尝试直接拼接（适用于中文）
        text = "".join(tokens)
        # 如果包含英文，可能需要添加空格
        # 简单处理：如果token之间没有标点，可能需要空格
        # 但为了简单，先直接拼接
        return text
    except:
        return "".join(tokens)


def extract_entities_from_text(text, model_path=None, verbose=False):
    """
    从文本中提取实体词
    
    Args:
        text: 输入文本
        model_path: 模型路径（可选）
        verbose: 是否打印详细信息
    
    Returns:
        entities: 实体词列表（去重）
    """
    global ner_model, ner_tokenizer, ner_device
    
    # 如果模型未加载，先加载
    if ner_model is None or ner_tokenizer is None:
        if not load_ner_model(model_path):
            if verbose:
                logger.warning("NER模型未加载，返回空列表")
            return []
    
    if not text or not isinstance(text, str):
        return []
    
    try:
        # Tokenize（不使用offset_mapping，因为Python tokenizer不支持）
        encoding = ner_tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )
        
        # 移动到设备
        input_ids = encoding['input_ids'].to(ner_device)
        attention_mask = encoding['attention_mask'].to(ner_device)
        
        # 预测
        with torch.no_grad():
            outputs = ner_model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        # 获取token文本（用于对齐）
        input_ids_list = input_ids[0].cpu().tolist()
        predictions_list = predictions[0].cpu().tolist()
        tokens = ner_tokenizer.convert_ids_to_tokens(input_ids_list)
        
        # 提取实体（支持多种实体类型）
        entities = []
        current_entity_tokens = []
        current_entity_type = None
        
        for i, (token, pred_id) in enumerate(zip(tokens, predictions_list)):
            label = ID_TO_LABEL.get(pred_id, 'O')
            
            # 跳过特殊token
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                # 如果遇到特殊token，结束当前实体
                if current_entity_tokens:
                    entity_text = _tokens_to_text(current_entity_tokens)
                    if entity_text and len(entity_text) >= 2:
                        entities.append(entity_text)
                    current_entity_tokens = []
                    current_entity_type = None
                continue
            
            # 处理token（移除##前缀，这是BERT tokenizer的子词标记）
            clean_token = token.replace('##', '')
            
            # 检查是否是B-标签（任何实体类型的开始）
            if label.startswith('B-'):
                # 保存之前的实体
                if current_entity_tokens:
                    entity_text = _tokens_to_text(current_entity_tokens)
                    if entity_text and len(entity_text) >= 2:
                        entities.append(entity_text)
                # 开始新实体
                current_entity_tokens = [clean_token]
                current_entity_type = label[2:]  # 提取实体类型
            elif label.startswith('I-') and current_entity_tokens:
                # 继续当前实体（检查类型是否匹配）
                entity_type = label[2:]
                if entity_type == current_entity_type:
                    current_entity_tokens.append(clean_token)
                else:
                    # 类型不匹配，结束当前实体
                    if current_entity_tokens:
                        entity_text = _tokens_to_text(current_entity_tokens)
                        if entity_text and len(entity_text) >= 2:
                            entities.append(entity_text)
                        current_entity_tokens = []
                        current_entity_type = None
            else:
                # O标签，结束当前实体
                if current_entity_tokens:
                    entity_text = _tokens_to_text(current_entity_tokens)
                    if entity_text and len(entity_text) >= 2:
                        entities.append(entity_text)
                    current_entity_tokens = []
                    current_entity_type = None
        
        # 处理最后一个实体
        if current_entity_tokens:
            entity_text = _tokens_to_text(current_entity_tokens)
            if entity_text and len(entity_text) >= 2:
                entities.append(entity_text)
        
        # 去重并保持顺序
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)
        
        if verbose:
            logger.info(f"从文本中提取到 {len(unique_entities)} 个实体: {unique_entities[:10]}")
        
        return unique_entities
        
    except Exception as e:
        logger.error(f"实体提取失败: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return []


def get_entity_words_from_text(text, lang='zh', model_path=None, verbose=False):
    """
    从文本中提取实体词（替代原来的词频统计方法）
    
    Args:
        text: 文本内容
        lang: 语言类型（'zh' 或 'en'），目前主要支持中文
        model_path: 模型路径（可选）
        verbose: 是否打印详细信息
    
    Returns:
        entity_words: 实体词列表
        entity_freq: 实体词频字典（每个实体出现1次）
    """
    if not text or not isinstance(text, str):
        return [], {}
    
    # 使用NER提取实体
    entities = extract_entities_from_text(text, model_path=model_path, verbose=verbose)
    
    # 转换为词频格式（每个实体出现1次）
    entity_freq = {entity: 1 for entity in entities}
    
    return entities, entity_freq


if __name__ == "__main__":
    # 测试
    test_text = "阿利·伯克级驱逐舰装备有宙斯盾作战系统和AN/SPY-13D相控阵雷达，主要用于防空作战。"
    
    print("=" * 70)
    print("NER实体提取测试")
    print("=" * 70)
    print(f"测试文本: {test_text}\n")
    
    entities = extract_entities_from_text(test_text, verbose=True)
    print(f"\n提取的实体: {entities}")

