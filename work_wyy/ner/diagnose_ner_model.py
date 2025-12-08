#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NER模型诊断脚本
用于分析微调后的NER模型效果不佳的原因
"""

import torch
from transformers import BertTokenizer, BertForTokenClassification
import os
import json
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 模型路径（使用相对路径）
NER_MODEL_PATH = './../model/ner_finetuned'
BASE_MODEL_PATH = './../model/chinese-roberta-wwm-ext-large'

MAX_LENGTH = 512

# 默认标签映射（如果无法加载label_mapping.json则使用）
LABEL_TO_ID = {'O': 0, 'B-ENTITY': 1, 'I-ENTITY': 2}
ID_TO_LABEL = {0: 'O', 1: 'B-ENTITY', 2: 'I-ENTITY'}


def load_label_mapping(model_path):
    """从label_mapping.json加载标签映射"""
    global LABEL_TO_ID, ID_TO_LABEL
    
    label_mapping_file = os.path.join(model_path, 'label_mapping.json')
    if os.path.exists(label_mapping_file):
        try:
            with open(label_mapping_file, 'r', encoding='utf-8') as f:
                label_info = json.load(f)
                LABEL_TO_ID = label_info.get('label_to_id', LABEL_TO_ID)
                # 确保ID_TO_LABEL的键是整数（JSON中的键可能是字符串）
                id_to_label_raw = label_info.get('id_to_label', {})
                ID_TO_LABEL = {int(k): v for k, v in id_to_label_raw.items()}
                logger.info(f"成功加载标签映射: {len(LABEL_TO_ID)} 个标签")
                logger.debug(f"标签映射示例: {dict(list(ID_TO_LABEL.items())[:5])}")
                return True
        except Exception as e:
            logger.warning(f"加载标签映射失败: {e}，使用默认映射")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    else:
        logger.warning(f"标签映射文件不存在: {label_mapping_file}，使用默认映射")
        return False


def load_model():
    """加载模型"""
    global LABEL_TO_ID, ID_TO_LABEL
    
    if not os.path.exists(NER_MODEL_PATH):
        logger.error(f"微调模型不存在: {NER_MODEL_PATH}")
        return None, None, None
    
    logger.info(f"加载模型: {NER_MODEL_PATH}")
    
    # 先加载标签映射
    load_label_mapping(NER_MODEL_PATH)
    
    tokenizer = BertTokenizer.from_pretrained(NER_MODEL_PATH)
    model = BertForTokenClassification.from_pretrained(NER_MODEL_PATH)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    return model, tokenizer, device


def analyze_training_data():
    """分析训练数据"""
    train_file = "./../data/nerdata/train.txt"
    
    if not os.path.exists(train_file):
        logger.warning(f"训练数据文件不存在: {train_file}")
        return
    
    logger.info("=" * 70)
    logger.info("分析训练数据")
    logger.info("=" * 70)
    
    total_samples = 0
    total_entities = 0
    entity_lengths = []
    entity_types = Counter()
    
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line.strip())
                content = data.get('content', '')
                result_list = data.get('result_list', [])
                prompt = data.get('prompt', '')
                
                if content and result_list:
                    total_samples += 1
                    total_entities += len(result_list)
                    entity_types[prompt] += len(result_list)
                    
                    for entity in result_list:
                        text = entity.get('text', '')
                        if text:
                            entity_lengths.append(len(text))
            except:
                continue
    
    logger.info(f"训练样本数: {total_samples}")
    logger.info(f"总实体数: {total_entities}")
    logger.info(f"平均每个样本的实体数: {total_entities / total_samples if total_samples > 0 else 0:.2f}")
    logger.info(f"实体平均长度: {sum(entity_lengths) / len(entity_lengths) if entity_lengths else 0:.2f} 字符")
    logger.info(f"\n实体类型分布:")
    for entity_type, count in entity_types.most_common(10):
        logger.info(f"  {entity_type}: {count} 个")


def test_prediction_detailed(text, model, tokenizer, device):
    """详细测试模型预测"""
    logger.info("\n" + "=" * 70)
    logger.info(f"测试文本: {text}")
    logger.info("=" * 70)
    
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        probabilities = torch.softmax(logits, dim=-1)
    
    input_ids_list = input_ids[0].cpu().tolist()
    predictions_list = predictions[0].cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids_list)
    
    logger.info("\nToken级别的预测结果（前50个token）:")
    logger.info("-" * 70)
    logger.info(f"{'Token':<30} {'Label':<15} {'Prob(O)':<10} {'Prob(B)':<10} {'Prob(I)':<10}")
    logger.info("-" * 70)
    
    for i, (token, pred_id) in enumerate(zip(tokens[:50], predictions_list[:50])):
        if token in ['[PAD]']:
            break
        label = ID_TO_LABEL.get(pred_id, 'O')
        probs = probabilities[0][i].cpu().tolist()
        # 计算所有B-和I-标签的概率
        prob_o = probs[0] if 0 < len(probs) else 0.0
        prob_b = sum([probs[j] for j in range(1, len(probs)) if j % 2 == 1]) if len(probs) > 1 else 0.0
        prob_i = sum([probs[j] for j in range(2, len(probs)) if j % 2 == 0]) if len(probs) > 2 else 0.0
        
        logger.info(f"{token:<30} {label:<15} {prob_o:<10.4f} {prob_b:<10.4f} {prob_i:<10.4f}")
    
    # 提取实体（支持多种实体类型）
    entities = []
    current_entity_tokens = []
    current_entity_type = None
    
    for i, (token, pred_id) in enumerate(zip(tokens, predictions_list)):
        # 确保pred_id是整数
        pred_id = int(pred_id)
        label = ID_TO_LABEL.get(pred_id, 'O')
        
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            if current_entity_tokens:
                entity_text = ''.join(current_entity_tokens).replace('##', '')
                if entity_text and len(entity_text) >= 2:
                    entities.append(entity_text)
                current_entity_tokens = []
                current_entity_type = None
            continue
        
        clean_token = token.replace('##', '')
        
        # 检查是否是B-标签（任何实体类型的开始）
        if label.startswith('B-'):
            # 保存之前的实体
            if current_entity_tokens:
                entity_text = ''.join(current_entity_tokens).replace('##', '')
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
                    entity_text = ''.join(current_entity_tokens).replace('##', '')
                    if entity_text and len(entity_text) >= 2:
                        entities.append(entity_text)
                    current_entity_tokens = []
                    current_entity_type = None
        else:
            # O标签，结束当前实体
            if current_entity_tokens:
                entity_text = ''.join(current_entity_tokens).replace('##', '')
                if entity_text and len(entity_text) >= 2:
                    entities.append(entity_text)
                current_entity_tokens = []
                current_entity_type = None
    
    if current_entity_tokens:
        entity_text = ''.join(current_entity_tokens).replace('##', '')
        if entity_text and len(entity_text) >= 2:
            entities.append(entity_text)
    
    logger.info(f"\n提取的实体: {entities}")
    
    return entities


def check_label_alignment():
    """检查标签对齐的准确性"""
    logger.info("\n" + "=" * 70)
    logger.info("检查标签对齐准确性")
    logger.info("=" * 70)
    
    # 读取一条训练数据
    train_file = "./../data/nerdata/train.txt"
    if not os.path.exists(train_file):
        logger.warning("训练数据文件不存在，跳过标签对齐检查")
        return
    
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line.strip())
                content = data.get('content', '')
                result_list = data.get('result_list', [])
                
                if content and result_list:
                    # 创建字符级标签
                    char_labels = ['O'] * len(content)
                    for entity in result_list:
                        start = entity.get('start', -1)
                        end = entity.get('end', -1)
                        if start >= 0 and end > start and end <= len(content):
                            char_labels[start] = 'B-ENTITY'
                            for i in range(start + 1, end):
                                if i < len(char_labels):
                                    char_labels[i] = 'I-ENTITY'
                    
                    # 加载tokenizer
                    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL_PATH)
                    encoding = tokenizer(
                        content,
                        truncation=True,
                        padding='max_length',
                        max_length=MAX_LENGTH,
                        return_tensors='pt'
                    )
                    
                    input_ids = encoding['input_ids'][0].tolist()
                    tokens = tokenizer.convert_ids_to_tokens(input_ids)
                    
                    # 手动对齐（简化版）
                    logger.info(f"\n示例文本: {content[:100]}...")
                    logger.info(f"实体: {[e.get('text', '') for e in result_list]}")
                    logger.info(f"\n前20个token及其对应的字符标签:")
                    logger.info("-" * 70)
                    
                    char_pos = 0
                    for i, token in enumerate(tokens[:20]):
                        if token in ['[CLS]', '[SEP]', '[PAD]']:
                            logger.info(f"{token:<30} -> O (特殊token)")
                            continue
                        
                        clean_token = token.replace('##', '')
                        # 简单查找
                        found = False
                        for pos in range(max(0, char_pos - 5), min(len(content), char_pos + 20)):
                            if pos + len(clean_token) <= len(content):
                                if content[pos:pos + len(clean_token)] == clean_token:
                                    if pos < len(char_labels):
                                        label = char_labels[pos]
                                        logger.info(f"{token:<30} -> {label} (位置 {pos})")
                                        char_pos = pos + len(clean_token)
                                        found = True
                                        break
                        
                        if not found:
                            if char_pos < len(char_labels):
                                label = char_labels[char_pos]
                                logger.info(f"{token:<30} -> {label} (位置 {char_pos}, 未精确匹配)")
                                char_pos += 1
                            else:
                                logger.info(f"{token:<30} -> O (超出范围)")
                    
                    break  # 只检查第一条
            except Exception as e:
                logger.error(f"处理数据时出错: {e}")
                continue


def main():
    """主函数"""
    logger.info("=" * 70)
    logger.info("NER模型诊断")
    logger.info("=" * 70)
    
    # 1. 分析训练数据
    analyze_training_data()
    
    # 2. 检查标签对齐
    check_label_alignment()
    
    # 3. 测试模型预测
    model, tokenizer, device = load_model()
    if model is None:
        logger.error("无法加载模型，跳过预测测试")
        return
    
    test_texts = [
        "阿利·伯克级驱逐舰装备有宙斯盾作战系统和AN/SPY-13D相控阵雷达，主要用于防空作战。",
        "该舰装备了战斧巡航导弹和标准系列防空导弹。",
        "美国海军计划建造朱姆沃尔特级驱逐舰作为下一代主力驱逐舰。"
    ]
    
    for text in test_texts:
        test_prediction_detailed(text, model, tokenizer, device)
    
    logger.info("\n" + "=" * 70)
    logger.info("诊断完成")
    logger.info("=" * 70)
    
    # 提供改进建议
    logger.info("\n改进建议:")
    logger.info("1. 训练数据量可能不足（294条），建议增加到1000+条")
    logger.info("2. 训练时间可能太短（33秒），建议增加训练轮数或检查训练是否正常完成")
    logger.info("3. 标签对齐方法可能需要优化，确保字符级标签正确映射到token级标签")
    logger.info("4. 建议添加验证集，以便在训练过程中监控模型性能")
    logger.info("5. 可以尝试调整学习率、批次大小等超参数")


if __name__ == "__main__":
    main()

