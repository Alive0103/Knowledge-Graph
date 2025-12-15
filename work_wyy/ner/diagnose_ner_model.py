#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NER模型诊断脚本
用于分析微调后的NER模型效果，提供科学的诊断和改进建议
"""

import torch
from transformers import BertTokenizer, BertForTokenClassification
import os
import json
import logging
from collections import Counter
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 模型路径（使用绝对路径，基于脚本位置）
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)  # work_wyy 目录

# 优先使用微调后的模型
NER_MODEL_PATH = os.path.join(_parent_dir, 'model', 'ner_finetuned')
BASE_MODEL_PATH = os.path.join(_parent_dir, 'model', 'chinese-roberta-wwm-ext-large')

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
                logger.info(f"✅ 成功加载标签映射: {len(LABEL_TO_ID)} 个标签")
                
                # 统计实体类型
                entity_types = set()
                for label in LABEL_TO_ID.keys():
                    if label.startswith('B-'):
                        entity_types.add(label[2:])
                
                if entity_types:
                    logger.info(f"   支持的实体类型: {len(entity_types)} 种")
                    logger.info(f"   类型列表: {', '.join(sorted(entity_types)[:10])}{'...' if len(entity_types) > 10 else ''}")
                
                return True
        except Exception as e:
            logger.warning(f"⚠️  加载标签映射失败: {e}，使用默认映射")
            return False
    else:
        logger.warning(f"⚠️  标签映射文件不存在: {label_mapping_file}，使用默认映射")
        return False


def load_model():
    """加载模型"""
    global LABEL_TO_ID, ID_TO_LABEL
    
    # 检查微调模型是否存在
    if not os.path.exists(NER_MODEL_PATH):
        logger.error(f"❌ 微调模型不存在: {NER_MODEL_PATH}")
        logger.error(f"   请确保模型位于: {NER_MODEL_PATH}")
        logger.error(f"   请先运行 finetune_ner_model.py 训练模型")
        return None, None, None
    
    logger.info(f"✅ 加载模型: {NER_MODEL_PATH}")
    
    # 先加载标签映射
    if not load_label_mapping(NER_MODEL_PATH):
        logger.warning("⚠️  标签映射加载失败，使用默认映射")
    
    try:
        tokenizer = BertTokenizer.from_pretrained(NER_MODEL_PATH)
        model = BertForTokenClassification.from_pretrained(NER_MODEL_PATH)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        logger.info(f"✅ 模型加载成功，使用设备: {device}")
        return model, tokenizer, device
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None


def analyze_training_data():
    """分析训练数据（使用data_loader统计所有数据源）"""
    logger.info("=" * 70)
    logger.info("分析训练数据")
    logger.info("=" * 70)
    
    try:
        # 导入数据加载模块
        import sys
        sys.path.insert(0, _script_dir)
        from data_loader import load_all_data_from_directories
        
        # 加载所有数据
        base_dir = os.path.join(_parent_dir, 'data')
        train_examples, dev_examples, all_entity_types = load_all_data_from_directories(base_dir)
        
        # 统计训练数据
        total_train_samples = len(train_examples)
        total_dev_samples = len(dev_examples)
        
        # 统计实体信息（简化版：只统计B-标签数量）
        total_entities = 0
        entity_types_counter = Counter()
        
        for example in train_examples:
            labels = example.get('labels', [])
            
            # 统计B-标签（实体开始）
            for label in labels:
                if label.startswith('B-'):
                    total_entities += 1
                    entity_type = label[2:]
                    if entity_type:
                        entity_types_counter[entity_type] += 1
        
        # 输出统计信息
        logger.info(f"训练样本数: {total_train_samples:,} 条")
        logger.info(f"验证样本数: {total_dev_samples:,} 条")
        logger.info(f"总实体数: {total_entities:,} 个")
        if total_train_samples > 0:
            logger.info(f"平均每个样本的实体数: {total_entities / total_train_samples:.2f}")
        logger.info(f"实体类型总数: {len(all_entity_types)} 种")
        
        if entity_types_counter:
            logger.info(f"\n实体类型分布（Top 10）:")
            for entity_type, count in entity_types_counter.most_common(10):
                logger.info(f"  {entity_type}: {count:,} 个")
        
        return {
            'train_samples': total_train_samples,
            'dev_samples': total_dev_samples,
            'total_entities': total_entities,
            'entity_types': len(all_entity_types),
            'avg_entities_per_sample': total_entities / total_train_samples if total_train_samples > 0 else 0
        }
        
    except Exception as e:
        logger.warning(f"⚠️  数据统计失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def analyze_training_log():
    """分析训练日志，提取训练信息"""
    log_file = os.path.join(_script_dir, 'ner_finetune.log')
    
    if not os.path.exists(log_file):
        logger.warning(f"⚠️  训练日志文件不存在: {log_file}")
        return None
    
    try:
        training_info = {
            'epochs': 0,
            'final_f1': None,
            'final_accuracy': None,
            'training_time': None
        }
        
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # 查找最后的评估结果和训练信息
        for line in reversed(lines):
            # 尝试解析字典格式的日志（如 {'eval_f1': 0.98, ...}）
            if '{' in line and ('eval_f1' in line or 'train_runtime' in line):
                try:
                    # 提取字典部分
                    dict_start = line.find('{')
                    dict_end = line.rfind('}') + 1
                    if dict_start >= 0 and dict_end > dict_start:
                        dict_str = line[dict_start:dict_end]
                        # 将单引号替换为双引号（Python字典格式转JSON）
                        dict_str = dict_str.replace("'", '"')
                        log_data = json.loads(dict_str)
                        
                        # 提取评估指标（取最后一个epoch的结果）
                        if 'eval_f1' in log_data:
                            f1_value = log_data.get('eval_f1', 0)
                            if training_info['final_f1'] is None or f1_value > training_info['final_f1']:
                                training_info['final_f1'] = f1_value
                        if 'eval_accuracy' in log_data:
                            acc_value = log_data.get('eval_accuracy', 0)
                            if training_info['final_accuracy'] is None or acc_value > training_info['final_accuracy']:
                                training_info['final_accuracy'] = acc_value
                        if 'epoch' in log_data:
                            epoch_value = float(log_data.get('epoch', 0))
                            training_info['epochs'] = max(training_info['epochs'], int(epoch_value))
                        if 'train_runtime' in log_data:
                            training_info['training_time'] = log_data.get('train_runtime', 0)
                except Exception as e:
                    # 如果JSON解析失败，尝试使用eval（不安全，但作为备选）
                    try:
                        dict_start = line.find('{')
                        dict_end = line.rfind('}') + 1
                        if dict_start >= 0 and dict_end > dict_start:
                            dict_str = line[dict_start:dict_end]
                            log_data = eval(dict_str)  # 使用eval解析Python字典格式
                            
                            if 'eval_f1' in log_data:
                                f1_value = log_data.get('eval_f1', 0)
                                if training_info['final_f1'] is None or f1_value > training_info['final_f1']:
                                    training_info['final_f1'] = f1_value
                            if 'eval_accuracy' in log_data:
                                acc_value = log_data.get('eval_accuracy', 0)
                                if training_info['final_accuracy'] is None or acc_value > training_info['final_accuracy']:
                                    training_info['final_accuracy'] = acc_value
                            if 'epoch' in log_data:
                                epoch_value = float(log_data.get('epoch', 0))
                                training_info['epochs'] = max(training_info['epochs'], int(epoch_value))
                            if 'train_runtime' in log_data:
                                training_info['training_time'] = log_data.get('train_runtime', 0)
                    except:
                        pass
        
        return training_info if training_info['epochs'] > 0 else None
        
    except Exception as e:
        logger.warning(f"⚠️  分析训练日志失败: {e}")
        return None


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
        label = ID_TO_LABEL.get(int(pred_id), 'O')
        probs = probabilities[0][i].cpu().tolist()
        
        # 计算所有B-和I-标签的概率
        prob_o = probs[0] if 0 < len(probs) else 0.0
        
        # 计算所有B-标签的概率总和
        prob_b = 0.0
        prob_i = 0.0
        for label_id, label_name in ID_TO_LABEL.items():
            if label_id < len(probs):
                if label_name.startswith('B-'):
                    prob_b += probs[label_id]
                elif label_name.startswith('I-'):
                    prob_i += probs[label_id]
        
        logger.info(f"{token:<30} {label:<15} {prob_o:<10.4f} {prob_b:<10.4f} {prob_i:<10.4f}")
    
    # 提取实体（支持多种实体类型）
    entities = []
    current_entity_tokens = []
    current_entity_type = None
    
    for i, (token, pred_id) in enumerate(zip(tokens, predictions_list)):
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


def generate_scientific_recommendations(data_stats, training_info, test_results):
    """基于实际数据生成科学的改进建议"""
    logger.info("\n" + "=" * 70)
    logger.info("模型诊断与改进建议")
    logger.info("=" * 70)
    
    recommendations = []
    
    # 1. 数据量分析
    if data_stats:
        train_samples = data_stats.get('train_samples', 0)
        dev_samples = data_stats.get('dev_samples', 0)
        entity_types = data_stats.get('entity_types', 0)
        
        logger.info(f"\n📊 数据统计:")
        logger.info(f"  训练样本: {train_samples:,} 条")
        logger.info(f"  验证样本: {dev_samples:,} 条")
        logger.info(f"  实体类型: {entity_types} 种")
        
        if train_samples < 1000:
            recommendations.append({
                'level': 'warning',
                'category': '数据量',
                'issue': f'训练数据量较少（{train_samples}条）',
                'suggestion': '建议增加训练数据到至少1,000条以上，以提高模型泛化能力'
            })
        elif train_samples < 5000:
            recommendations.append({
                'level': 'info',
                'category': '数据量',
                'issue': f'训练数据量适中（{train_samples:,}条）',
                'suggestion': '数据量充足，可以继续优化模型结构或超参数'
            })
        else:
            recommendations.append({
                'level': 'success',
                'category': '数据量',
                'issue': f'训练数据量充足（{train_samples:,}条）',
                'suggestion': '数据量充足，模型有良好的训练基础'
            })
        
        if dev_samples == 0:
            recommendations.append({
                'level': 'warning',
                'category': '验证集',
                'issue': '缺少验证集',
                'suggestion': '建议添加验证集，以便在训练过程中监控模型性能，防止过拟合'
            })
        elif dev_samples < train_samples * 0.1:
            recommendations.append({
                'level': 'info',
                'category': '验证集',
                'issue': f'验证集比例较低（{dev_samples/train_samples*100:.1f}%）',
                'suggestion': '建议验证集比例达到10-20%，以便更好地评估模型性能'
            })
    
    # 2. 训练信息分析
    if training_info:
        final_f1 = training_info.get('final_f1')
        final_accuracy = training_info.get('final_accuracy')
        epochs = training_info.get('epochs', 0)
        training_time = training_info.get('training_time')
        
        logger.info(f"\n🎯 训练性能:")
        if final_f1:
            logger.info(f"  最终F1-Score: {final_f1:.4f} ({final_f1*100:.2f}%)")
        if final_accuracy:
            logger.info(f"  最终准确率: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
        if epochs:
            logger.info(f"  训练轮数: {epochs} epochs")
        if training_time:
            logger.info(f"  训练时间: {training_time:.1f} 秒 ({training_time/60:.1f} 分钟)")
        
        if final_f1:
            if final_f1 >= 0.98:
                recommendations.append({
                    'level': 'success',
                    'category': '模型性能',
                    'issue': f'F1-Score优秀（{final_f1*100:.2f}%）',
                    'suggestion': '模型性能优秀，已达到生产级别标准'
                })
            elif final_f1 >= 0.95:
                recommendations.append({
                    'level': 'info',
                    'category': '模型性能',
                    'issue': f'F1-Score良好（{final_f1*100:.2f}%）',
                    'suggestion': '模型性能良好，可以尝试进一步优化以提升到98%以上'
                })
            elif final_f1 >= 0.90:
                recommendations.append({
                    'level': 'warning',
                    'category': '模型性能',
                    'issue': f'F1-Score一般（{final_f1*100:.2f}%）',
                    'suggestion': '建议增加训练数据、调整超参数或增加训练轮数'
                })
            else:
                recommendations.append({
                    'level': 'error',
                    'category': '模型性能',
                    'issue': f'F1-Score较低（{final_f1*100:.2f}%）',
                    'suggestion': '模型性能不理想，建议检查数据质量、增加训练数据量或重新训练'
                })
        
        if epochs < 3:
            recommendations.append({
                'level': 'warning',
                'category': '训练轮数',
                'issue': f'训练轮数较少（{epochs}轮）',
                'suggestion': '建议增加训练轮数到5-10轮，确保模型充分学习'
            })
        elif epochs > 10:
            recommendations.append({
                'level': 'info',
                'category': '训练轮数',
                'issue': f'训练轮数较多（{epochs}轮）',
                'suggestion': '注意监控验证集性能，防止过拟合'
            })
    
    # 3. 测试结果分析
    if test_results:
        extracted_count = sum(1 for r in test_results if len(r) > 0)
        total_tests = len(test_results)
        
        logger.info(f"\n🧪 测试结果:")
        logger.info(f"  测试样本数: {total_tests}")
        logger.info(f"  成功提取实体: {extracted_count}/{total_tests}")
        
        if extracted_count == 0:
            recommendations.append({
                'level': 'error',
                'category': '实体提取',
                'issue': '测试样本中未提取到任何实体',
                'suggestion': '模型可能存在问题，建议检查模型加载、标签映射或重新训练'
            })
        elif extracted_count < total_tests * 0.5:
            recommendations.append({
                'level': 'warning',
                'category': '实体提取',
                'issue': f'实体提取成功率较低（{extracted_count}/{total_tests}）',
                'suggestion': '建议检查测试文本是否包含训练数据中的实体类型，或增加相关类型的训练数据'
            })
    
    # 4. 实体类型分析
    if data_stats and data_stats.get('entity_types', 0) > 0:
        entity_types_count = data_stats.get('entity_types', 0)
        if entity_types_count > 30:
            recommendations.append({
                'level': 'info',
                'category': '实体类型',
                'issue': f'实体类型较多（{entity_types_count}种）',
                'suggestion': '实体类型较多，建议检查是否有类型重叠或可以合并的类型'
            })
    
    # 输出建议
    logger.info(f"\n💡 改进建议:")
    
    if not recommendations:
        logger.info("  ✅ 模型状态良好，无需特别改进")
    else:
        # 按级别排序：error > warning > info > success
        level_order = {'error': 0, 'warning': 1, 'info': 2, 'success': 3}
        recommendations.sort(key=lambda x: level_order.get(x['level'], 4))
        
        for i, rec in enumerate(recommendations, 1):
            level_icon = {
                'error': '❌',
                'warning': '⚠️ ',
                'info': 'ℹ️ ',
                'success': '✅'
            }.get(rec['level'], '•')
            
            logger.info(f"\n  {i}. {level_icon} [{rec['category']}] {rec['issue']}")
            logger.info(f"     建议: {rec['suggestion']}")


def main():
    """主函数"""
    logger.info("=" * 70)
    logger.info("NER模型诊断")
    logger.info("=" * 70)
    
    # 1. 分析训练数据
    data_stats = analyze_training_data()
    
    # 2. 分析训练日志
    training_info = analyze_training_log()
    
    # 3. 测试模型预测
    model, tokenizer, device = load_model()
    test_results = []
    
    if model is None:
        logger.error("❌ 无法加载模型，跳过预测测试")
        generate_scientific_recommendations(data_stats, training_info, [])
        return
    
    test_texts = [
        "阿利·伯克级驱逐舰装备有宙斯盾作战系统和AN/SPY-13D相控阵雷达，主要用于防空作战。",
        "该舰装备了战斧巡航导弹和标准系列防空导弹。",
        "美国海军计划建造朱姆沃尔特级驱逐舰作为下一代主力驱逐舰。"
    ]
    
    for text in test_texts:
        entities = test_prediction_detailed(text, model, tokenizer, device)
        test_results.append(entities)
    
    # 4. 生成科学的改进建议
    generate_scientific_recommendations(data_stats, training_info, test_results)
    
    logger.info("\n" + "=" * 70)
    logger.info("诊断完成")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
