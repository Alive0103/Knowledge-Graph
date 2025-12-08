"""
通用向量生成模块
支持使用基础模型或微调后的NER模型（提取encoder部分）进行向量生成
"""
import os
import torch
import numpy as np
from transformers import BertTokenizer, BertModel, BertForTokenClassification
import logging

logger = logging.getLogger(__name__)

# 全局变量
_vector_model = None
_vector_tokenizer = None
_vector_device = None
_model_path = None


def load_vector_model(model_path=None, use_finetuned=True):
    """
    加载向量生成模型
    
    Args:
        model_path: 模型路径，如果为None则自动选择
        use_finetuned: 是否优先使用微调后的模型
    
    Returns:
        (model, tokenizer, device): 模型、分词器、设备
    """
    global _vector_model, _vector_tokenizer, _vector_device, _model_path
    
    # 如果已经加载过相同的模型，直接返回
    if _vector_model is not None and _vector_tokenizer is not None:
        if model_path is None or _model_path == model_path:
            return _vector_model, _vector_tokenizer, _vector_device
    
    # 确定模型路径（使用绝对路径，避免工作目录问题）
    if model_path is None:
        # 获取脚本所在目录（work_wyy），然后构建模型路径
        # 尝试多种方式找到work_wyy目录
        script_dir = None
        current_file = os.path.abspath(__file__)

        # 方法1: 如果vector_model.py在work_wyy目录下
        if 'work_wyy' in current_file:
            work_wyy_idx = current_file.find('work_wyy')
            script_dir = current_file[:current_file.find('work_wyy') + len('work_wyy')]
        else:
            # 方法2: 假设当前文件在work_wyy目录下
            script_dir = os.path.dirname(current_file)

        # 优先使用微调后的模型（如果存在）
        finetuned_path = os.path.join(script_dir, 'model', 'ner_finetuned')
        base_path = os.path.join(script_dir, 'model', 'chinese-roberta-wwm-ext-large')

        # 转换为绝对路径
        finetuned_path = os.path.abspath(finetuned_path)
        base_path = os.path.abspath(base_path)

        if use_finetuned and os.path.exists(finetuned_path):
            model_path = finetuned_path
            logger.info(f"✅ 使用微调后的模型: {model_path}")
        elif os.path.exists(base_path):
            model_path = base_path
            logger.info(f"⚠️  使用基础模型: {model_path} (微调模型不存在: {finetuned_path})")
        else:
            # 如果都找不到，尝试相对路径（兼容旧代码）
            finetuned_path_rel = './model/ner_finetuned'
            base_path_rel = './model/chinese-roberta-wwm-ext-large'
            if use_finetuned and os.path.exists(finetuned_path_rel):
                model_path = os.path.abspath(finetuned_path_rel)
                logger.info(f"✅ 使用微调后的模型（相对路径）: {model_path}")
            elif os.path.exists(base_path_rel):
                model_path = os.path.abspath(base_path_rel)
                logger.info(f"⚠️  使用基础模型（相对路径）: {model_path}")
            else:
                raise FileNotFoundError(
                    f"无法找到模型！尝试的路径：\n"
                    f"  微调模型: {finetuned_path}\n"
                    f"  基础模型: {base_path}\n"
                    f"  相对路径微调: {finetuned_path_rel}\n"
                    f"  相对路径基础: {base_path_rel}\n"
                    f"当前工作目录: {os.getcwd()}"
                )
    
    _model_path = model_path
    
    try:
        # 加载tokenizer
        _vector_tokenizer = BertTokenizer.from_pretrained(model_path)
        
        # 尝试加载为NER模型（微调后的模型）
        try:
            ner_model = BertForTokenClassification.from_pretrained(model_path)
            # 从NER模型中提取bert encoder部分用于向量生成
            _vector_model = ner_model.bert
            logger.info(f"从NER模型中提取encoder用于向量生成: {model_path}")
        except:
            # 如果不是NER模型，直接加载BertModel
            _vector_model = BertModel.from_pretrained(model_path)
            logger.info(f"加载基础BERT模型用于向量生成: {model_path}")
        
        _vector_model.eval()
        
        # 检查GPU
        if torch.cuda.is_available():
            _vector_device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            logger.info(f"✅ 使用GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            _vector_device = torch.device('cpu')
            logger.info("⚠️ 使用CPU模式")
        
        _vector_model = _vector_model.to(_vector_device)
        
        # 获取模型维度
        model_dim = _vector_model.config.hidden_size
        logger.info(f"模型维度: {model_dim}")
        
        return _vector_model, _vector_tokenizer, _vector_device
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise


def generate_vector(text, model_path=None, use_finetuned=True, target_dim=1024):
    """
    生成文本向量
    
    Args:
        text: 输入文本
        model_path: 模型路径（可选）
        use_finetuned: 是否优先使用微调后的模型
        target_dim: 目标向量维度（默认1024，ES要求）
    
    Returns:
        向量列表（已L2归一化，维度为target_dim）
    """
    global _vector_model, _vector_tokenizer, _vector_device
    
    if not text or not str(text).strip():
        return None
    
    # 加载模型（如果未加载）
    if _vector_model is None or _vector_tokenizer is None:
        load_vector_model(model_path, use_finetuned)
    
    try:
        # Tokenize
        inputs = _vector_tokenizer(
            str(text),
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(_vector_device) for k, v in inputs.items()}
        
        # 生成向量
        with torch.no_grad():
            outputs = _vector_model(**inputs)
            # 使用[CLS] token的向量
            vector = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        
        # L2归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        else:
            return None
        
        # 处理维度
        vector_dim = len(vector)
        if vector_dim == target_dim:
            pass
        elif vector_dim < target_dim:
            # 填充到目标维度
            vector = np.pad(vector, (0, target_dim - vector_dim), 'constant', constant_values=0)
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
        else:
            # 截断到目标维度
            vector = vector[:target_dim]
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
        
        return vector.tolist()
        
    except Exception as e:
        logger.error(f"向量生成失败: {e}")
        return None


def batch_generate_vectors(texts, model_path=None, use_finetuned=True, target_dim=1024, batch_size=32):
    """
    批量生成向量
    
    Args:
        texts: 文本列表
        model_path: 模型路径（可选）
        use_finetuned: 是否优先使用微调后的模型
        target_dim: 目标向量维度
        batch_size: 批次大小
    
    Returns:
        向量列表（与输入文本顺序一致）
    """
    global _vector_model, _vector_tokenizer, _vector_device
    
    if not texts:
        return []
    
    # 加载模型（如果未加载）
    if _vector_model is None or _vector_tokenizer is None:
        load_vector_model(model_path, use_finetuned)
    
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        valid_texts = [(idx, text) for idx, text in enumerate(batch_texts) if text and str(text).strip()]
        
        if not valid_texts:
            results.extend([None] * len(batch_texts))
            continue
        
        try:
            # 批量tokenize
            batch_texts_list = [str(text) for _, text in valid_texts]
            inputs = _vector_tokenizer(
                batch_texts_list,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(_vector_device) for k, v in inputs.items()}
            
            # 批量生成向量
            with torch.no_grad():
                outputs = _vector_model(**inputs)
                batch_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # 处理每个向量
            batch_results = [None] * len(batch_texts)
            for j, (orig_idx, _) in enumerate(valid_texts):
                vector = batch_vectors[j]
                
                # L2归一化
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
                else:
                    continue
                
                # 处理维度
                vector_dim = len(vector)
                if vector_dim == target_dim:
                    pass
                elif vector_dim < target_dim:
                    vector = np.pad(vector, (0, target_dim - vector_dim), 'constant', constant_values=0)
                    norm = np.linalg.norm(vector)
                    if norm > 0:
                        vector = vector / norm
                else:
                    vector = vector[:target_dim]
                    norm = np.linalg.norm(vector)
                    if norm > 0:
                        vector = vector / norm
                
                batch_results[orig_idx] = vector.tolist()
            
            results.extend(batch_results)
            
        except Exception as e:
            logger.error(f"批量向量生成失败: {e}")
            # 失败时返回None
            results.extend([None] * len(batch_texts))
    
    return results

