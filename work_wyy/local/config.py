#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一配置文件
管理所有路径、ES连接、数据源开关等配置
"""

import os

# ============================================================================
# 路径配置
# ============================================================================

# 获取配置文件所在目录（local目录）
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

# 工作目录（work_wyy）
WORK_DIR = os.path.dirname(_CONFIG_DIR)

# 数据目录
DATA_DIR = os.path.join(WORK_DIR, 'data')

# 模型目录
MODEL_DIR = os.path.join(WORK_DIR, 'model')

# 基础模型路径
BASE_MODEL_PATH = os.path.join(MODEL_DIR, 'chinese-roberta-wwm-ext-large')

# 微调模型路径
FINETUNED_MODEL_PATH = os.path.join(MODEL_DIR, 'ner_finetuned')

# 日志目录
TRAINLOG_DIR = os.path.join(WORK_DIR, 'trainlog')

# 数据子目录
TRAINDATA_DIR = os.path.join(DATA_DIR, 'traindata')
CCKS_NER_DIR = os.path.join(DATA_DIR, 'ccks_ner', 'militray', 'PreModel_Encoder_CRF')
TRAIN_TXT_FILE = os.path.join(DATA_DIR, 'train.txt')
MSRA_DIR = os.path.join(DATA_DIR, 'nlp_datasets', 'ner', 'msra')

# 向量化数据文件
ENTITY_WORDS_ZH_FILE = os.path.join(DATA_DIR, 'entity_words_zh.jsonl')
ENTITY_WORDS_EN_FILE = os.path.join(DATA_DIR, 'entity_words_en.jsonl')

# 原始数据文件（用于向量化）
ZH_WIKI_FILE = os.path.join(DATA_DIR, 'zh_wiki_v2.jsonl')
EN_WIKI_FILE = os.path.join(DATA_DIR, 'en_wiki_v3.jsonl')

# ============================================================================
# Elasticsearch 配置
# ============================================================================

# ES连接方式：'local' 或 'aliyun'
ES_MODE = 'local'  # 本地ES或阿里云ES

# 本地ES配置（Docker环境）
ES_LOCAL_CONFIG = {
    'url': 'http://elasticsearch:9200',
    'request_timeout': 180,
    'max_retries': 3,
    'retry_on_timeout': True,
    'basic_auth': None,  # 本地ES通常不需要认证
    'headers': None,
    'http_compress': False
}

# 阿里云ES配置
ES_ALIYUN_CONFIG = {
    'url': 'http://kgcode-xw7.public.cn-hangzhou.es-serverless.aliyuncs.com:9200',
    'username': 'kgcode-xw7',
    'password': 'Ln216812_',
    'request_timeout': 180,
    'max_retries': 3,
    'retry_on_timeout': True,
    'headers': {"accept": "application/vnd.elasticsearch+json;compatible-with=8"},
    'http_compress': True
}

# ES索引名称
ES_INDEX_NAME = 'data'

# ============================================================================
# 数据源开关配置
# ============================================================================

# 训练数据源开关（True=启用，False=禁用）
DATA_SOURCE_SWITCHES = {
    # traindata目录（主要训练数据）
    'traindata': True,
    
    # ccks_ner目录（CCKS军事领域数据）
    'ccks_json': True,        # ccks_8_data_v2/train/*.json (400个文件)
    'ccks_validate': True,     # validate_data.json（如果有标注）
    'ccks_fold0': True,        # fold0/train (BIO格式)
    'ccks_fold1': True,        # fold1/train (BIO格式)
    'ccks_fold2': True,        # fold2/train (BIO格式)
    'ccks_fold3': True,        # fold3/train (BIO格式)
    'ccks_fold4': True,        # fold4/train (BIO格式)
    
    # train.txt（JSONL格式训练数据）
    'train_txt': True,
    
    # MSRA通用NER数据
    'msra_train': True,       # msra_train_bio.txt
    'msra_test': True,        # msra_test_bio.txt（作为验证集）
}

# ============================================================================
# 模型训练配置
# ============================================================================

# 训练超参数
TRAINING_CONFIG = {
    'max_length': 512,
    'batch_size': 8,
    'learning_rate': 2e-5,
    'num_epochs': 5,
    'warmup_steps': 100,
    'weight_decay': 0.01,
    'logging_steps': 50,
    'save_strategy': 'epoch',
    'save_total_limit': 3,
    'fp16': True,  # 如果支持，使用混合精度
}

# ============================================================================
# 向量化配置
# ============================================================================

# 向量维度
VECTOR_DIMS = 1024

# 批量处理大小
VECTOR_BATCH_SIZE = 64

# 是否使用微调后的模型进行向量化
USE_FINETUNED_FOR_VECTORIZATION = True

# ============================================================================
# 实体提取配置
# ============================================================================

# 最小文本长度（用于过滤太短的文本）
MIN_TEXT_LENGTH = 2

# 中文实体词最小长度
MIN_ENTITY_LENGTH_ZH = 2

# 英文实体词最小长度
MIN_ENTITY_LENGTH_EN = 3

# ============================================================================
# 检索系统配置
# ============================================================================

# 智谱AI API Key
ZHIPUAI_API_KEY = "1a2a485fe1fc4bd5aa0d965bf452c8c8.se8RZdT8cH8skEDo"

# 向量检索top_k
VECTOR_SEARCH_TOP_K = 30

# LLM重排序top_k
LLM_RERANK_TOP_K = 30

# ============================================================================
# 辅助函数
# ============================================================================

def get_es_config():
    """获取ES配置（根据ES_MODE）"""
    if ES_MODE == 'local':
        return ES_LOCAL_CONFIG
    else:
        return ES_ALIYUN_CONFIG


def create_es_client():
    """创建Elasticsearch客户端"""
    from elasticsearch import Elasticsearch
    
    config = get_es_config()
    
    if ES_MODE == 'local':
        # 本地ES连接（不需要认证）
        es = Elasticsearch(
            config['url'],
            request_timeout=config['request_timeout'],
            max_retries=config['max_retries'],
            retry_on_timeout=config['retry_on_timeout']
        )
    else:
        # 阿里云ES连接（需要认证）
        es = Elasticsearch(
            config['url'],
            basic_auth=(config['username'], config['password']),
            headers=config['headers'],
            http_compress=config['http_compress'],
            request_timeout=config['request_timeout'],
            max_retries=config['max_retries'],
            retry_on_timeout=config['retry_on_timeout']
        )
    
    return es


def get_data_base_dir():
    """获取数据基础目录"""
    return DATA_DIR


def is_data_source_enabled(source_name):
    """检查数据源是否启用"""
    return DATA_SOURCE_SWITCHES.get(source_name, False)

