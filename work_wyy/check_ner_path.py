#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查NER模型路径是否正确
"""

import os
import sys

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 模拟ner_extract_entities.py中的路径计算
_current_file = os.path.abspath(__file__)  # 当前文件的绝对路径
_current_dir = os.path.dirname(_current_file)  # work_wyy目录的绝对路径
_ner_file = os.path.join(_current_dir, 'ner', 'ner_extract_entities.py')

if os.path.exists(_ner_file):
    # 从ner_extract_entities.py中导入路径
    import importlib.util
    spec = importlib.util.spec_from_file_location("ner_extract_entities", _ner_file)
    ner_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ner_module)
    
    print("=" * 70)
    print("NER模型路径检查")
    print("=" * 70)
    print(f"\nwork_wyy目录: {ner_module._parent_dir}")
    print(f"\n基础模型路径: {ner_module.BASE_MODEL_PATH}")
    print(f"模型是否存在: {os.path.exists(ner_module.BASE_MODEL_PATH)}")
    
    if os.path.exists(ner_module.BASE_MODEL_PATH):
        print("✅ 模型路径正确！")
        # 检查模型文件
        config_file = os.path.join(ner_module.BASE_MODEL_PATH, 'config.json')
        model_file = os.path.join(ner_module.BASE_MODEL_PATH, 'pytorch_model.bin')
        tokenizer_file = os.path.join(ner_module.BASE_MODEL_PATH, 'tokenizer.json')
        
        print(f"\n模型文件检查:")
        print(f"  config.json: {'✅' if os.path.exists(config_file) else '❌'}")
        print(f"  pytorch_model.bin: {'✅' if os.path.exists(model_file) else '❌'}")
        print(f"  tokenizer.json: {'✅' if os.path.exists(tokenizer_file) else '❌'}")
    else:
        print("❌ 模型路径不存在！")
        print(f"\n请确保模型位于: {ner_module.BASE_MODEL_PATH}")
        print(f"\n当前工作目录: {os.getcwd()}")
        print(f"work_wyy目录: {ner_module._parent_dir}")
        print(f"期望的模型路径: {ner_module.BASE_MODEL_PATH}")
else:
    print(f"❌ 找不到ner_extract_entities.py: {_ner_file}")

