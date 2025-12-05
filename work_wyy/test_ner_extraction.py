#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试NER提取功能
用于验证NER模型是否能正确提取实体词
"""

import sys
import os

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from ner.ner_extract_entities import extract_entities_from_text, load_ner_model

def test_ner_extraction():
    """测试NER提取"""
    print("=" * 70)
    print("NER实体提取测试")
    print("=" * 70)
    
    # 加载模型
    print("\n1. 加载NER模型...")
    if load_ner_model():
        print("✅ NER模型加载成功")
    else:
        print("❌ NER模型加载失败")
        return
    
    # 测试文本（类似图片中的内容）
    test_texts = [
        "阿利·伯克级驱逐舰装备有宙斯盾作战系统和AN/SPY-13D相控阵雷达，主要用于防空作战。",
        "该舰装备了战斧巡航导弹和标准系列防空导弹。",
        "美国海军计划建造朱姆沃尔特级驱逐舰作为下一代主力驱逐舰。",
    ]
    
    print("\n2. 测试实体提取...")
    print("-" * 70)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n测试文本 {i}: {text}")
        entities = extract_entities_from_text(text, verbose=True)
        print(f"提取的实体: {entities}")
        print("-" * 70)

if __name__ == "__main__":
    test_ner_extraction()

