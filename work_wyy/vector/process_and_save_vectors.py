#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
处理数据并生成向量，然后保存到新的JSONL文件中
"""

import json
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import os
import numpy as np

# 模型路径 - 根据实际模型确定维度
model_name = './model/chinese-roberta-wwm-ext-large'  # 使用large模型
VECTOR_DIMS = 1024  # large模型是1024维，base模型是768维

# 初始化 BERT 模型和分词器
print("加载BERT模型...")
try:
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()
    print(f"✓ BERT模型加载成功: {model_name}")
    print(f"  预期向量维度: {VECTOR_DIMS}")
except Exception as e:
    print(f"✗ BERT模型加载失败: {e}")
    exit(1)

def generate_vector(text):
    """生成文本向量"""
    if text and text.strip():
        try:
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            
            vector = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            
            # 检查向量维度
            if hasattr(vector, 'shape'):
                if len(vector.shape) == 0:
                    vector = vector.reshape(1)
                actual_dims = vector.shape[0] if len(vector.shape) == 1 else vector.shape[1]
                
                if actual_dims != VECTOR_DIMS:
                    print(f"警告: 向量维度不匹配! 期望: {VECTOR_DIMS}, 实际: {actual_dims}")
                    return None
                
                # 添加L2归一化
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
                    
                return vector.tolist()
            else:
                print("警告: 向量格式异常")
                return None
                
        except Exception as e:
            # 不打印每个向量生成的错误，避免输出过多
            return None
    return None

def process_single_item(item):
    """处理单条数据项 - 确保所有数据都生成向量"""
    # 提取关键字段
    label = item.get("label", "")
    link = item.get("wikipedia") or item.get("wikipediaLink", "")
    aliases_en = item.get("en_aliases") or item.get("aliases_en", [])
    aliases_zh = item.get("zh_aliases") or item.get("aliases_zh", [])
    descriptions_en = item.get("en_description") or item.get("descriptions_en", "")
    descriptions_zh = item.get("zh_description") or item.get("descriptions_zh", "")
    content = item.get("content", "")
    
    # 构建数据对象
    new_data = {
        "label": label,
        "wikipediaLink": link,  # 统一使用 wikipediaLink 字段名
        "en_aliases": aliases_en if isinstance(aliases_en, list) else [],
        "zh_aliases": aliases_zh if isinstance(aliases_zh, list) else [],
        "en_description": descriptions_en,
        "zh_description": descriptions_zh,
        "content": content
    }

    # 为中文描述生成向量
    # 优先使用描述文本，如果没有或者太短，则使用标签和别名
    zh_text_for_vector = ""
    if descriptions_zh and len(descriptions_zh.strip()) > 10:
        zh_text_for_vector = descriptions_zh
    elif label:
        # 使用标签+中文别名作为替代
        zh_text_for_vector = label
        if aliases_zh and isinstance(aliases_zh, list):
            zh_text_for_vector += " " + " ".join(aliases_zh[:5])  # 限制别名数量
    
    if zh_text_for_vector:
        vector = generate_vector(zh_text_for_vector)
        if vector and len(vector) == VECTOR_DIMS:
            new_data["descriptions_zh_vector"] = vector

    # 为英文描述生成向量
    # 优先使用描述文本，如果没有或者太短，则使用标签和别名
    en_text_for_vector = ""
    if descriptions_en and len(descriptions_en.strip()) > 10:
        en_text_for_vector = descriptions_en
    elif label:
        # 使用标签+英文别名作为替代
        en_text_for_vector = label
        if aliases_en and isinstance(aliases_en, list):
            en_text_for_vector += " " + " ".join(aliases_en[:5])  # 限制别名数量
    
    if en_text_for_vector:
        vector = generate_vector(en_text_for_vector)
        if vector and len(vector) == VECTOR_DIMS:
            new_data["descriptions_en_vector"] = vector

    return new_data

def count_lines(filename):
    """快速计算文件行数"""
    with open(filename, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

def process_and_save_vectors(input_path, output_path, batch_size=1000):
    """处理JSONL文件并保存带向量的数据到新文件"""
    print(f"开始处理文件: {input_path}")
    
    total_lines = count_lines(input_path)
    print(f"文件总行数: {total_lines}")
    
    vector_count = 0
    processed_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        progress_bar = tqdm(total=total_lines, desc="处理进度", unit="条")
        
        for line_num, line in enumerate(infile, 1):
            try:
                if not line.strip():
                    progress_bar.update(1)
                    continue
                    
                data = json.loads(line.strip())
                transformed_data = process_single_item(data)
                
                # 统计向量数量
                if 'descriptions_zh_vector' in transformed_data:
                    vector_count += 1
                if 'descriptions_en_vector' in transformed_data:
                    vector_count += 1
                
                # 写入到输出文件
                outfile.write(json.dumps(transformed_data, ensure_ascii=False) + '\n')
                
                processed_count += 1
                progress_bar.update(1)
                
                # 定期刷新文件缓冲区
                if line_num % batch_size == 0:
                    outfile.flush()
                    progress_bar.set_postfix({
                        '已处理': processed_count, 
                        '向量数': vector_count
                    })
                
            except Exception as e:
                print(f"第{line_num}行处理失败: {str(e)[:100]}")
                # 即使出错也继续处理下一行
                progress_bar.update(1)

        progress_bar.close()

    print(f"\n{'='*50}")
    print(f"处理完成!")
    print(f"处理数据条数: {processed_count}条")
    print(f"生成向量总数: {vector_count}个")
    print(f"向量生成率: {vector_count/(processed_count*2)*100:.1f}%")
    print(f"输出文件: {output_path}")
    print(f"{'='*50}")

if __name__ == "__main__":
    print("=" * 60)
    print("开始处理数据并生成向量")
    print("=" * 60)
    
    # 处理数据文件
    data_files = [
        ("zh_wiki_v2.jsonl", "zh_wiki_v2_with_vectors.jsonl"),
        ("en_wiki_v3.jsonl", "en_wiki_v3_with_vectors.jsonl")
    ]
    
    processed_files = []
    for input_file, output_file in data_files:
        if os.path.exists(input_file):
            print(f"\n处理数据文件: {input_file} -> {output_file}")
            process_and_save_vectors(input_file, output_file)
            processed_files.append((input_file, output_file))
        else:
            print(f"警告: 数据文件 {input_file} 不存在")
    
    if not processed_files:
        print("错误: 未找到任何数据文件")
        exit(1)
    
    print(f"\n已完成处理以下文件:")
    for input_f, output_f in processed_files:
        print(f"  - {input_f} -> {output_f}")
    
    print("\n处理完成! 现在可以运行 toesdata.py 导入这些带向量的数据文件")