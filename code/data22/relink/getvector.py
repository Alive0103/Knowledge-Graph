# import os
# import json
# import torch
# from opencc import OpenCC  # 用于繁体转简体
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from transformers import BertTokenizer, BertModel  # 导入BERT相关库

# # 初始化 OpenCC
# cc = OpenCC('t2s')  # 将繁体转为简体

# # 模型下载的地址
# model_name = 'D:/model/chinese-roberta-wwm-ext-large'

# # BERT模型初始化
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertModel.from_pretrained(model_name)
# model.eval()

# # 处理单个项目
# def process_single_item(item):
#     new_data = {}
#     try:
#         # 提取 entityLabel 作为标签
#         entity_label = item.get('entityLabel', '')  # 默认值为空字符串
#         print(f"处理的实体标签: {entity_label}")  # 添加打印语句以查看标签内容
#         if isinstance(entity_label, dict):  # 如果它是一个字典（复杂对象）
#             new_data['label'] = entity_label.get('value', '')  # 提取实际值
#         else:
#             if any('\u4e00' <= char <= '\u9fff' for char in entity_label):  # 判断是否包含中文字符
#                 new_data['label'] = cc.convert(entity_label)  # 繁体转简体
#             else:
#                 new_data['label'] = entity_label  # 直接使用英文标签

#         # 提取别名
#         aliases = item.get('aliases', {})
#         new_data['aliases_en'] = aliases.get('en', [])  
#         new_data['aliases_zh'] = [cc.convert(alias) for alias in aliases.get('zh', [])]  

#         # 提取 content
#         new_data['content'] = cc.convert(item.get('content', ''))  

#         # 提取 descriptions
#         new_data['descriptions_zh'] = cc.convert(item.get('description_zh', ''))  
#         new_data['descriptions_en'] = item.get('description_en', '')  

#         # 生成向量
#         descriptions_zh_vector = generate_vector(new_data['descriptions_zh'])
#         print(f"生成的中文描述向量: {descriptions_zh_vector}")  # 打印生成的向量

#         # 添加向量到新数据字典
#         new_data['descriptions_zh_vector'] = descriptions_zh_vector

#     except Exception as e:
#         print(f"处理实体 {entity_label} 时出错: {e}")

#     return new_data  # 返回处理后的数据

# # 生成文本向量的函数
# def generate_vector(text):
#     if text:
#         inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
#         with torch.no_grad():
#             outputs = model(**inputs)
#         # 取 [CLS] token 的输出作为句子的向量表示
#         vector = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
#         return vector.tolist()  # 将向量转换为列表
#     return None

# # 处理 JSONL 文件并保存新的 JSONL 文件
# def process_jsonl_file(input_path, output_path):
#     print(f"开始处理文件: {input_path}")  # 添加打印语句以确认文件处理开始
#     with open(input_path, 'r', encoding='utf-8') as f, open(output_path, 'w', encoding='utf-8') as out_f:
#         with ThreadPoolExecutor(max_workers=8) as executor:
#             futures = []
#             for line in f:
#                 item = json.loads(line.strip())  
#                 future = executor.submit(process_single_item, item)
#                 futures.append(future)

#             for future in as_completed(futures):
#                 try:
#                     result = future.result()
#                     if result:  # 确保结果不为空
#                         out_f.write(json.dumps(result, ensure_ascii=False) + '\n')  # 将结果写入新的 JSONL 文件
#                 except Exception as e:
#                     print(f"处理时出错: {e}")

# if __name__ == "__main__":
#     input_file = 'D:\\毕设\\wikidata\\ytest.jsonl'  # 替换为实际输入文件路径
#     output_file = 'D:\\毕设\\wikidata\\ytest2.jsonl'  # 输出文件路径
#     print(f"处理文件: {input_file}")
#     process_jsonl_file(input_file, output_file)
#     print("文件处理完成。")

import json
import torch
from transformers import BertTokenizer, BertModel

# 模型路径
model_name = 'D:/model/chinese-roberta-wwm-ext-large'

# 初始化 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()

# 生成文本向量的函数
def generate_vector(text):
    if text:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # 取 [CLS] token 的输出作为句子的向量表示
        vector = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        return vector.tolist()  # 将向量转换为列表
    return None

# 处理单条数据
def process_single_item(item):
    new_data = item.copy()  # 复制原始数据

    # 生成中文描述向量
    zh_description_vector = generate_vector(item.get('zh_description', ''))
    new_data['zh_descriptions_vector'] = zh_description_vector

    # 生成英文描述向量
    en_description_vector = generate_vector(item.get('en_description', ''))
    new_data['en_descriptions_vector'] = en_description_vector

    return new_data

# 处理 JSONL 文件并保存新的 JSONL 文件
def process_jsonl_file(input_path, output_path):
    print(f"开始处理文件: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f, open(output_path, 'w', encoding='utf-8') as out_f:
        for line in f:
            item = json.loads(line.strip())
            result = process_single_item(item)
            if result:
                out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"文件处理完成，结果已保存到 {output_path}")

if __name__ == "__main__":
    input_file = 'zh.jsonl'  # 输入文件路径
    output_file = 'new_zh.jsonl'  # 输出文件路径
    process_jsonl_file(input_file, output_file)

    input_file = 'zh2.jsonl'  # 输入文件路径
    output_file = 'new_zh2.jsonl'  # 输出文件路径
    process_jsonl_file(input_file, output_file)

    input_file = 'en.jsonl'  # 输入文件路径
    output_file = 'new_en.jsonl'  # 输出文件路径
    process_jsonl_file(input_file, output_file)

    input_file = 'en2.jsonl'  # 输入文件路径
    output_file = 'new_en2.jsonl'  # 输出文件路径
    process_jsonl_file(input_file, output_file)