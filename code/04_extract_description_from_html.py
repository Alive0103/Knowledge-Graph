# # # from transformers import BertTokenizer, BertModel
# # # import torch

# # # # 模型下载的地址
# # # model_name = 'D:\model\chinese-roberta-wwm-ext-large'

# # # def embeddings(docs, max_length=300):
# # #     tokenizer = BertTokenizer.from_pretrained(model_name)
# # #     model = BertModel.from_pretrained(model_name)
# # #     model.eval()  # 设置模型为评估模式
# # #     # print("Tokenizer and model loaded successfully.")

# # #     # 对文本进行分词、编码和填充
# # #     input_ids = []
# # #     attention_masks = []
# # #     for doc in docs:
# # #         # print("Input document:", doc)  # 打印输入文本
# # #         encoded_dict = tokenizer.encode_plus(
# # #             doc,
# # #             add_special_tokens=True,
# # #             max_length=max_length,
# # #             padding='max_length',
# # #             truncation=True,
# # #             return_attention_mask=True,
# # #             return_tensors='pt'
# # #         )
# # #         input_ids.append(encoded_dict['input_ids'])
# # #         attention_masks.append(encoded_dict['attention_mask'])

# # #     input_ids = torch.cat(input_ids, dim=0)
# # #     attention_masks = torch.cat(attention_masks, dim=0)

# # #     # 打印调试信息
# # #     # print("Input IDs shape:", input_ids.shape)
# # #     # print("Attention Masks shape:", attention_masks.shape)

# # #     # 前向传播
# # #     with torch.no_grad():
# # #         outputs = model(input_ids, attention_mask=attention_masks)
# # #         # print("Model outputs:", outputs)  # 打印输出对象


# # #     # 提取最后一层的CLS向量作为文本表示
# # #     last_hidden_state = outputs.last_hidden_state
# # #     cls_embeddings = last_hidden_state[:, 0, :]
# # #     return cls_embeddings

# # # if __name__ == '__main__':
# # #     try:
# # #         res = embeddings(["你好，你叫什么名字"])
# # #         print(res)
# # #         print(len(res))
# # #         print(len(res[0]))
# # #     except Exception as e:
# # #         print("Error:", e)

# # import json
# # import os
# # from transformers import BertTokenizer, BertModel
# # import torch
# # from concurrent.futures import ThreadPoolExecutor, as_completed

# # # 模型下载的地址
# # model_name = 'D:/model/chinese-roberta-wwm-ext-large'

# # # BERT模型初始化
# # tokenizer = BertTokenizer.from_pretrained(model_name)
# # model = BertModel.from_pretrained(model_name)
# # model.eval()

# # def embeddings(text, max_length=300):
# #     encoded_dict = tokenizer.encode_plus(
# #         text,
# #         add_special_tokens=True,
# #         max_length=max_length,
# #         padding='max_length',
# #         truncation=True,
# #         return_attention_mask=True,
# #         return_tensors='pt'
# #     )
# #     input_ids = encoded_dict['input_ids']
# #     attention_mask = encoded_dict['attention_mask']

# #     with torch.no_grad():
# #         outputs = model(input_ids, attention_mask=attention_mask)

# #     # 提取CLS向量
# #     last_hidden_state = outputs.last_hidden_state
# #     cls_embedding = last_hidden_state[:, 0, :].numpy().tolist()

# #     return cls_embedding

# # def process_entry(data):
# #     try:
# #         # 获取description_zh
# #         description_zh = data.get('description_zh', "")  # 确保字段存在

# #         # 生成向量
# #         zh_vector = embeddings(description_zh) if description_zh else []

# #         # 新增向量字段
# #         data['descriptions_zh_vector'] = zh_vector

# #         return data
# #     except Exception as e:
# #         print(f"Error processing entry: {e}")
# #         return None

# # def process_jsonl(input_file, output_file, max_workers=64):
# #     with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
# #         # 使用ThreadPoolExecutor进行多线程处理
# #         with ThreadPoolExecutor(max_workers=max_workers) as executor:
# #             future_to_data = {}

# #             for line in infile:
# #                 line = line.strip()  # 去掉两端的空白符
# #                 if not line:  # 如果是空行，则跳过
# #                     continue
# #                 try:
# #                     data = json.loads(line)
# #                     future = executor.submit(process_entry, data)
# #                     future_to_data[future] = data  # 保存未来对象和原始数据
# #                 except json.JSONDecodeError as e:
# #                     print(f"Error decoding JSON in line: {line} - {e}")
# #                     continue  # 如果解析失败则跳过此行
# #                 except Exception as e:
# #                     print(f"Unexpected error in line: {line} - {e}")
# #                     continue  # 捕获其他未预料的错误

# #             for future in as_completed(future_to_data):
# #                 processed_data = future.result()
# #                 if processed_data:
# #                     outfile.write(json.dumps(processed_data, ensure_ascii=False) + '\n')

# # if __name__ == '__main__':
# #     input_path = 'D:\\毕设\\wikidata\\wikidata_zh.jsonl'  # 你的输入文件路径
# #     output_path = 'D:\\毕设\\wikidata\\wikidata_vector.jsonl'  # 输出文件路径
# #     process_jsonl(input_path, output_path, max_workers=64)

# import json
# import os
# from transformers import BertTokenizer, BertModel
# import torch
# from concurrent.futures import ThreadPoolExecutor, as_completed

# # 模型下载的地址
# model_name = 'D:/model/chinese-roberta-wwm-ext-large'

# # BERT模型初始化
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertModel.from_pretrained(model_name)
# model.eval()

# def embeddings(text, max_length=300):
#     encoded_dict = tokenizer.encode_plus(
#         text,
#         add_special_tokens=True,
#         max_length=max_length,
#         padding='max_length',
#         truncation=True,
#         return_attention_mask=True,
#         return_tensors='pt'
#     )
#     input_ids = encoded_dict['input_ids']
#     attention_mask = encoded_dict['attention_mask']

#     with torch.no_grad():
#         outputs = model(input_ids, attention_mask=attention_mask)

#     # 提取CLS向量
#     last_hidden_state = outputs.last_hidden_state
#     cls_embedding = last_hidden_state[:, 0, :].numpy().tolist()

#     return cls_embedding

# def process_entry(data):
#     try:
#         # 获取description_zh, description_en和content
#         description_zh = data.get('description_zh', "")
#         # description_en = data.get('description_en', "")
#         # content = data.get('content', "")

#         # 生成向量
#         zh_vector = embeddings(description_zh) if description_zh else []
#         # en_vector = embeddings(description_en) if description_en else []
#         # content_vector = embeddings(content) if content else []

#         # 新增向量字段
#         data['descriptions_zh_vector'] = zh_vector
#         # data['descriptions_en_vector'] = en_vector
#         # data['content_vector'] = content_vector

#         return data
#     except Exception as e:
#         print(f"Error processing entry: {e}")
#         return None

# def process_jsonl(input_file, output_file, max_workers=64):
#     with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
#         data_list = [json.loads(line) for line in infile]

#         # 使用ThreadPoolExecutor进行多线程处理
#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             future_to_data = {executor.submit(process_entry, data): data for data in data_list}
            
#             for future in as_completed(future_to_data):
#                 processed_data = future.result()
#                 if processed_data:
#                     outfile.write(json.dumps(processed_data, ensure_ascii=False) + '\n')

# if __name__ == '__main__':
#     input_folder = 'last'  # 输入文件夹路径
#     output_folder = 'zhvector'  # 输出文件夹路径

#     # 确保输出文件夹存在
#     os.makedirs(output_folder, exist_ok=True)

#     # 遍历输入文件夹中的每个文件
#     for filename in os.listdir(input_folder):
#         if filename.endswith('.jsonl'):
#             input_path = os.path.join(input_folder, filename)
#             output_path = os.path.join(output_folder, f'{filename}')  # 输出文件命名为 processed_原文件名
#             process_jsonl(input_path, output_path, max_workers=8)


# #从last中的文件 取出有中文wikipedia的数据
# import os
# import json

# input_folder = "last"
# output_folder = "last2"
# os.makedirs(output_folder, exist_ok=True)

# def filter_wikipedia_links(file_path, output_path):
#     with open(file_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
#         for line in infile:
#             line = line.strip()  # 去除多余空白字符
#             if not line:  # 跳过空行
#                 continue
#             try:
#                 data = json.loads(line)
#                 if data.get("wikipediaLink", "").startswith("https://zh.wikipedia.org/"):
#                     outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
#             except json.JSONDecodeError:
#                 print(f"JSON 解码错误，跳过此行: {line}")

# # 遍历last文件夹中的所有jsonl文件
# for filename in os.listdir(input_folder):
#     if filename.endswith(".jsonl"):
#         input_path = os.path.join(input_folder, filename)
#         output_path = os.path.join(output_folder, filename)
#         filter_wikipedia_links(input_path, output_path)

# print("筛选完成，结果已保存到文件夹 'last2' 中。")


import os
import json
from bs4 import BeautifulSoup
import re

input_folder = "wikidata6"
output_folder = "wikidata7"
os.makedirs(output_folder, exist_ok=True)

def extract_first_sentence(html_content):
    # 解析 HTML 内容
    soup = BeautifulSoup(html_content, "html.parser")
    
    # 查找页面主体的第一个 <p> 元素
    first_paragraph = soup.find("p")
    
    if first_paragraph:
        # 提取第一个 <p> 中的文本内容
        text = first_paragraph.get_text()
        
        # 使用正则表达式匹配第一句话
        first_sentence = re.split(r'[。？！\n]', text)[0]
        
        return first_sentence.strip()
    return None

def process_jsonl_file(file_path, output_path):
    with open(file_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                
                # 如果包含 content 字段，则提取第一句话
                if "content" in data and data["content"]:
                    first_sentence = extract_first_sentence(data["content"])
                    data["define"] = first_sentence if first_sentence else "N/A"
                else:
                    data["define"] = "N/A"  # 如果没有content字段，define为空
                
                outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
            except json.JSONDecodeError:
                print(f"JSON 解码错误，跳过此行: {line}")

# 遍历 last2 文件夹中的所有 jsonl 文件
for filename in os.listdir(input_folder):
    if filename.endswith(".jsonl"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        process_jsonl_file(input_path, output_path)

print("处理完成，结果已保存到文件夹 'last2_processed' 中。")
