import os
import json
from elasticsearch import Elasticsearch
from itertools import chain
from opencc import OpenCC
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import BertTokenizer, BertModel
import torch

cc = OpenCC('t2s')

es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
index_name = 'wikidata_military_model'

model_name = './model/chinese-roberta-wwm-ext-large'  
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()

def embeddings(text, max_length=300):
    encoded_dict = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    last_hidden_state = outputs.last_hidden_state
    cls_embedding = last_hidden_state[:, 0, :].numpy().tolist()

    return cls_embedding

def process_single_item(item):
    new_data = {}
    try:
        entity_label = item.get('itemLabel', {}).get('value', '')  
        if any('\u4e00' <= char <= '\u9fff' for char in entity_label):  
            new_data['label'] = cc.convert(entity_label)  
        else:
            new_data['label'] = entity_label

        new_data['aliases_en'] = item.get('en_aliases', [])
        new_data['aliases_zh'] = [cc.convert(alias) for alias in item.get('zh_aliases', [])]
        new_data['content'] = cc.convert(item.get('content', ''))
        new_data['descriptions_zh'] = cc.convert(item.get('define', ''))   #从content提取的第一段作为描述
        new_data['descriptions_en'] = item.get('description_en', '')

        if new_data['descriptions_zh']:
            vector = embeddings(new_data['descriptions_zh'])
            if vector is not None:
                if isinstance(vector, list) and len(vector) == 1 and isinstance(vector[0], list):
                    vector = vector[0] 
                if len(vector) == 1024:  
                    new_data['descriptions_zh_vector'] = vector
                else:
                    new_data['descriptions_zh_vector'] = None  
                    print(f"警告: 实体 {entity_label} 的向量维度不正确，当前维度: {len(vector)}")
            else:
                new_data['descriptions_zh_vector'] = None  

        else:
            new_data['descriptions_zh_vector'] = None  


    except Exception as e:
        print(f"处理实体 {entity_label} 时出错: {e}")

    return new_data

def insert_into_es(data):
    
    if data:
        document = {
            'label': data.get('label', ''),
            'aliases_en': data.get('aliases_en', []),
            'aliases_zh': data.get('aliases_zh', []),
            'content': data.get('content', ''),
            'descriptions_zh': data.get('descriptions_zh', ''),
            'descriptions_en': data.get('descriptions_en', ''),
            'descriptions_zh_vector': data.get('descriptions_zh_vector','')
        }

        try:
            es.index(index=index_name, body=document)
        except Exception as e:
            print(f"插入 Elasticsearch 时出错: {e}")

def process_jsonl_file(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for line in f:
                line = line.strip()
                if not line:
                    continue  
                try:
                    item = json.loads(line)
                    future = executor.submit(process_single_item, item)
                    futures.append(future)
                except json.JSONDecodeError as e:
                    print(f"JSON 解析错误，跳过此行：{line}，错误详情：{e}")

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:  
                        insert_into_es(result)
                except Exception as e:
                    print(f"处理时出错: {e}")

def process_jsonl_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.jsonl'):
            input_file = os.path.join(folder_path, filename)
            print(f"处理文件: {input_file}")
            process_jsonl_file(input_file)
    print("所有文件处理完成。")

if __name__ == "__main__":
    folder_path = 'last2_processed'  
    print(f"处理文件夹: {folder_path}")
    process_jsonl_folder(folder_path)
