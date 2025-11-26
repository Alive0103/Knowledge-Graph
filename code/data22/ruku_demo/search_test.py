import json
import torch
from transformers import BertTokenizer, BertModel
from elasticsearch import Elasticsearch
import pandas as pd
from tqdm import tqdm  # 导入 tqdm 库

# 初始化 Elasticsearch 客户端
es = Elasticsearch(["http://localhost:9200"])

# 初始化 BERT 模型和分词器
model_name = 'D:/model/chinese-roberta-wwm-ext-large'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()

# 生成文本向量的函数
def generate_vector(text):
    if text:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        vector = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        return vector.tolist()
    return None

# 混合搜索函数
def hybrid_search(query_text, top_k=10):
    query_vector = generate_vector(query_text)
    search_query = {
        "query": {
            "bool": {
                "should": [
                    {
                        "match": {
                            "label": {
                                "query": query_text,
                                "boost": 2.0
                            }
                        }
                    },
                    {
                        "match": {
                            "aliases_zh": {
                                "query": query_text,
                                "boost": 2.0
                            }
                        }
                    }
                ]
            }
        },
        "knn": [
            {
                "field": "descriptions_zh_vector",
                "query_vector": query_vector,
                "k": 5,
                "num_candidates": 10,
                "boost": 1.0
            }
        ],
        "size": top_k
    }
    response = es.search(index="data1", body=search_query)
    hits = response["hits"]["hits"]
    results = [hit["_source"]["link"] for hit in hits]
    return results

# 读取 Excel 文件
def read_excel(file_path):
    df = pd.read_excel(file_path, header=None)
    queries = df[0].tolist()
    # correct_links = [link.split("：")[1] for link in df[1].tolist()]
    correct_links = df[1].tolist()
    return queries, correct_links

# 计算评估指标
def calculate_metrics(queries, correct_links):
    mrr = 0
    hit_at_1 = 0
    hit_at_5 = 0
    hit_at_10 = 0

    for query, correct_link in tqdm(zip(queries, correct_links), total=len(queries), desc="Processing queries"):
        results = hybrid_search(query)
        rank = None

        for i, result in enumerate(results):
            if correct_link in result:
                rank = i + 1
                break

        if rank is not None:
            mrr += 1 / rank
            hit_at_1 += 1 if rank <= 1 else 0
            hit_at_5 += 1 if rank <= 5 else 0
            hit_at_10 += 1 if rank <= 10 else 0

    mrr /= len(queries)
    hit_at_1 /= len(queries)
    hit_at_5 /= len(queries)
    hit_at_10 /= len(queries)

    return mrr, hit_at_1, hit_at_5, hit_at_10

# 主函数
def main():
    file_path = "find.xlsx"
    # file_path = "linked_entities.xlsx"
    queries, correct_links = read_excel(file_path)
    mrr, hit_at_1, hit_at_5, hit_at_10 = calculate_metrics(queries, correct_links)
    print(f"MRR: {mrr:.4f}")
    print(f"Hit@1: {hit_at_1:.4f}")
    print(f"Hit@5: {hit_at_5:.4f}")
    print(f"Hit@10: {hit_at_10:.4f}")

if __name__ == "__main__":
    main()