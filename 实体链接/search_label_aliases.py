import json
import torch
from transformers import BertTokenizer, BertModel
from elasticsearch import Elasticsearch
import pandas as pd
from tqdm import tqdm  

es = Elasticsearch(["http://localhost:9200"])

model_name = 'D:/model/chinese-roberta-wwm-ext-large'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()

def generate_vector(text):
    if text:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        vector = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        return vector.tolist()
    return None

def hybrid_search(query_text, top_k=10):
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
        "size": top_k
    }
    response = es.search(index="data1", body=search_query)
    hits = response["hits"]["hits"]
    results = [hit["_source"]["link"] for hit in hits]
    return results

def read_excel(file_path):
    df = pd.read_excel(file_path, header=None)
    queries = df[0].tolist()
    correct_links = df[1].tolist()
    return queries, correct_links

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

def main():
    file_path = "find.xlsx"
    queries, correct_links = read_excel(file_path)
    mrr, hit_at_1, hit_at_5, hit_at_10 = calculate_metrics(queries, correct_links)
    print(f"MRR: {mrr:.4f}")
    print(f"Hit@1: {hit_at_1:.4f}")
    print(f"Hit@5: {hit_at_5:.4f}")
    print(f"Hit@10: {hit_at_10:.4f}")

if __name__ == "__main__":
    main()