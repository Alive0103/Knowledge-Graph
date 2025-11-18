import json
import torch
from transformers import BertTokenizer, BertModel
from elasticsearch import Elasticsearch
import pandas as pd
from tqdm import tqdm
from zhipuai import ZhipuAI
import concurrent.futures
from tqdm import tqdm

es = Elasticsearch(["http://localhost:9200"])

model_name = 'D:/model/chinese-roberta-wwm-ext-large'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()

client = ZhipuAI(api_key="dab23a7b3db0459dbeb2a1e1941721a3.qbD9kfVvLcHfFrtc")  

def generate_vector(text):
    if text:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        vector = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        return vector.tolist()
    return None

def hybrid_search(query_text, top_k=20):
    # try:
    #     response_content = get_alias_and_definition(query_text)
    #     input_definition = response_content.split("定义：")[1].strip()
    #     query_vector = generate_vector(input_definition)
    #     use_llm = True  
    # except (ValueError, IndexError) as e:
    #     print(f"Failed to get alias and definition from LLM: {e}. Falling back to original query text.")
    #     query_vector = generate_vector(query_text)
    #     use_llm = False  

    search_query = {
        "query": {
            "bool": {
                "should": [
                    {
                        "match": {
                            "label": {
                                "query": query_text,
                                "boost": 1.0
                            }
                        }
                    },
                    {
                        "match": {
                            "aliases_zh": {
                                "query": query_text,
                                "boost": 1.0
                            }
                        }
                    },
                ]
            }
        },
        # "knn": [
        #     {
        #         "field": "descriptions_zh_vector",
        #         "query_vector": query_vector,
        #         "k": 10,
        #         "num_candidates": 20,
        #         "boost": 1.0
        #     }
        # ],
        "size": top_k
    }
    response = es.search(index="data1", body=search_query)
    hits = response["hits"]["hits"]
    # results = [hit["_source"] for hit in hits]
    results = [
        {
            "label": hit["_source"].get("label", ""),
            "aliases_zh": hit["_source"].get("aliases_zh", []),
            "aliases_en": hit["_source"].get("aliases_en", []),
            "descriptions_zh": hit["_source"].get("descriptions_zh", ""),
            "link": hit["_source"].get("link", "")
        } 
        for hit in hits
    ]
    return results

def get_alias_and_definition(mention):
    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=[
            {
                "role": "user",
                "content": (
                    f"你现在是军事领域专家，需要参照以下例子给出提及对应的别名和定义。\n"
                    f"例子：提及：Steyr HS .50、别名：斯泰尔HS .50狙击步枪、定义：斯泰尔HS .50（Steyr HS.50）是由奥地利斯泰尔-曼利夏公司研制的一款手动枪机式反器材狙击步枪。\n\n"
                    f"输入提及：{mention}\n\n"
                    f"请按照标签：{mention}、中文别名：、英文别名：、定义：的格式直接返回所需内容，不要解释或附加内容。"
                )
            }
        ],
    )
    response_content = response.choices[0].message.content.strip()
    
    if not response_content:
        raise ValueError(f"No response content for mention '{mention}'")
    
    return response_content

def generate_prompt_and_sort(mention, results):
    input_label = mention
    response_content = ""
    try:
        response_content = get_alias_and_definition(mention)
        input_aliases_zh = response_content.split("中文别名：")[1].split("英文别名")[0].strip()
        input_aliases_en = response_content.split("英文别名：")[1].split("定义")[0].strip()
        input_definition = response_content.split("定义：")[1].strip()
    except (ValueError, IndexError, Exception) as e:
        print(f"LLM failed to generate valid response for mention '{mention}'. Error: {e}")
        return [result['link'] for result in results]  

    options = []
    original_links = []  
    for idx, result in enumerate(results, start=1):
        option = (
            f"选项{idx}：\n"
            f"label: {result['label']}\n"
            f"aliases_zh: {', '.join(result['aliases_zh'])}\n"
            f"aliases_en: {', '.join(result['aliases_en'])}\n"
            f"descriptions_zh: {result['descriptions_zh']}\n"
            f"link: {result['link']}\n"
        )
        options.append(option)
        original_links.append(result['link'])  

    prompt = (
        f"现在你是军事领域专家，需要根据输入信息与选项列表的候选的匹配度进行从高到低排序\n"
        f"输入标签名：{input_label}\n"
        f"输入中文别名：{input_aliases_zh}\n"
        f"输入英文别名：{input_aliases_en}\n"
        f"输入定义：{input_definition}\n\n"
        f"选项列表：\n"
        f"{''.join(options)}\n\n"
        f"请根据输入信息与选项的匹配度，从高到低严格返回所有候选的link值，确保返回的link值是原始选项列表中的link值的排序，不能有缺失或重复，不要解释或附加内容。"
    )

    try:
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": prompt}]
        )
        sorted_links = response.choices[0].message.content.strip().split("\n")
        sorted_links = ensure_links_match(sorted_links, original_links)
        return sorted_links
    except Exception as e:
        print(f"LLM failed to sort links for mention '{mention}'. Error: {e}")
        return original_links  
    
def ensure_links_match(sorted_links, original_links):
    """
    确保排序后的链接与原始链接一致，替换不匹配的链接。
    """
    sorted_links_set = set(sorted_links)
    original_links_set = set(original_links)

    if sorted_links_set != original_links_set:
        sorted_links = [link for link in original_links if link in sorted_links_set]
        sorted_links.extend([link for link in original_links if link not in sorted_links_set])

    return sorted_links

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
    total_processed = 0

    def process_query(query, correct_link):
        try:
            results = hybrid_search(query)
            sorted_links = generate_prompt_and_sort(query, results)
            rank = None

            for i, link in enumerate(sorted_links):
                if correct_link in link:
                    rank = i + 1
                    break

            if rank is not None:
                return 1 / rank, 1 if rank <= 1 else 0, 1 if rank <= 5 else 0, 1 if rank <= 10 else 0
            return 0, 0, 0, 0
        except Exception as e:
            print(f"Error processing query '{query}': {e}")
            return 0, 0, 0, 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(process_query, query, correct_link) for query, correct_link in zip(queries, correct_links)]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(queries), desc="Processing queries"):
            result = future.result()
            mrr += result[0]
            hit_at_1 += result[1]
            hit_at_5 += result[2]
            hit_at_10 += result[3]
            total_processed += 1

    if total_processed > 0:
        mrr /= total_processed
        hit_at_1 /= total_processed
        hit_at_5 /= total_processed
        hit_at_10 /= total_processed
    else:
        print("No queries were processed successfully.")
        return 0, 0, 0, 0

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