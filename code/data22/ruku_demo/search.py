# from elasticsearch import Elasticsearch
# import json
# import torch
# from transformers import BertTokenizer, BertModel
# from zhipuai import ZhipuAI

# # 初始化 Elasticsearch 客户端
# es = Elasticsearch(["http://localhost:9200"])

# model_name = 'D:/model/chinese-roberta-wwm-ext-large'
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertModel.from_pretrained(model_name)
# model.eval()

# # 初始化智谱AI客户端
# client = ZhipuAI(api_key="dab23a7b3db0459dbeb2a1e1941721a3.qbD9kfVvLcHfFrtc")

# # 生成文本向量的函数
# def generate_vector(text):
#     if text:
#         inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
#         with torch.no_grad():
#             outputs = model(**inputs)
#         vector = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
#         return vector.tolist()
#     return None

# # 混合搜索函数
# def hybrid_search(query_text, top_k=10):
#     query_vector = generate_vector(query_text)

#     search_query = {
#         "query": {
#             "bool": {
#                 "should": [
#                     {
#                         "match": {
#                             "label": {
#                                 "query": query_text,
#                                 "boost": 2.0  
#                             }
#                         }
#                     },
#                     {
#                         "match": {
#                             "aliases_zh": {
#                                 "query": query_text,
#                                 "boost": 2.0  
#                             }
#                         }
#                     }
#                 ]
#             }
#         },
#         "knn": [
#             {
#                 "field": "descriptions_zh_vector",
#                 "query_vector": query_vector,
#                 "k": 5,
#                 "num_candidates": 10,
#                 "boost": 1.0  
#             }
#         ],
#         "size": top_k
#     }

#     response = es.search(index="data1", body=search_query)
#     hits = response["hits"]["hits"]

#     # results = [hit["_source"]["label"] for hit in hits]
#     # return results

#     results = []
#     for hit in hits:
#         source = hit["_source"]
#         result = {
#             "label": source.get("label", ""),  
#             "aliases_zh": source.get("aliases_zh", ""),  
#             "aliases_en": source.get("aliases_en", ""),  
#             "descriptions_zh": source.get("descriptions_zh", ""),  
#             "link": source.get("link", "")  
#         }
#         results.append(result)

#     return results

# # 提及信息补充函数
# def get_alias_and_definition(mention):
#     response = client.chat.completions.create(
#         model="glm-4-flash", 
#         messages=[
#             {   
#                 "role": "user", 
#                 "content": (
#                     f"你现在是军事领域专家，需要参照以下例子给出提及对应的别名和定义。"
#                     f"例子：提及：Steyr HS .50、别名：斯泰尔HS .50狙击步枪、定义：斯泰尔HS .50（Steyr HS.50）是由奥地利斯泰尔-曼利夏公司研制的一款手动枪机式反器材狙击步枪提及。\n\n"
#                     f"输入提及：{mention}\n\n"
#                     f"请按照标签：{mention}、中文别名：、英文别名：、定义：的格式直接返回所需内容，不要解释或附加内容。"
#                 )
#             }
#         ],
#     )
#     return "".join(response.choices[0].message.content.split('\n'))


# # 示例调用
# mention = "阿利·伯克级驱逐舰"
# results = hybrid_search(mention)
# print(json.dumps(results, ensure_ascii=False, indent=4))


# result = get_alias_and_definition(mention)
# print(result)


from elasticsearch import Elasticsearch
import json
import torch
from transformers import BertTokenizer, BertModel
from zhipuai import ZhipuAI

# 初始化 Elasticsearch 客户端
es = Elasticsearch(["http://localhost:9200"])

# 初始化 BERT 模型和分词器
model_name = 'D:/model/chinese-roberta-wwm-ext-large'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()

# 初始化智谱AI客户端
client = ZhipuAI(api_key="dab23a7b3db0459dbeb2a1e1941721a3.qbD9kfVvLcHfFrtc")  

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

    results = []
    for hit in hits:
        source = hit["_source"]
        result = {
            "label": source.get("label", ""),
            "aliases_zh": source.get("aliases_zh", []),
            "aliases_en": source.get("aliases_en", []),
            "descriptions_zh": source.get("descriptions_zh", ""),
            "link": source.get("link", "")
        }
        results.append(result)

    return results

# 提及信息补充函数
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
    return response.choices[0].message.content.strip()

# 排序函数
def generate_prompt_and_sort(mention, results):

    input_label = mention
    input_aliases_zh = get_alias_and_definition(mention).split("中文别名：")[1].split("英文别名")[0].strip()
    input_aliases_en = get_alias_and_definition(mention).split("英文别名：")[1].split("定义")[0].strip()
    input_definition = get_alias_and_definition(mention).split("定义：")[1].strip()

    options = []
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

    prompt = (
        f"现在你是军事领域专家，需要按选项与输入的匹配度对下面十个选项进行排序。\n"
        f"输入标签名：{input_label}\n"
        f"输入中文别名：{input_aliases_zh}\n"
        f"输入英文别名：{input_aliases_en}\n"
        f"输入定义：{input_definition}\n\n"
        f"选项列表：\n"
        f"{''.join(options)}\n\n"
        f"请根据输入提及与选项的匹配度，从高到低返回十个link值，不要解释或附加内容。"
    )

    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    sorted_links = response.choices[0].message.content.strip().split("\n")
    return sorted_links

# 示例调用
mention = "阿利·伯克级驱逐舰"
results = hybrid_search(mention)
sorted_links = generate_prompt_and_sort(mention, results)

print("排序后的链接：")
print("\n".join(sorted_links))