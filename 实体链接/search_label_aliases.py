import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
from tqdm import tqdm
from 实体链接.es_client import es

# 模型加载（可选，如果不需要向量功能可以注释掉）
# 使用原始字符串（r"..."）或正斜杠来避免转义问题
model_name = r'D:\work\毕设\知识图谱\Knowledge-Graph\实体链接\model\chinese-roberta-wwm-ext'
# 或者使用正斜杠：'D:/work/毕设/知识图谱/Knowledge-Graph/实体链接/model/chinese-roberta-wwm-ext'
model = None
tokenizer = None
try:
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()
    print("✓ 模型加载成功")
except Exception as e:
    print(f"警告: 模型加载失败 ({e})，将跳过向量生成功能")

def generate_vector(text):
    if text and model is not None and tokenizer is not None:
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
    # 尝试多个索引名称
    index_names = ["data2", "data1"]
    response = None
    for index_name in index_names:
        try:
            if es.indices.exists(index=index_name):
                response = es.search(index=index_name, body=search_query)
                break
        except:
            continue
    
    if response is None:
        raise Exception(f"未找到可用的索引，尝试过的索引: {index_names}")
    hits = response["hits"]["hits"]
    # 返回完整结果（包含label和link），并去重
    results = []
    seen_links = set()
    for hit in hits:
        source = hit["_source"]
        link = source.get("link", "")
        label = source.get("label", "")
        # 去重：只保留第一次出现的链接
        if link and link not in seen_links:
            results.append({
                "label": label,
                "link": link,
                "score": hit.get("_score", 0)
            })
            seen_links.add(link)
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
            # 处理结果可能是字典或字符串的情况
            if isinstance(result, dict):
                result_link = result.get("link", "")
            else:
                result_link = result
            if correct_link in result_link:
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

def test_search():
    """测试搜索功能（不需要评测文件）"""
    print("=" * 50)
    print("测试搜索功能")
    print("=" * 50)
    
    test_queries = ["AK47", "F-16", "步枪", "驱逐舰", "战斗机"]
    
    for query in test_queries:
        try:
            results = hybrid_search(query, top_k=5)
            print(f"\n查询: '{query}'")
            print(f"找到 {len(results)} 个结果:")
            for i, result in enumerate(results[:5], 1):
                if isinstance(result, dict):
                    label = result.get("label", "未知")
                    link = result.get("link", "")
                    score = result.get("score", 0)
                    print(f"  {i}. {label} (得分: {score:.2f})")
                    print(f"     链接: {link}")
                else:
                    print(f"  {i}. {result}")
        except Exception as e:
            print(f"\n查询 '{query}' 失败: {e}")
            import traceback
            traceback.print_exc()

def main():
    import os
    
    file_path = "find.xlsx"
    
    # 如果评测文件不存在，运行测试搜索
    if not os.path.exists(file_path):
        print(f"未找到评测文件: {file_path}")
        print("运行测试搜索模式...\n")
        test_search()
        return
    
    try:
        queries, correct_links = read_excel(file_path)
        print(f"读取了 {len(queries)} 个查询")
        mrr, hit_at_1, hit_at_5, hit_at_10 = calculate_metrics(queries, correct_links)
        print(f"\n{'='*50}")
        print("评测结果:")
        print(f"{'='*50}")
        print(f"MRR: {mrr:.4f}")
        print(f"Hit@1: {hit_at_1:.4f}")
        print(f"Hit@5: {hit_at_5:.4f}")
        print(f"Hit@10: {hit_at_10:.4f}")
    except Exception as e:
        print(f"评测失败: {e}")
        print("\n运行测试搜索模式...\n")
        test_search()

if __name__ == "__main__":
    main()