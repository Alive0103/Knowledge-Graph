import json
import os
import time
from SPARQLWrapper import SPARQLWrapper, JSON
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置SPARQL endpoint
sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.setReturnFormat(JSON)
sparql.addCustomHttpHeader("User-Agent",  "curl/8.9.1")

# 根据实体ID查询Wikipedia链接
def query_wikipedia_link(entity_id, retries=3):
    query = f"""
        SELECT ?entity ?entityLabel ?wikipediaLink ?englishLink WHERE {{
            BIND(wd:{entity_id} AS ?entity) .
            OPTIONAL {{
                ?entity rdfs:label ?entityLabel FILTER (lang(?entityLabel) = "en") .
            }}
            OPTIONAL {{
                ?article schema:about ?entity .
                ?article schema:inLanguage "zh" .
                FILTER (SUBSTR(str(?article), 1, 25) = "https://zh.wikipedia.org/")
                BIND(?article AS ?wikipediaLink)
            }}
            OPTIONAL {{
                ?article_en schema:about ?entity .
                ?article_en schema:inLanguage "en" .
                FILTER (SUBSTR(str(?article_en), 1, 25) = "https://en.wikipedia.org/")
                BIND(?article_en AS ?englishLink)
            }}
        }}
    """
    sparql.setQuery(query)

    # 执行查询并返回结果
    for attempt in range(retries):
        try:
            results = sparql.query().convert()
            return results
        except Exception as e:
            print(f"查询出错: {e} (尝试次数: {attempt + 1})")
            time.sleep(5)
            continue
    return None

# 处理单个实体数据
def process_entity(data):
    try:
        entity_value = data['item']['value']
        entity_id = entity_value.split('/')[-1]

        print(f"正在处理实体ID: {entity_id}")

        # 查询维基百科链接
        wiki_results = query_wikipedia_link(entity_id)
        if wiki_results and 'results' in wiki_results:
            for result in wiki_results['results']['bindings']:
                wikipedia_zh = result.get('wikipediaLink', {}).get('value', '')

                # 如果找不到中文维基百科链接，则尝试获取英文链接
                # if  wikipedia_zh:
                data['wikipediaLink_zh'] = wikipedia_zh
                data['wikipediaLink_en'] = result.get('englishLink', {}).get('value', '')

    except Exception as e:
        print(f"处理实体出错: {e}")
    return data

# 读取 JSONL 文件并处理每行数据
def process_jsonl_file(input_path, output_path, batch_size=100):
    results = []

    def process_batch(batch_data):
        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = [executor.submit(process_entity, data) for data in batch_data]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"任务出错: {e}")

    with open(input_path, 'r', encoding='utf-8') as f:
        batch_data = []
        for line in f:
            try:
                data = json.loads(line.strip())
                batch_data.append(data)
                if len(batch_data) == batch_size:
                    process_batch(batch_data)  # 批量处理
                    batch_data = []
            except json.JSONDecodeError:
                print(f"解析出错: {line}")

        if batch_data:  # 处理最后剩下的数据
            process_batch(batch_data)

    # 写入结果
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    input_folder = "data"
    output_folder = "data2"
    os.makedirs(output_folder, exist_ok=True)

    for input_file in os.listdir(input_folder):
        if input_file.endswith('.jsonl'):
            input_path = os.path.join(input_folder, input_file)
            output_path = os.path.join(output_folder, input_file)
            print(f"处理文件: {input_file}")
            process_jsonl_file(input_path, output_path)

    print("所有文件处理完成。")
