import json
from elasticsearch import Elasticsearch, helpers

# Elasticsearch 配置
ES_HOST = "http://localhost:9200"  # 修改为你的 Elasticsearch 地址
es = Elasticsearch([ES_HOST])

# 索引名称
INDEX_RELINK = "data2_relink"
INDEX_IMAGE = "data2_image"
INDEX_MILITARY = "data2"

# 分批处理数据并导入到 Elasticsearch
def import_data_to_es(input_file, index_name, transform_func, batch_size=1000, request_timeout=60):
    """
    分批处理数据并导入到 Elasticsearch。
    :param input_file: 输入的 JSONL 文件路径
    :param index_name: Elasticsearch 索引名称
    :param transform_func: 数据转换函数
    :param batch_size: 每批处理的数据量，默认为 1000
    :param request_timeout: 请求超时时间，默认为 60 秒
    """
    actions = []  # 用于存储当前批次的数据
    total_imported = 0  # 总导入数据量

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            transformed_data = transform_func(data)
            if transformed_data:
                actions.append({
                    "_index": index_name,
                    "_source": transformed_data
                })

            # 当达到批次大小时，执行批量导入
            if len(actions) >= batch_size:
                helpers.bulk(es, actions, request_timeout=request_timeout)
                total_imported += len(actions)
                actions = []  # 清空当前批次数据

    # 处理剩余的数据
    if actions:
        helpers.bulk(es, actions, request_timeout=request_timeout)
        total_imported += len(actions)

    print(f"已成功导入 {total_imported} 条数据到 {index_name} 索引。")

# 数据转换函数
def transform_relink_data(data):
    """
    转换 relink 数据。
    """
    return {
        "relink_url": data.get("link"),
        "relink_data": data.get("content")
    }

def transform_image_data(data):
    """
    转换图片数据。
    """
    return {
        "image_url": data.get("link"),
        "image_data": data.get("content")
    }

def transform_military_data(data):
    """
    转换军事数据，检查向量字段是否有效。
    """
    zh_vector = data.get("zh_descriptions_vector")
    en_vector = data.get("en_descriptions_vector")

    # # 检查向量字段是否为有效的列表且不为空
    # if not isinstance(zh_vector, list):
    #     print(f"警告：'zh_description_vector' 字段为空或格式不正确，跳过该条数据")
    #     return None

    # 转换后的数据
    transformed_data = {
        "label": data.get("label"),
        "aliases_en": data.get("en_aliases"),
        "aliases_zh": data.get("zh_aliases"),
        "descriptions_en": data.get("en_description"),
        "descriptions_zh": data.get("zh_description"),
        "content": data.get("content"),
        "vector_descriptions_zh": zh_vector,
        "vector_descriptions_en": en_vector
    }
    return transformed_data

# 主函数
if __name__ == "__main__":
    # if es.indices.exists(index=INDEX_MILITARY):
    #     es.delete_by_query(index=INDEX_MILITARY, body={"query": {"match_all": {}}}, refresh=True)
    #     print(f"索引 {INDEX_MILITARY} 中的所有数据已被清空。")

    # 导入数据
    # import_data_to_es("data2_relink.jsonl", INDEX_RELINK, transform_relink_data)
    # import_data_to_es("data2_images.jsonl", INDEX_IMAGE, transform_image_data)
    import_data_to_es("data2.jsonl", INDEX_MILITARY, transform_military_data)


##查询索引中数据条数
# from elasticsearch import Elasticsearch

# # Elasticsearch 配置
# ES_HOST = "http://localhost:9200"  # 修改为你的 Elasticsearch 地址
# es = Elasticsearch([ES_HOST])

# # 索引名称
# INDEX_RELINK = "data1_relink"
# INDEX_IMAGE = "data1_image"
# INDEX_MILITARY = "data1"

# def get_document_count(index_name):
#     """
#     获取指定索引中的文档数量。
#     :param index_name: 索引名称
#     :return: 文档数量
#     """
#     try:
#         # 使用 count API 获取文档数量
#         response = es.count(index=index_name)
#         return response["count"]
#     except Exception as e:
#         print(f"获取索引 {index_name} 的文档数量时出错：{e}")
#         return None

# def main():
#     # 检查每个索引的文档数量
#     indices = [INDEX_RELINK, INDEX_IMAGE, INDEX_MILITARY]
#     for index in indices:
#         count = get_document_count(index)
#         if count is not None:
#             print(f"索引 {index} 中的文档数量：{count}")

# if __name__ == "__main__":
#     main()