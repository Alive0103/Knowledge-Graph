import json
from elasticsearch import helpers
from tqdm import tqdm
import time
from es_client import es

INDEX_RELINK = "data2_relink"
INDEX_IMAGE = "data2_image"
INDEX_MILITARY = "data2"

def import_data_to_es(input_file, index_name, transform_func, batch_size=100, request_timeout=120):
    """
    分批处理数据并导入到 Elasticsearch。
    :param input_file: 输入的 JSONL 文件路径
    :param index_name: Elasticsearch 索引名称
    :param transform_func: 数据转换函数
    :param batch_size: 每批处理的数据量，默认为 1000
    :param request_timeout: 请求超时时间，默认为 60 秒
    """
    import os
    
    # 获取文件总行数（用于进度条）
    print(f"正在读取数据文件: {input_file}")
    total_lines = 0
    with open(input_file, "r", encoding="utf-8") as f:
        for _ in f:
            total_lines += 1
    
    print(f"文件总行数: {total_lines}")
    print(f"开始导入数据，批次大小: {batch_size}...")
    
    actions = []  
    total_imported = 0
    failed_count = 0
    start_time = time.time()
    current_batch_size = 0  # 当前批次的数据大小（字节）

    with open(input_file, "r", encoding="utf-8") as f:
        # 使用tqdm显示进度
        progress_bar = tqdm(total=total_lines, desc="导入进度", unit="条", mininterval=0.5)
        
        for line_num, line in enumerate(f, 1):
            try:
                # 跳过空行
                if not line.strip():
                    progress_bar.update(1)
                    continue
                    
                data = json.loads(line.strip())
                transformed_data = transform_func(data)
                if transformed_data:
                    # 检查数据大小
                    import json as json_module
                    data_size = len(json_module.dumps(transformed_data, ensure_ascii=False))
                    
                    # 如果单条数据超过 1MB，跳过（避免超过限制）
                    if data_size > 1024 * 1024:
                        tqdm.write(f"警告: 第{line_num}行数据过大 ({data_size/1024:.1f}KB)，跳过")
                        failed_count += 1
                        progress_bar.update(1)
                        continue
                    
                    # 检查批次总大小（阿里云 ES Serverless 限制单次请求大小）
                    # 如果当前批次加上这条数据超过 5MB，先提交当前批次
                    MAX_BATCH_SIZE = 5 * 1024 * 1024  # 5MB
                    if current_batch_size + data_size > MAX_BATCH_SIZE and len(actions) > 0:
                        # 先提交当前批次
                        try:
                            tqdm.write(f"批次大小达到限制，提前提交 (共{len(actions)}条)...")
                            success, failed = helpers.bulk(
                                es.options(request_timeout=request_timeout), 
                                actions,
                                raise_on_error=False,
                                stats_only=False
                            )
                            if failed:
                                error_count = len(failed)
                                failed_count += error_count
                                total_imported += (len(actions) - error_count)
                            else:
                                total_imported += len(actions)
                            actions = []
                            current_batch_size = 0
                        except Exception as e:
                            tqdm.write(f"提前提交批次失败: {e}")
                            failed_count += len(actions)
                            actions = []
                            current_batch_size = 0
                    
                    actions.append({
                        "_index": index_name,
                        "_source": transformed_data
                    })
                    current_batch_size += data_size
                    # 每处理一条就更新进度（即使还没导入）
                    progress_bar.update(1)
            except json.JSONDecodeError as e:
                failed_count += 1
                progress_bar.update(1)
                if failed_count <= 5:  # 只显示前5个错误
                    tqdm.write(f"警告: 第{line_num}行JSON解析失败: {e}")
            except Exception as e:
                failed_count += 1
                progress_bar.update(1)
                if failed_count <= 5:
                    tqdm.write(f"警告: 第{line_num}行处理失败: {e}")

            # 当达到批次大小时，批量导入
            if len(actions) >= batch_size:
                try:
                    tqdm.write(f"正在导入批次 (共{len(actions)}条)...")
                    success, failed = helpers.bulk(
                        es.options(request_timeout=request_timeout), 
                        actions,
                        raise_on_error=False,  # 不抛出异常，返回详细结果
                        stats_only=False  # 返回详细统计信息
                    )
                    
                    if failed:
                        # 显示详细的错误信息
                        error_count = len(failed)
                        tqdm.write(f"⚠ 批次中有 {error_count} 条失败")
                        # 显示前3个错误的详细信息
                        for i, item in enumerate(failed[:3]):
                            error_info = item.get('index', {}).get('error', {})
                            error_type = error_info.get('type', 'unknown')
                            error_reason = error_info.get('reason', 'unknown')
                            tqdm.write(f"  错误 {i+1}: {error_type} - {error_reason[:100]}")
                        if error_count > 3:
                            tqdm.write(f"  ... 还有 {error_count - 3} 个错误")
                        failed_count += error_count
                        total_imported += (len(actions) - error_count)
                    else:
                        total_imported += len(actions)
                        tqdm.write(f"✓ 成功导入 {len(actions)} 条")
                    
                    actions = []
                    current_batch_size = 0
                except Exception as e:
                    tqdm.write(f"错误: 批量导入异常: {e}")
                    failed_count += len(actions)
                    actions = []
                    current_batch_size = 0
                    # 继续处理，不中断

        # 导入剩余的数据
        if actions:
            try:
                tqdm.write(f"正在导入最后一批 (共{len(actions)}条)...")
                success, failed = helpers.bulk(
                    es.options(request_timeout=request_timeout), 
                    actions,
                    raise_on_error=False,
                    stats_only=False
                )
                
                if failed:
                    error_count = len(failed)
                    tqdm.write(f"⚠ 最后一批中有 {error_count} 条失败")
                    for i, item in enumerate(failed[:3]):
                        error_info = item.get('index', {}).get('error', {})
                        error_type = error_info.get('type', 'unknown')
                        error_reason = error_info.get('reason', 'unknown')
                        tqdm.write(f"  错误 {i+1}: {error_type} - {error_reason[:100]}")
                    failed_count += error_count
                    total_imported += (len(actions) - error_count)
                else:
                    total_imported += len(actions)
                    tqdm.write(f"✓ 成功导入最后 {len(actions)} 条")
            except Exception as e:
                tqdm.write(f"错误: 最后一批导入异常: {e}")
                failed_count += len(actions)
        
        progress_bar.close()

    elapsed_time = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"导入完成！")
    print(f"成功导入: {total_imported} 条")
    print(f"失败/跳过: {failed_count} 条")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    print(f"平均速度: {total_imported/elapsed_time:.2f} 条/秒" if elapsed_time > 0 else "")
    print(f"{'='*50}")

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
    支持多种数据格式：
    - 格式1: label, wikipedia, en_aliases, zh_aliases, en_description, zh_description, content
    - 格式2: label, wikipediaLink, en_aliases, zh_aliases, en_description, zh_description, content
    """
    # 兼容不同的字段名
    label = data.get("label", "")
    link = data.get("wikipedia") or data.get("wikipediaLink", "")
    aliases_en = data.get("en_aliases") or data.get("aliases_en", [])
    aliases_zh = data.get("zh_aliases") or data.get("aliases_zh", [])
    descriptions_en = data.get("en_description") or data.get("descriptions_en", "")
    descriptions_zh = data.get("zh_description") or data.get("descriptions_zh", "")
    content = data.get("content", "")
    
    # 向量字段（可选，需要验证维度）
    zh_vector = data.get("zh_descriptions_vector") or data.get("descriptions_zh_vector")
    en_vector = data.get("en_descriptions_vector") or data.get("descriptions_en_vector")

    transformed_data = {
        "label": label,
        "link": link,
        "aliases_en": aliases_en if isinstance(aliases_en, list) else [],
        "aliases_zh": aliases_zh if isinstance(aliases_zh, list) else [],
        "descriptions_en": descriptions_en or "",
        "descriptions_zh": descriptions_zh or "",
        "content": content or "",
    }
    
    # 只有当向量存在、为列表且维度正确（1024）时才添加
    # 索引中定义的向量维度是 1024
    VECTOR_DIMS = 1024
    
    if zh_vector and isinstance(zh_vector, list) and len(zh_vector) == VECTOR_DIMS:
        transformed_data["descriptions_zh_vector"] = zh_vector
    elif zh_vector and isinstance(zh_vector, list) and len(zh_vector) != VECTOR_DIMS:
        # 向量维度不匹配，不添加（避免导入失败）
        pass
    
    if en_vector and isinstance(en_vector, list) and len(en_vector) == VECTOR_DIMS:
        transformed_data["descriptions_en_vector"] = en_vector
    elif en_vector and isinstance(en_vector, list) and len(en_vector) != VECTOR_DIMS:
        # 向量维度不匹配，不添加（避免导入失败）
        pass
    
    return transformed_data

if __name__ == "__main__":
    import os
    from pathlib import Path
    
    # 查找数据文件
    data_files = [
        "zh_wiki_v2.jsonl",
        "中英文维基-部分/zh_wiki_v2.jsonl",
        "../中英文维基-部分/zh_wiki_v2.jsonl",
        "data2.jsonl"
    ]
    
    data_file = None
    for file_path in data_files:
        if os.path.exists(file_path):
            data_file = file_path
            print(f"找到数据文件: {data_file}")
            break
    
    if not data_file:
        print("错误: 未找到数据文件")
        print("请确保以下文件之一存在:")
        for f in data_files:
            print(f"  - {f}")
        exit(1)
    
    # 如果索引已存在，可以选择清空
    # if es.indices.exists(index=INDEX_MILITARY):
    #     es.delete_by_query(index=INDEX_MILITARY, body={"query": {"match_all": {}}}, refresh=True)
    #     print(f"索引 {INDEX_MILITARY} 中的所有数据已被清空。")

    # import_data_to_es("data2_relink.jsonl", INDEX_RELINK, transform_relink_data)
    # import_data_to_es("data2_images.jsonl", INDEX_IMAGE, transform_image_data)
    import_data_to_es(data_file, INDEX_MILITARY, transform_military_data)


# from elasticsearch import Elasticsearch

# ES_HOST = "http://localhost:9200"  
# es = Elasticsearch([ES_HOST])

# INDEX_RELINK = "data1_relink"
# INDEX_IMAGE = "data1_image"
# INDEX_MILITARY = "data1"

# def get_document_count(index_name):
#     try:
#         # 使用 count API 获取文档数量
#         response = es.count(index=index_name)
#         return response["count"]
#     except Exception as e:
#         print(f"获取索引 {index_name} 的文档数量时出错：{e}")
#         return None

# def main():
#     indices = [INDEX_RELINK, INDEX_IMAGE, INDEX_MILITARY]
#     for index in indices:
#         count = get_document_count(index)
#         if count is not None:
#             print(f"索引 {index} 中的文档数量：{count}")

# if __name__ == "__main__":
#     main()