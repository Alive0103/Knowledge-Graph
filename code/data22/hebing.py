import json
import os

def merge_and_deduplicate_jsonl_files(output_file):
    """
    合并当前目录下的所有JSONL文件并重去，然后将结果写入输出文件。
    :param output_file: 输出文件路径
    """
    # 获取当前目录
    current_dir = os.getcwd()
    print(f"当前目录: {current_dir}")

    # 用于存储唯一的数据
    unique_data = set()

    # 遍历当前目录中的所有文件
    for file_name in os.listdir(current_dir):
        if file_name.endswith('.jsonl'):
            file_path = os.path.join(current_dir, file_name)
            print(f"正在处理文件: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # 将每行数据转换为JSON对象
                    try:
                        data = json.loads(line)
                        # 将JSON对象转换为字符串并添加到集合中
                        unique_data.add(json.dumps(data, sort_keys=True))
                    except json.JSONDecodeError as e:
                        print(f"解析JSON时出错: {e}")

    # 将去重后的数据写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for data_str in unique_data:
            f.write(data_str + '\n')

    # 输出去重后的数据条数
    print(f"去重后的数据条数: {len(unique_data)}")


# 示例用法
output_file_path = 'data1.jsonl'  # 输出文件名（会在当前目录下生成）

merge_and_deduplicate_jsonl_files(output_file_path)