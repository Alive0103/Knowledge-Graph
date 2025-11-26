# --------------------------------------------------------------------------------------------------------------------
# 提取链接并去重
import os
import json
import re

def extract_and_deduplicate_links():
    # 获取当前目录下的所有 .jsonl 文件
    jsonl_files = [f for f in os.listdir('.') if f.endswith('.jsonl')]
    
    # 用于存储链接的集合（自动去重）
    links_set = set()
    
    # 遍历每个 .jsonl 文件
    for file in jsonl_files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # 解析每一行的 JSON 数据
                    data = json.loads(line)
                    # 提取 redirect_link 字段
                    links = data.get('redirect_link', [])
                    if links:
                        for link in links:
                            # 使用正则表达式提取 URL
                            match = re.search(r'https?://[^\s]+', link)
                            if match:
                                url = match.group(0)
                                links_set.add(url)
                except json.JSONDecodeError:
                    print(f"Warning: Unable to parse line in file {file}")
    
    # 将去重后的链接写入新的 .jsonl 文件
    output_file = 'deduplicated_links.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for link in links_set:
            f.write(json.dumps({'redirect_link': link}, ensure_ascii=False) + '\n')
    
    # 显示链接条数
    print(f"Total unique links: {len(links_set)}")
    print(f"Links have been saved to {output_file}")

# 调用函数
extract_and_deduplicate_links()

# ----------------------------------------------------------------------------
# 以第一段代码格式存放
# import json

# def extract_links_from_jsonl(input_file, output_file):
#     try:
#         # 打开输入文件并读取数据
#         with open(input_file, "r", encoding="utf-8") as f:
#             lines = [json.loads(line) for line in f]

#         # 提取所有 redirect_link 的值
#         all_links = []
#         for line in lines:
#             redirect_link = line.get("redirect_link")
#             if redirect_link:
#                 all_links.append(redirect_link)

#         # 将所有链接存入输出文件
#         with open(output_file, "w", encoding="utf-8") as f:
#             json.dump({"redirect_link": all_links}, f, ensure_ascii=False, indent=2)

#         print(f"已提取 {len(all_links)} 条链接，保存到 {output_file}")
#     except json.JSONDecodeError as e:
#         print(f"JSON 解析错误: {e}")
#     except Exception as e:
#         print(f"处理文件 {input_file} 时出错: {e}")

# if __name__ == "__main__":
#     input_file = "demo.jsonl"  # 输入文件
#     output_file = "new.jsonl"  # 输出文件
#     extract_links_from_jsonl(input_file, output_file)

