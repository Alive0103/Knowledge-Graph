import json
import re

def process_wiki_links(html_content):
    """
    处理 HTML 内容，将 /wiki/ 开头的超链接替换为完整的维基百科链接。
    """
    def replace_wiki_link(match):
        link = match.group(1)
        if link.startswith("/wiki/"):
            # 补全为完整的维基百科链接
            full_link = f"https://en.wikipedia.org{link}"
            return f'<a href="{full_link}"{match.group(2)}>{match.group(3)}</a>'
        return match.group(0)

    # 替换 HTML 中的 <a> 标签
    processed_content = re.sub(r'<a href="([^"]+)"([^>]*)>([^<]+)</a>', replace_wiki_link, html_content)
    return processed_content

def process_jsonl_file(input_file, output_file):
    """
    读取 JSONL 文件，处理每条记录中的 content 字段，并保存到新的文件。
    """
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            record = json.loads(line)  # 解析 JSON 对象
            if "content" in record:
                # 处理 content 字段
                record["content"] = process_wiki_links(record["content"])
            # 写入新的文件
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

# 输入文件和输出文件路径
input_file = "new_data1.jsonl"
output_file = "data1.jsonl"

# 处理文件
process_jsonl_file(input_file, output_file)
print(f"处理完成，结果已保存到 {output_file}")

