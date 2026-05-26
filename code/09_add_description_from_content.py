import json
import os
from bs4 import BeautifulSoup

def extract_first_paragraph(content):
    try:
        soup = BeautifulSoup(content, 'html.parser')
        first_paragraph = soup.find('p') 
        if first_paragraph:
            return "".join(first_paragraph.text.strip().split("\n"))  
        else:
            return None
    except Exception as e:
        print(f"提取段落时出错: {e}")
        return None

def add_description_to_data3(data3_folder, data4_folder):
    wikicontent = {}
    for data4_file in os.listdir(data4_folder):
        if data4_file.endswith('.jsonl'):
            with open(os.path.join(data4_folder, data4_file), 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    wikicontent[data['link']] = data['content']

    for wikibase in os.listdir(data3_folder):
        if wikibase.endswith('.jsonl'):
            file_path = os.path.join(data3_folder, wikibase)
            results = []

            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    wikipedia_link_zh = data.get('wikipediaLink_zh')

                    if wikipedia_link_zh and wikipedia_link_zh in wikicontent:
                        content = wikicontent[wikipedia_link_zh]
                        description = extract_first_paragraph(content)
                        data['description'] = description if description else ""
                    else:
                        data['description'] = ""

                    results.append(data)

            with open(file_path, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

            print(f"文件 {wikibase} 处理完成并已更新。")


if __name__ == "__main__":
    data3_folder = "data3"     
    data4_folder = "data4"      

    add_description_to_data3(data3_folder, data4_folder)

    print("所有文件已更新完成。")
