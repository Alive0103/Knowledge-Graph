import json
import requests
from opencc import OpenCC
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import unquote
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from tqdm import tqdm
import os

# 繁体转简体
converter = OpenCC('t2s')

# 维基百科 API 前缀
WIKI_ZH_PREFIX = "https://zh.wikipedia.org/wiki/"
WIKI_EN_PREFIX = "https://en.wikipedia.org/wiki/"

# 代理（如果需要）
PROXIES = {"http": "127.0.0.1:7890", "https": "127.0.0.1:7890"}

# 过滤 Wikipedia 特殊页面和文件页面
def is_special_wikipedia_page(url):
    """ 过滤 Wikipedia 特殊页面 """
    return "/Special:" in url or "/Template:" in url or "/Help:" in url

def is_image_wikipedia_page(url):
    """ 过滤 Wikipedia 纯图片页面 (File:) """
    return "/File:" in url

# 下载 Wikipedia 页面 HTML
def get_wikipedia_page_content(wikipedia_link, retries=3):
    if is_special_wikipedia_page(wikipedia_link):
        print(f"跳过特殊页面: {wikipedia_link}")
        return None, wikipedia_link, []
    
    if is_image_wikipedia_page(wikipedia_link):
        print(f"跳过图片页面: {wikipedia_link}")
        return None, wikipedia_link, []

    if wikipedia_link.startswith(WIKI_ZH_PREFIX):
        title = wikipedia_link[len(WIKI_ZH_PREFIX):]
        api_url = "https://zh.wikipedia.org/w/api.php"
        is_zh = True
    elif wikipedia_link.startswith(WIKI_EN_PREFIX):
        title = wikipedia_link[len(WIKI_EN_PREFIX):]
        api_url = "https://en.wikipedia.org/w/api.php"
        is_zh = False
    else:
        print(f"不支持的维基百科链接格式: {wikipedia_link}")
        return None, wikipedia_link, []

    title = unquote(title)

    params = {
        'action': 'parse',
        'page': title,
        'format': 'json',
        'prop': 'text',
        'utf8': 1,
        'disablelimitreport': True,
        'disableeditsection': True
    }

    session = requests.Session()
    retries = Retry(total=retries, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    try:
        response = session.get(api_url, params=params, proxies=PROXIES, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'parse' in data and 'text' in data['parse']:
            content = data['parse']['text']['*']
            if is_zh:
                content = converter.convert(content)

            # 提取页面中的图片链接
            soup = BeautifulSoup(content, 'html.parser')
            images = [f"https:{img['src']}" for img in soup.find_all('img') if 'src' in img.attrs and img['src'].startswith('//')]

            return content, wikipedia_link, images
        else:
            print(f"无法解析页面内容: {wikipedia_link}")
            return None, wikipedia_link, []
    except requests.RequestException as e:
        print(f"请求出错: {wikipedia_link}, 错误: {e}")
        print("请检查链接的合法性，或者稍后重试。")
        return None, wikipedia_link, []

# 处理 `redirect_link` 字段，抓取页面 HTML 并存入文件
def process_redirect_links(input_file, output_file):
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        results = []
        with ThreadPoolExecutor(max_workers=64) as executor:  # 64线程
            future_to_link = {}

            for item in data:
                redirect_links = item.get("redirect_link", [])
                if not isinstance(redirect_links, list):  # 确保是列表
                    redirect_links = [redirect_links]

                for link in redirect_links:
                    if is_image_wikipedia_page(link):
                        print(f"过滤掉图片页面: {link}")
                        continue
                    future = executor.submit(get_wikipedia_page_content, link)
                    future_to_link[future] = link

            total_tasks = sum(1 for item in data for link in item.get("redirect_link", []) if not is_special_wikipedia_page(link) and not is_image_wikipedia_page(link))
            for future in tqdm(as_completed(future_to_link), total=total_tasks, desc=f"抓取维基百科页面 ({input_file})"):
                content, link, images = future.result()
                if content:
                    results.append({"link": link, "content": content, "images_link": images})

        with open(output_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"已保存 {len(results)} 条数据到 {output_file}")
    except Exception as e:
        print(f"处理文件 {input_file} 时出错: {e}")

if __name__ == "__main__":
    # 获取当前目录下所有以 zh 开头的 jsonl 文件
    files = [f for f in os.listdir(".") if f.startswith("zh") and f.endswith(".jsonl")]

    for file in files:
        input_file = file
        output_file = f"{file.rsplit('.', 1)[0]}_relink_content.jsonl"  # 原文件名 + _relink_content.jsonl
        process_redirect_links(input_file, output_file)



#--------------------------------------------------------------------------------------------------------------------
#提取链接并去重
# import os
# import json
# import re

# def extract_and_deduplicate_links():
#     # 获取当前目录下的所有 .jsonl 文件
#     jsonl_files = [f for f in os.listdir('.') if f.endswith('.jsonl')]
    
#     # 用于存储链接的集合（自动去重）
#     links_set = set()
    
#     # 遍历每个 .jsonl 文件
#     for file in jsonl_files:
#         with open(file, 'r', encoding='utf-8') as f:
#             for line in f:
#                 try:
#                     # 解析每一行的 JSON 数据
#                     data = json.loads(line)
#                     # 提取 redirect_link 字段
#                     links = data.get('redirect_link', [])
#                     if links:
#                         for link in links:
#                             # 使用正则表达式提取 URL
#                             match = re.search(r'https?://[^\s]+', link)
#                             if match:
#                                 url = match.group(0)
#                                 links_set.add(url)
#                 except json.JSONDecodeError:
#                     print(f"Warning: Unable to parse line in file {file}")
    
#     # 将去重后的链接写入新的 .jsonl 文件
#     output_file = 'deduplicated_links.jsonl'
#     with open(output_file, 'w', encoding='utf-8') as f:
#         for link in links_set:
#             f.write(json.dumps({'redirect_link': link}, ensure_ascii=False) + '\n')
    
#     # 显示链接条数
#     print(f"Total unique links: {len(links_set)}")
#     print(f"Links have been saved to {output_file}")

# # 调用函数
# extract_and_deduplicate_links()

#----------------------------------------------------------------------------
#以第一段代码格式存放
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