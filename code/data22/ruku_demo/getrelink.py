#对于第一层页面，获取第二层页面的html，得到第二层html和第二层的图片地址

import json
import requests
from opencc import OpenCC
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import unquote
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from tqdm import tqdm

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
            # 提取页面中的图片链接
            images = [f"https:{img['src']}" for img in soup.find_all('img') if 'src' in img.attrs and img['src'].startswith('//')]

            return content, wikipedia_link, images
        else:
            print(f"无法解析页面内容: {wikipedia_link}")
            return None, wikipedia_link, []
    except requests.RequestException as e:
        print(f"请求出错: {wikipedia_link}, 错误: {e}")
        return None, wikipedia_link, []

# 处理 `redirect_link` 字段，抓取页面 HTML 并存入 `demo_relink.jsonl`
def process_redirect_links(input_file="demo.jsonl", output_file="demo_relink.jsonl"):
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
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
        for future in tqdm(as_completed(future_to_link), total=total_tasks, desc="抓取维基百科页面"):

            content, link, images = future.result()
            if content:
                results.append({"link": link, "content": content, "images_link": images})

    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"已保存 {len(results)} 条数据到 {output_file}")

if __name__ == "__main__":
    process_redirect_links()



