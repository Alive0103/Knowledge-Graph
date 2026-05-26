import os
import json
import requests
import base64
from bs4 import BeautifulSoup
from urllib.parse import unquote, urljoin
from requests.adapters import HTTPAdapter, Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import opencc  # 导入opencc库

# 设置保存文件路径
HTML_OUTPUT_PATH = "data4/wiki_page.jsonl"
IMAGE_OUTPUT_PATH = "data4/wiki_image.jsonl"

ZH_WIKI_PREFIX = "https://zh.wikipedia.org/wiki/"
EN_WIKI_PREFIX = "https://en.wikipedia.org/wiki/"

# 使用队列来防止重复下载相同的页面
downloaded_pages = set()

# 初始化OpenCC转换器，配置为将繁体转换为简体
converter = opencc.OpenCC('t2s.json')  # 't2s.json' 表示繁体转简体

# 下载维基百科页面
def download_wikipedia_page(wikipedia_link, retries=3):
    if wikipedia_link.startswith(ZH_WIKI_PREFIX):
        title = wikipedia_link[len(ZH_WIKI_PREFIX):]
        api_url = "https://zh.wikipedia.org/w/api.php"
        is_zh = True  # 页面是中文维基
    elif wikipedia_link.startswith(EN_WIKI_PREFIX):
        title = wikipedia_link[len(EN_WIKI_PREFIX):]
        api_url = "https://en.wikipedia.org/w/api.php"
        is_zh = False  # 页面是英文维基
    else:
        print("不支持的维基百科链接格式")
        return None, None
    
    title = unquote(title)
    params = {
        'action': 'parse',
        'page': title,
        'format': 'json',
        'prop': 'text',
        'redirects': 1,  # 自动处理重定向
        'utf8': 1
    }
    
    session = requests.Session()
    retries = Retry(total=retries, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    try:
        response = session.get(api_url, params=params, proxies={"http":"127.0.0.1:7890", "https":"127.0.0.1:7890"}, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'parse' in data and 'text' in data['parse']:
            page_url = f"{EN_WIKI_PREFIX}{data['parse']['title']}" if not is_zh else f"{ZH_WIKI_PREFIX}{data['parse']['title']}"
            return data['parse']['text']['*'], page_url
    except requests.RequestException as e:
        print(f"请求出错: {e}")
    return None, None

# 下载并存储图片
def extract_and_store_images(page_html, wikipedia_link):
    soup = BeautifulSoup(page_html, "html.parser")
    images = soup.find_all("img")
    
    for img_tag in images:
        img_url = img_tag.get("src")
        if img_url:
            if img_url.startswith("//"):
                img_url = "https:" + img_url
            elif img_url.startswith("/"):
                img_url = urljoin(wikipedia_link, img_url)

            try:
                img_response = requests.get(img_url)
                if img_response.status_code == 200:
                    img_base64 = base64.b64encode(img_response.content).decode("utf-8")
                    document = {
                        "image_url": img_url,
                        "image_data": img_base64
                    }
                    # 将图片数据保存到指定的 JSONL 文件中
                    with open(IMAGE_OUTPUT_PATH, 'a', encoding='utf-8') as img_file:
                        json.dump(document, img_file, ensure_ascii=False)
                        img_file.write('\n')
                    print(f"图片 {img_url} 已成功存储")
            except requests.RequestException as e:
                print(f"下载图片 {img_url} 时出错：{e}")

# 解析页面中的链接并存储
def extract_and_store_links(soup, current_wikipedia_link):
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and href.startswith("/wiki/") and not href.startswith("/wiki/File:"):  
            if current_wikipedia_link.startswith(ZH_WIKI_PREFIX):
                full_url = ZH_WIKI_PREFIX + href.lstrip("/wiki/")  # 如果是中文维基，拼接中文维基的完整 URL
            elif current_wikipedia_link.startswith(EN_WIKI_PREFIX):
                full_url = EN_WIKI_PREFIX + href.lstrip("/wiki/")  # 如果是英文维基，拼接英文维基的完整 URL
            else:
                continue

            link_html, link_url = download_wikipedia_page(full_url)
            if link_html and link_url:
                # 将链接页面的 HTML 存入指定的 JSONL 文件中
                document = {"url": link_url, "content": link_html}
                with open(HTML_OUTPUT_PATH, 'a', encoding='utf-8') as page_file:
                    json.dump(document, page_file, ensure_ascii=False)
                    page_file.write('\n')
                print(f"链接内容已存储：{link_url}")
                extract_and_store_images(link_html, link_url)

# 存储页面内容
def store_page_content(wikipedia_link):
    if wikipedia_link in downloaded_pages:
        print(f"页面 {wikipedia_link} 已经下载过，跳过处理。")
        return

    page_html, page_url = download_wikipedia_page(wikipedia_link)
    if not page_html:
        print(f"无法获取页面内容：{wikipedia_link}")
        return
    
    # 标记此页面已被下载
    downloaded_pages.add(wikipedia_link)

    # 如果是中文页面，进行繁体转简体
    if wikipedia_link.startswith(ZH_WIKI_PREFIX):
        page_html = converter.convert(page_html)
    
    # 将页面 HTML 存入指定的 JSONL 文件中
    document = {"url": page_url, "content": page_html}
    with open(HTML_OUTPUT_PATH, 'a', encoding='utf-8') as page_file:
        json.dump(document, page_file, ensure_ascii=False)
        page_file.write('\n')
    print(f"页面内容已存储：{page_url}")
    
    extract_and_store_images(page_html, page_url)
    
    soup = BeautifulSoup(page_html, "html.parser")
    extract_and_store_links(soup, page_url)

# 处理所有维基百科链接
def process_all_wikipedia_links(folder_path):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".jsonl")]
    with ThreadPoolExecutor(max_workers=10) as executor:  # 设置最大线程数
        futures = []
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    data = json.loads(line)
                    wikipedia_link = data.get("wikipediaLink_zh")
                    if wikipedia_link:
                        futures.append(executor.submit(store_page_content, wikipedia_link))

        for future in as_completed(futures):
            future.result()  

process_all_wikipedia_links("temp/3")
