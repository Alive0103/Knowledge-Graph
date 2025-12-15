import json
import os
import requests
from opencc import OpenCC
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import unquote
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

converter = OpenCC('t2s')

def download_wikipedia_page(wikipedia_link, retries=3):
    zh_prefix = 'https://zh.wikipedia.org/wiki/'
    en_prefix = 'https://en.wikipedia.org/wiki/'

    is_zh = False  

    if wikipedia_link.startswith(zh_prefix):
        title = wikipedia_link[len(zh_prefix):]
        api_url = "https://zh.wikipedia.org/w/api.php"
        is_zh = True
    elif wikipedia_link.startswith(en_prefix):
        title = wikipedia_link[len(en_prefix):]
        api_url = "https://en.wikipedia.org/w/api.php"
    else:
        print("不支持的维基百科链接格式")
        return None

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
        response = session.get(api_url, params=params, proxies={"http":"127.0.0.1:7890", "https":"127.0.0.1:7890"}, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'parse' in data and 'text' in data['parse']:
            content = data['parse']['text']['*']
            if is_zh:
                content = converter.convert(content)
            return content
        else:
            print(f"无法解析页面内容: {wikipedia_link}")
            return None
    except requests.RequestException as e:
        print(f"请求出错: {e}")
        return None

def process_jsonl_file(input_path, output_path):
    results = []
    tasks = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                label = data.get("itemLabel", {}).get("value", "")
                zh_link = data.get("wikipediaLink_zh")
                en_link = data.get("wikipediaLink_en")

                if zh_link:
                    tasks.append((label, zh_link, "zh"))
                if en_link:
                    tasks.append((label, en_link, "en"))
            except json.JSONDecodeError:
                print(f"解析出错: {line}")
                continue

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_task = {executor.submit(download_wikipedia_page, link): (label, link, lang)
                          for label, link, lang in tasks}

        for future in as_completed(future_to_task):
            label, link, lang = future_to_task[future]
            try:
                html_content = future.result()
                if html_content:
                    result = {
                        "label": label,
                        "link": link,
                        "content": html_content
                    }
                    results.append(result)
            except Exception as e:
                print(f"下载出错: {e}")

    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    input_folder = "data3"  
    output_folder = "data4"     

    os.makedirs(output_folder, exist_ok=True)

    for input_file in os.listdir(input_folder):
        if input_file.endswith('.jsonl'):
            input_path = os.path.join(input_folder, input_file)
            output_path = os.path.join(output_folder, input_file)

            print(f"处理文件: {input_file}")
            process_jsonl_file(input_path, output_path)

    print("所有文件处理完成。")
