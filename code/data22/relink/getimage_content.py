# #通过图片链接得到bs64存入字段content
# import requests
# import base64
# import json
# from requests.adapters import HTTPAdapter
# from requests.packages.urllib3.util.retry import Retry

# # 增加重试机制
# def download_and_convert_to_base64(image_url, retries=3):
#     session = requests.Session()
#     retry_strategy = Retry(
#         total=retries,
#         backoff_factor=1,
#         status_forcelist=[429, 500, 502, 503, 504],
#         allowed_methods=["GET"]
#     )
#     adapter = HTTPAdapter(max_retries=retry_strategy)
#     session.mount("https://", adapter)
#     session.mount("http://", adapter)

#     try:
#         response = session.get(image_url, timeout=10)
#         response.raise_for_status()
#         # 将图片内容转换为 Base64 编码
#         image_base64 = base64.b64encode(response.content).decode("utf-8")
#         return image_base64
#     except requests.RequestException as e:
#         print(f"下载图片失败: {image_url}, 错误: {e}")
#         return None

# def process_image_links(input_file, output_file):
#     results = []
#     with open(input_file, "r", encoding="utf-8") as f:
#         for line in f:
#             data = json.loads(line)
#             image_url = data.get("link")
#             if image_url:
#                 print(f"处理图片链接: {image_url}")
#                 image_base64 = download_and_convert_to_base64(image_url)
#                 if image_base64:
#                     data["content"] = image_base64
#                 else:
#                     data["content"] = None
#                 results.append(data)
    
#     with open(output_file, "w", encoding="utf-8") as f:
#         for result in results:
#             f.write(json.dumps(result, ensure_ascii=False) + "\n")

# if __name__ == "__main__":
#     input_file = "demo.jsonl"  # 包含图片链接的文件
#     output_file = "demo_image_with_content.jsonl"  # 输出文件
#     process_image_links(input_file, output_file)

import requests
import base64
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# 增加重试机制和设置 User-Agent
def download_and_convert_to_base64(image_url, retries=3):
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = session.get(image_url, headers=headers, timeout=10)
        response.raise_for_status()
        # 将图片内容转换为 Base64 编码
        image_base64 = base64.b64encode(response.content).decode("utf-8")
        return image_base64
    except requests.RequestException as e:
        print(f"下载图片失败: {image_url}, 错误: {e}")
        return None

def process_image_links(input_file, output_file, max_workers=16):
    results = []
    futures = []
    with open(input_file, "r", encoding="utf-8") as f:
        data_list = [json.loads(line) for line in f]

    # 使用线程池处理图片链接
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for data in data_list:
            image_url = data.get("link")
            if image_url:
                print(f"提交任务: {image_url}")
                future = executor.submit(download_and_convert_to_base64, image_url)
                futures.append((future, data))

    # 收集结果
    for future, data in futures:
        image_base64 = future.result()
        if image_base64:
            data["content"] = image_base64
        else:
            data["content"] = None
        results.append(data)

    # 写入输出文件
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    input_file = "image_relink4.jsonl"  # 包含图片链接的文件
    output_file = "image_relink4_with_content.jsonl"  # 输出文件
    process_image_links(input_file, output_file)

# import requests
# import base64
# import json
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from requests.adapters import HTTPAdapter
# from requests.packages.urllib3.util.retry import Retry

# # 设置代理
# PROXIES = {"http": "127.0.0.1:7890", "https": "127.0.0.1:7890"}

# # 增加重试机制和设置 User-Agent
# def download_and_convert_to_base64(image_url, retries=10):
#     session = requests.Session()
#     retry_strategy = Retry(
#         total=retries,
#         backoff_factor=1,
#         status_forcelist=[429, 500, 502, 503, 504],
#         allowed_methods=["GET"]
#     )
#     adapter = HTTPAdapter(max_retries=retry_strategy)
#     session.mount("https://", adapter)
#     session.mount("http://", adapter)

#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
#     }

#     try:
#         # 使用代理发送请求
#         response = session.get(image_url, headers=headers, proxies=PROXIES, timeout=20)
#         response.raise_for_status()
#         # 将图片内容转换为 Base64 编码
#         image_base64 = base64.b64encode(response.content).decode("utf-8")
#         return image_base64
#     except requests.RequestException as e:
#         print(f"下载图片失败: {image_url}, 错误: {e}")
#         return None

# def process_image_links(input_file, output_file, max_workers=16):
#     results = []
#     futures = []
#     with open(input_file, "r", encoding="utf-8") as f:
#         data_list = [json.loads(line) for line in f]

#     # 使用线程池处理图片链接
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         for data in data_list:
#             image_url = data.get("link")
#             if image_url:
#                 print(f"提交任务: {image_url}")
#                 future = executor.submit(download_and_convert_to_base64, image_url)
#                 futures.append((future, data))

#     # 收集结果
#     for future, data in futures:
#         image_base64 = future.result()
#         if image_base64:
#             data["content"] = image_base64
#         else:
#             data["content"] = None
#         results.append(data)

#     # 写入输出文件
#     with open(output_file, "w", encoding="utf-8") as f:
#         for result in results:
#             f.write(json.dumps(result, ensure_ascii=False) + "\n")

# if __name__ == "__main__":
#     input_file = "data1_image1.jsonl"  # 包含图片链接的文件
#     output_file = "image_with_content6.jsonl"  # 输出文件
#     process_image_links(input_file, output_file)
