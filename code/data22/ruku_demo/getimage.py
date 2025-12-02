# #从demo和demo_relink获取图片链接

# import json

# def extract_images(input_files, output_file):
#     # 用于存储所有图片链接
#     all_images = set()  # 使用集合去重

#     # 遍历所有输入文件
#     for input_file in input_files:
#         with open(input_file, "r", encoding="utf-8") as f:
#             for line in f:
#                 data = json.loads(line)
#                 # 提取 images 和 images_link 字段
#                 images = data.get("images", [])
#                 images_link = data.get("images_link", [])
#                 # 将图片链接添加到集合中
#                 all_images.update(images)
#                 all_images.update(images_link)

#     # 将去重后的图片链接写入输出文件
#     with open(output_file, "w", encoding="utf-8") as f:
#         for image in all_images:
#             f.write(json.dumps({"link": image}, ensure_ascii=False) + "\n")

#     print(f"已提取 {len(all_images)} 条图片链接到 {output_file}")

# if __name__ == "__main__":
#     # 输入文件列表
#     input_files = ["demo.jsonl", "demo_relink.jsonl"]
#     # 输出文件
#     output_file = "demo_image.jsonl"
#     # 执行提取操作
#     extract_images(input_files, output_file)

#--------------------------------------------------------------------------------------------------------------------
#通过图片链接得到bs64存入字段content
import requests
import base64
import json
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# 增加重试机制
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

    try:
        response = session.get(image_url, timeout=10)
        response.raise_for_status()
        # 将图片内容转换为 Base64 编码
        image_base64 = base64.b64encode(response.content).decode("utf-8")
        return image_base64
    except requests.RequestException as e:
        print(f"下载图片失败: {image_url}, 错误: {e}")
        return None

def process_image_links(input_file, output_file):
    results = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            image_url = data.get("link")
            if image_url:
                print(f"处理图片链接: {image_url}")
                image_base64 = download_and_convert_to_base64(image_url)
                if image_base64:
                    data["content"] = image_base64
                else:
                    data["content"] = None
                results.append(data)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    input_file = "demo_image.jsonl"  # 包含图片链接的文件
    output_file = "demo_image_with_content.jsonl"  # 输出文件
    process_image_links(input_file, output_file)
