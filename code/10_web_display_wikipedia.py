import os
import json
from flask import Flask, render_template_string, jsonify
from bs4 import BeautifulSoup
import base64

app = Flask(__name__)

WIKIDATA_PATH = "data4/wikidata.jsonl"
IMAGE_PATH = "data4/wiki_image.jsonl"
PAGE_PATH = "data4/wiki_page.jsonl"

def load_wikidata(link):
    """从 wikidata.jsonl 文件中加载对应页面的 HTML 内容"""
    with open(WIKIDATA_PATH, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            if data.get("link") == link:
                return data.get("content")
    return None

def load_images_from_html(page_html):
    """从页面 HTML 中提取图片链接，并将其转换为 Base64 编码"""
    soup = BeautifulSoup(page_html, "html.parser")
    img_urls = []
    
    for img_tag in soup.find_all("img"):
        img_url = img_tag.get("src")
        if img_url:
            if img_url.startswith("//"):
                img_url = "https:" + img_url
            elif img_url.startswith("/"):
                img_url = "https://zh.wikipedia.org" + img_url  # 假设所有图片是 zh.wikipedia
            img_urls.append(img_url)
    
    return img_urls

def load_images(img_urls):
    """根据图片链接从 wiki_image.jsonl 中加载对应图片的 Base64 编码内容"""
    images = {}
    with open(IMAGE_PATH, 'r', encoding='utf-8') as file:
        for line in file:
            img_data = json.loads(line)
            img_url = img_data.get("image_url")
            if img_url in img_urls:
                images[img_url] = img_data.get("image_data")
    return images

def load_page_content(link):
    """从 wiki_page.jsonl 中加载页面内容"""
    with open(PAGE_PATH, 'r', encoding='utf-8') as file:
        for line in file:
            page_data = json.loads(line)
            if page_data.get("url") == link:
                return page_data.get("content")
    return None

@app.route("/page/<path:link>")
def show_page(link):
    # 检查链接是否以 'wiki/' 开头，如果是，则补全为完整的 URL
    if link.startswith("wiki/"):
        link = "https://zh.wikipedia.org/" + link

    # 加载页面内容
    page_html = load_wikidata(link)
    if not page_html:
        print(link)
        return "页面内容未找到", 404

    # 提取图片 URL
    img_urls = load_images_from_html(page_html)
    # 加载图片 Base64 编码
    images = load_images(img_urls)
    # 替换页面 HTML 中的图片链接为 Base64 数据
    soup = BeautifulSoup(page_html, "html.parser")
    for img_tag in soup.find_all("img"):
        img_url = img_tag.get("src")
        if img_url in images:
            img_tag["src"] = f"data:image/jpeg;base64,{images[img_url]}"

    # 载入文章内容
    page_content = load_page_content(link)
    if page_content:
        page_html += page_content  # 文章内容添加到页面

    # 渲染页面
    return render_template_string(page_html)

if __name__ == "__main__":
    app.run(debug=True)


