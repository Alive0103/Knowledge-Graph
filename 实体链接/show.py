from flask import Flask, jsonify, request, Response
import re
import hashlib
from elasticsearch import Elasticsearch

es = Elasticsearch(["http://localhost:9200"])

image_cache = {}

def md5(text):
    """计算文本的 MD5 哈希值"""
    return hashlib.md5(text.encode()).hexdigest()

def get_image_data(image_url):
    """从 ES 获取图片的 Base64 数据"""
    if image_url in image_cache:
        print(f"[CACHE HIT] {image_url}")
        return image_cache[image_url]

    query = {"query": {"match": {"image_url": image_url}}}
    response = es.search(index="data1_image", body=query)
    hits = response["hits"]["hits"]

    if hits:
        image_data = hits[0]["_source"]["image_data"]
        print(f"[ES FOUND] {image_url} -> {image_data[:50]}...")  
        image_cache[image_url] = image_data  
        return image_data

    print(f"[ES NOT FOUND] {image_url}")
    return None

def replace_src(match):
    """替换 img 标签的 src 属性"""
    img_tag = match.group(0)
    image_url_match = re.search(r'src="([^"]+)"', img_tag)
    if not image_url_match:
        return img_tag

    image_url = image_url_match.group(1)
    if image_url.startswith("//"):
        image_url = "https:" + image_url 

    image_data = get_image_data(image_url)

    if image_data:
        new_src = f'src="data:image/jpeg;base64,{image_data}"'
        print(f"[REPLACE] {image_url} -> Base64")
        return re.sub(r'src="[^"]+"', new_src, img_tag)
    else:
        print(f"[KEEP ORIGINAL] {image_url}")
        return img_tag

def search_demo_military(label):
    """从 ES 搜索相关内容"""
    query = {"query": {"match": {"label": label}}}
    response = es.search(index="data1", body=query)
    hits = response["hits"]["hits"]

    if hits:
        return hits[0]["_source"]["content"]
    
    print(f"[SEARCH NOT FOUND] Label: {label}")
    return None

app = Flask(__name__)

@app.route("/search/<label>")
def search(label):
    content = search_demo_military(label)
    if not content:
        return "未找到相关内容", 404

    print(f"[PROCESSING] Cleaning images in content for {label}")

    content = re.sub(r'\s+srcset="[^"]*"', '', content)

    content = re.sub(r'<img [^>]*src="[^"]+"', replace_src, content)

    script = """
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const links = document.querySelectorAll('a[href]');
            links.forEach(link => {
                link.addEventListener('click', function(event) {
                    event.preventDefault();
                    const url = this.getAttribute('href');
                    fetch(`/get_relink_data?url=${encodeURIComponent(url)}`)
                        .then(response => response.json())
                        .then(data => {
                            if (data.content) {
                                document.body.innerHTML = data.content;
                            } else {
                                alert('链接内容未找到。');
                            }
                        })
                        .catch(error => {
                            console.error('Error:', error);
                            alert('链接内容加载失败。');
                        });
                });
            });
        });
    </script>
    """

    return Response(content + script, mimetype='text/html')

@app.route("/get_relink_data")
def get_relink_data():
    url = request.args.get("url")
    if not url:
        return jsonify({"error": "URL 参数缺失"}), 400

    query = {"query": {"match": {"relink_url": url}}}
    response = es.search(index="data1_relink", body=query)
    hits = response["hits"]["hits"]

    if hits:
        relink_data = hits[0]["_source"]["relink_data"]
        print(f"[RELINK FOUND] {url}")

        relink_data = re.sub(r'\s+srcset="[^"]*"', '', relink_data)

        relink_data = re.sub(r'<img [^>]*src="[^"]+"', replace_src, relink_data)

        return jsonify({"content": relink_data})

    print(f"[RELINK NOT FOUND] {url}")
    return jsonify({"error": "链接内容未找到"}), 404

if __name__ == "__main__":
    app.run(debug=True)
