# es_utils.py
from elasticsearch import Elasticsearch
from flask import render_template_string
from bs4 import BeautifulSoup  # 需要安装 BeautifulSoup

# 定义全局的 Elasticsearch 客户端
es = Elasticsearch(["http://localhost:9200"])

def search_demo_military(label):
    query = {
        "query": {
            "match": {
                "label": label
            }
        }
    }
    response = es.search(index="demo_military", body=query)
    hits = response["hits"]["hits"]
    if hits:
        return hits[0]["_source"]["content"]
    return None

def get_image_data(image_url):
    query = {
        "query": {
            "match": {
                "image_url": image_url
            }
        }
    }
    response = es.search(index="demo_image", body=query)
    hits = response["hits"]["hits"]
    if hits:
        return hits[0]["_source"]["image_data"]
    return None

