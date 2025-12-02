from es_client import es

# 创建 data2 索引（主索引）
if es.indices.exists(index="data2"):
    es.indices.delete(index="data2")

index_settings = {
    "settings": {
        "analysis": {
            "filter": {
                "pinyin_filter": {
                    "type": "pinyin",
                    "first_letter": "prefix",  
                    "padding_char": ""
                },
                "abbreviation_filter": {  
                    "type": "ngram",
                    "min_gram": 2,  
                }
            },
            "analyzer": {
                "pinyin_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "pinyin_filter"]
                },
                "ik_smart_analyzer": {
                    "type": "custom",
                    "tokenizer": "ik_smart",  
                    "filter": ["lowercase"]  
                },
                "abbreviation_analyzer": {  
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "abbreviation_filter"]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "label": {
                "type": "text",
                "fields": {
                    "pinyin": {
                        "type": "text",
                        "analyzer": "pinyin_analyzer"  
                    },
                    "ik": {
                        "type": "text",
                        "analyzer": "ik_smart_analyzer"  
                    },
                    "abbreviation": {  
                        "type": "text",
                        "analyzer": "abbreviation_analyzer"
                    }
                }
            },
            "link": {
                "type": "text",
            },
            "aliases_en": {
                "type": "text",
                "fields": {
                    "pinyin": {
                        "type": "text",
                        "analyzer": "pinyin_analyzer"
                    },
                    "ik": {
                        "type": "text",
                        "analyzer": "ik_smart_analyzer"
                    },
                    "abbreviation": {  
                        "type": "text",
                        "analyzer": "abbreviation_analyzer"
                    }
                }
            },
            "aliases_zh": {
                "type": "text",
                "fields": {
                    "pinyin": {
                        "type": "text",
                        "analyzer": "pinyin_analyzer"
                    },
                    "ik": {
                        "type": "text",
                        "analyzer": "ik_smart_analyzer"
                    },
                    "abbreviation": {  
                        "type": "text",
                        "analyzer": "abbreviation_analyzer"
                    }
                }
            },
            "descriptions_en": {
                "type": "text",
                "fields": {
                    "pinyin": {
                        "type": "text",
                        "analyzer": "pinyin_analyzer"
                    },
                    "ik": {
                        "type": "text",
                        "analyzer": "ik_smart_analyzer"
                    },
                    "abbreviation": {  
                        "type": "text",
                        "analyzer": "abbreviation_analyzer"
                    }
                }
            },
            "descriptions_en_vector": {
                "type": "dense_vector",
                "dims": 1024 
            },
            "descriptions_zh": {
                "type": "text",
                "fields": {
                    "pinyin": {
                        "type": "text",
                        "analyzer": "pinyin_analyzer"
                    },
                    "ik": {
                        "type": "text",
                        "analyzer": "ik_smart_analyzer"
                    },
                    "abbreviation": {  
                        "type": "text",
                        "analyzer": "abbreviation_analyzer"
                    }
                }
            },
            "descriptions_zh_vector": {
                "type": "dense_vector",
                "dims": 1024
            },
            "content": {
                "type": "text",
                "fields": {
                    "pinyin": {
                        "type": "text",
                        "analyzer": "pinyin_analyzer"
                    },
                    "ik": {
                        "type": "text",
                        "analyzer": "ik_smart_analyzer"
                    },
                    "abbreviation": {  
                        "type": "text",
                        "analyzer": "abbreviation_analyzer"
                    }
                }
            },
            "content_vector": {
                "type": "dense_vector",
                "dims": 1024 
            },
        }
    }
}

es.indices.create(index="data2", body=index_settings)
print("成功创建索引 data2")

# 创建 data2_relink 索引
index_relink_mapping = {
    "mappings": {
        "properties": {
            "relink_url": {
                "type": "keyword" 
            },
            "relink_data": {
                "type": "text"  
            }
        }
    }
}

if es.indices.exists(index="data2_relink"):
    es.indices.delete(index="data2_relink")

es.indices.create(index="data2_relink", body=index_relink_mapping)
print("成功创建索引 data2_relink")

# 创建 data2_image 索引
index_image_mapping = {
    "mappings": {
        "properties": {
            "image_url": {
                "type": "keyword"  
            },
            "image_data": {
                "type": "text"  
            }
        }
    }
}

if es.indices.exists(index="data2_image"):
    es.indices.delete(index="data2_image")

es.indices.create(index="data2_image", body=index_image_mapping)
print("成功创建索引 data2_image")