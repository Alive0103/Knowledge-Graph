# 本地 Elasticsearch 使用说明

`work_wyy/es_client.py` 现在默认连接：

```text
http://localhost:9200
```

也就是说，只要本机起了一个本地 Elasticsearch，`work_wyy` 这条实体链接检索线就可以直接改走本地，不再依赖之前挂掉的远端阿里云实例。

## 1. 安装 Python 侧依赖

```bash
pip install elasticsearch
```

## 2. 启动本地 ES

推荐使用单节点本地实例。只要能从本机访问 `http://localhost:9200` 即可。

## 3. 可选环境变量

如果你想改地址、索引名、认证信息，可以在运行前设置：

```powershell
$env:KG_ES_URL="http://localhost:9200"
$env:KG_ES_INDEX_NAME="data2"
```

如果你使用的是 Elastic 官方 Windows `.zip` 默认安装，通常要改成：

```powershell
$env:KG_ES_URL="https://localhost:9200"
$env:KG_ES_USERNAME="elastic"
$env:KG_ES_PASSWORD="your-password"
$env:KG_ES_VERIFY_CERTS="true"
$env:KG_ES_CA_CERTS="D:\\path\\to\\http_ca.crt"
```

如果 ES 开了认证：

```powershell
$env:KG_ES_USERNAME="elastic"
$env:KG_ES_PASSWORD="your-password"
```

如果要切回远端或 HTTPS，也可以继续设置：

```powershell
$env:KG_ES_VERIFY_CERTS="true"
$env:KG_ES_CA_CERTS="D:\\path\\to\\http_ca.crt"
$env:KG_ES_COMPAT_MODE="false"
```

说明：

- `KG_ES_URL` 默认值是 `http://localhost:9200`
- `KG_ES_INDEX_NAME` 默认值是 `data2`
- `KG_ES_COMPAT_MODE` 主要给阿里云兼容头用，本地通常不需要

## 4. 建索引并导入数据

在 `work_wyy/vector/` 下执行：

```bash
python vector2ES.py
```

它会：

- 使用 `ES_INDEX_NAME`
- 创建向量索引
- 读取 `work_wyy/data/entity_words_zh.jsonl` 和 `entity_words_en.jsonl`
- 把文本字段和向量字段一起导入 ES

## 5. 运行检索评测

```bash
python work_wyy/search_label_aliases.py
python work_wyy/search_vllm.py --es-text-only
python work_wyy/search_vllm.py --vector-only
```

## 6. 代码行为

以下脚本现在都能读取同一套 ES 环境变量：

- `work_wyy/es_client.py`
- `work_wyy/search_label_aliases.py`
- `work_wyy/search_vllm.py`
- `work_wyy/vector/vector2ES.py`
