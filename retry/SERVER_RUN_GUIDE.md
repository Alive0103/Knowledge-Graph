# 服务器运行指南

这份文档对应全新服务器环境，目标是把 `retry/` 这条链完整跑完，并生成可直接写进老师汇报的结果。

主线入口已经统一为两步：

1. `retry/run_prepare_experiment_assets.py`
2. `retry/run_rigorous_full_experiment.py`

`run_full_pipeline.py` 只保留给 smoke 用，不再作为正式实验唯一入口。

如果你不想把中间产物、模型缓存一起上传，可以先在本地生成一个最小上传目录：

```bash
python retry/build_server_raw_bundle.py
```

默认目录：

- `server_upload_raw_only/`

## 1. 最终会产出什么

完整跑完后，你应该能得到：

- 实体链接微调模型
- 重建后的 `entity_words_zh.jsonl` / `entity_words_en.jsonl`
- ES 文本检索与向量检索评测结果
- `DBP15K zh_en` 的编号查实体 / 查关系 / 查三元组 / 关系检索工具
- `Raw LaBSE`、`final_model`、`Raw BGE-M3` 三组对齐结果
- 老师汇报用的“对齐前 vs 对齐后”报告

## 2. 跑实验前先确认的文件

必须存在：

- `work_wyy/data/zh_wiki_v2.jsonl`
- `work_wyy/data/en_wiki_v3.jsonl`
- `work_wyy/data/find.xlsx`

建议存在：

- `data/models/final_model.pth`

如果 `data/processed/alignment/DBP15K/zh_en` 缺少这些文件：

- `cleaned_ent_ids_1`
- `cleaned_ent_ids_2`
- `cleaned_rel_ids_1`
- `cleaned_rel_ids_2`
- `ref_ent_ids`
- `test`
- `triples_1`
- `triples_2`
- `valid`

不用手工补。`run_prepare_experiment_assets.py` 会优先检查当前仓库，并在需要时自动恢复到：

- `retry/recovered/alignment/DBP15K/zh_en`

注意：

- 如果 `final_model.pth` 缺失，你仍然可以完成实体链接、Raw LaBSE、Raw BGE-M3
- 但老师要求里的“对齐后效果”就不能直接复现，因为那一步依赖 `final_model.pth`
- 如果 `raw_LaBSE_emb_*.pkl` 缺失，严格流程会自动下载 `LaBSE` 并重建 raw baseline

## 3. 环境准备

推荐：

- Python `3.10`
- `conda` 或 `venv`
- Docker

如果服务器有 GPU，建议用 GPU；如果只有 CPU，也能完整跑，只是会慢很多。

### 3.1 创建 Python 环境

```bash
conda create -n kg_retry python=3.10 -y
conda activate kg_retry
```

### 3.2 安装 PyTorch

CPU 示例：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

常见 CUDA 12.1 示例：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 3.3 安装其余依赖

```bash
pip install -r retry/requirements_server.txt
```

## 4. 启动本地 Elasticsearch

推荐直接用 Docker 单节点模式：

```bash
docker rm -f kg-elasticsearch >/dev/null 2>&1 || true
docker run -d --name kg-elasticsearch \
  -p 9200:9200 \
  -e discovery.type=single-node \
  -e xpack.security.enabled=false \
  -e xpack.security.http.ssl.enabled=false \
  -e ES_JAVA_OPTS="-Xms4g -Xmx4g" \
  docker.elastic.co/elasticsearch/elasticsearch:9.2.3
```

检查 ES 是否正常：

```bash
curl http://localhost:9200
```

如果服务器没有 `curl`，也可以用：

```bash
python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:9200').read().decode('utf-8'))"
```

## 5. 先做资源预检

这一步会检查：

- 原始 wiki 数据
- `find.xlsx`
- DBP15K 数据完整性
- 基础模型是否能下载/复用
- `BGE-M3` 是否能下载/复用
- 本地 ES 是否可连通

推荐命令：

```bash
mkdir -p retry/output
python retry/run_prepare_experiment_assets.py \
  --dataset zh_en \
  --check-es \
  --prepare-bge-model \
  --json \
  > retry/output/prepare_assets_zh_en.json
```

如果 Hugging Face 官方下载慢，可以加镜像：

```bash
python retry/run_prepare_experiment_assets.py \
  --dataset zh_en \
  --check-es \
  --prepare-bge-model \
  --hf-endpoint https://hf-mirror.com \
  --json \
  > retry/output/prepare_assets_zh_en.json
```

预检通过后，重点看这几个字段：

- `alignment.is_complete` 应为 `true`
- `base_model_dir` 应存在
- `bge_model_ready` 应为 `true`
- `es_info` 应非空

如果 `alignment.source_kind` 显示为 `git_head`，这是正常的，表示脚本已从仓库历史恢复了 DBP15K 缺失文本文件。

## 6. 正式跑完整实验

### 6.1 CPU 版本

```bash
mkdir -p retry/logs
nohup python retry/run_rigorous_full_experiment.py \
  --dataset zh_en \
  --device cpu \
  --include-bge-m3 \
  > retry/logs/rigorous_full_zh_en.log 2>&1 &
```

### 6.2 GPU 版本

```bash
mkdir -p retry/logs
nohup python retry/run_rigorous_full_experiment.py \
  --dataset zh_en \
  --device cuda:0 \
  --include-bge-m3 \
  --hf-endpoint https://hf-mirror.com \
  > retry/logs/rigorous_full_zh_en.log 2>&1 &
```

说明：

- `--include-bge-m3` 会自动补做 `Raw BGE-M3 baseline`
- `--device` 既影响实体链接微调，也影响 `final_model` 和 `BGE-M3` 编码
- 如果你想更细地观察进度，可以加 `--heartbeat-seconds 60`

## 7. 如何监控是否卡住

严格流程会把状态写到：

- `retry/output/rigorous_full_run_<run_tag>/state.json`

先找最新一次运行目录：

```bash
python - <<'PY'
from pathlib import Path
paths = sorted(Path('retry/output').glob('rigorous_full_run_*'))
print(paths[-1] if paths else 'NO_RUN')
PY
```

查看状态：

```bash
python - <<'PY'
from pathlib import Path
import json
paths = sorted(Path('retry/output').glob('rigorous_full_run_*'))
state_path = paths[-1] / 'state.json'
print(state_path)
print(state_path.read_text(encoding='utf-8'))
PY
```

同时看日志：

```bash
tail -f retry/logs/rigorous_full_zh_en.log
```

现在长步骤运行时会持续刷新 `updated_at`，所以如果时间还在往前走，基本就不是卡死。

## 8. 跑完后重点检查哪些产物

实体链接模型：

- `retry/output/entity_linking_training/ner_finetuned_distilbert_mbert_<run_tag>/`

实体链接重建结果：

- `retry/output/entity_linking_transformer_distilbert_mbert_<run_tag>/entity_words_zh.jsonl`
- `retry/output/entity_linking_transformer_distilbert_mbert_<run_tag>/entity_words_en.jsonl`

ES 评测：

- `retry/output/entity_linking_eval/text_only_metrics.json`
- `retry/output/entity_linking_eval/vector_only_metrics.json`

对齐评测：

- `retry/output/alignment_eval_<run_tag>/raw_labse_test.json`
- `retry/output/alignment_eval_<run_tag>/final_model_test.json`
- `retry/output/alignment_eval_<run_tag>/raw_bge_m3_test.json`

老师报告：

- `retry/output/experiment_comparison/zh_en_test_comparison.json`
- `retry/output/experiment_comparison/zh_en_test_comparison.md`

## 9. 老师汇报时怎么说

主线必须这样说：

- 对齐前：`Raw LaBSE baseline`
- 对齐后：`LaBSE + neighbor graph model`

补充横向对比可以加：

- `Raw BGE-M3 baseline`

当前这份工作区不要写成“已经具备并完成评测”的方法：

- `LaBSE SSL`

原因很简单：

- 当前工作区没有可直接使用的 `LaBSE SSL` 源码/权重
- 因此它不应被写成“当前仓库已完成的第三种方法”

## 10. 快速验功能命令

编号查实体：

```bash
python retry/run_alignment.py entity --kg 1 --id 6426
```

编号查关系：

```bash
python retry/run_alignment.py relation --kg 1 --id 106
```

关系名搜索：

```bash
python retry/run_alignment.py search-relations --kg 1 --query 使用 --limit 5 --json
```

按关系检索三元组：

```bash
python retry/run_alignment.py retrieve --kg 1 --relation-id 106 --entity-id 6426 --limit 5 --json
```

Raw LaBSE：

```bash
python retry/run_alignment.py eval --mode raw --split test --json
```

final_model：

```bash
python retry/run_alignment.py eval --mode final_model --split test --device cpu --json
```

Raw BGE-M3：

```bash
python retry/run_alignment.py eval --mode raw --embedding-name bge_m3 --split test --json
```

## 11. 当前本地已验证结果

这组结果已经在本机严格全流程上跑出，可作为你上服务器后的对照：

- `Raw LaBSE baseline`: `MRR=0.478`, `Hits@1=0.410`, `Hits@5=0.559`, `Hits@10=0.606`
- `LaBSE + neighbor graph model`: `MRR=0.690`, `Hits@1=0.621`, `Hits@5=0.773`, `Hits@10=0.810`
- `Raw BGE-M3 baseline`: `MRR=0.679`, `Hits@1=0.624`, `Hits@5=0.745`, `Hits@10=0.776`

实体链接 ES 评测：

- `text_only`: `MRR=0.6536`, `Hits@1=0.5676`, `Hits@5=0.7635`, `Hits@10=0.8063`
- `vector_only`: `MRR=0.0223`, `Hits@1=0.0158`, `Hits@5=0.0383`, `Hits@10=0.0428`

## 12. 常见问题

### 12.1 跑到一半可以断网吗

可以分两种情况：

- 还在下载 Hugging Face 模型时，不要断网
- 模型已经下载完、训练和评测已经开始后，后续大多数步骤不再依赖外网

### 12.2 必须 GPU 吗

不是。

- CPU 可以完整跑
- 只是实体链接微调和 `BGE-M3` 编码会明显更慢
- 如果你要补更多学习型对比方法，GPU 更现实

### 12.3 `final_model.pth` 缺失怎么办

这会直接影响“对齐后结果”。

你有两个选择：

1. 从原实验目录拷贝已有 `final_model.pth`
2. 另行恢复并重训邻居图模型

如果这一步没解决，就不要在老师报告里写“对齐后”结果已经重现。
