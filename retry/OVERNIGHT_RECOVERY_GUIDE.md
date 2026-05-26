# 夜间恢复与服务器复现实验说明

这份文档对应当前 `retry/` 这条恢复链，目标是把以下流程串起来：

1. 原始 wiki 数据 -> 弱监督 NER 训练
2. NER 微调模型 -> 实体链接处理后数据重建
3. 实体链接处理后数据 -> 本地 Elasticsearch 导入
4. Elasticsearch -> 文本检索 / 向量检索评测
5. DBP15K -> 对齐前后对比评测
6. 新增 `BGE-M3` raw baseline -> 横向对比

## 1. 当前夜间脚本做什么

当前新增的后台脚本是：

`retry/run_overnight_recovery.py`

它会按顺序执行：

1. 等待当前训练完成
2. 如果当前训练中途失败，自动切到 fallback 全量训练
3. 如果只跑了 1 个 epoch，则继续补到目标 epoch 数
4. 用微调后的模型重建 `entity_words_zh.jsonl / entity_words_en.jsonl`
5. 导入本地 ES
6. 跑实体链接 `text_only / vector_only` 评测
7. 跑 DBP15K `raw LaBSE`
8. 跑 DBP15K `final_model`
9. 尝试构建并评测 `raw BGE-M3`
10. 生成老师要的“对齐前 vs 对齐后”对比报告

## 2. 明天先看哪里

优先看这个状态文件：

`retry/output/overnight_recovery/state.json`

它会持续更新每一步的状态，关键字段是：

- `steps.wait_initial_training`
- `steps.fallback_full_training`
- `steps.resume_training`
- `steps.entity_linking_rebuild`
- `steps.es_index`
- `steps.es_eval`
- `steps.alignment_raw_labse`
- `steps.alignment_final_model`
- `steps.build_bge_m3_baseline`
- `steps.alignment_raw_bge_m3`
- `steps.experiment_comparison`

如果最后有：

- `status = completed`
- `completed_at`

就说明整条夜间链路已经跑完。

## 3. 跑完后重点产物

### 3.1 实体链接训练与重建

- 微调模型：
  - `retry/output/entity_linking_training/ner_finetuned_distilbert_mbert_e2/`
- 处理后实体链接数据：
  - `retry/output/entity_linking_transformer_distilbert_mbert/entity_words_zh.jsonl`
  - `retry/output/entity_linking_transformer_distilbert_mbert/entity_words_en.jsonl`

### 3.2 ES 导入与实体链接评测

- ES 导入摘要：
  - `retry/output/overnight_recovery/es_index_summary.json`
- 实体链接评测摘要：
  - `retry/output/overnight_recovery/entity_linking_eval_summary.json`
- 分模式评测结果：
  - `retry/output/entity_linking_eval/text_only_metrics.json`
  - `retry/output/entity_linking_eval/vector_only_metrics.json`

### 3.3 DBP15K 对齐评测

- Raw LaBSE：
  - `retry/output/alignment_eval/raw_labse_test.json`
- Final model：
  - `retry/output/alignment_eval/final_model_test.json`
- Raw BGE-M3：
  - `retry/output/alignment_eval/raw_bge_m3_test.json`

### 3.4 老师报告用对比结果

- JSON 报告：
  - `retry/output/experiment_comparison/zh_en_test_comparison.json`
- Markdown 报告：
  - `retry/output/experiment_comparison/zh_en_test_comparison.md`

这里已经会整理：

- 对齐前：`Raw LaBSE baseline`
- 对齐后：`LaBSE + neighbor graph model`
- 新增基线：`Raw BGE-M3 baseline`（如果构建成功）

## 4. 本机重新手工启动

如果你想在本机重新从头挂一条夜间链：

```powershell
D:\software\anaconda\python.exe retry\run_overnight_recovery.py
```

如果只想单独导入 ES：

```powershell
D:\software\anaconda\python.exe retry\run_entity_linking_es.py index `
  --input-dir retry\output\entity_linking_transformer_distilbert_mbert `
  --index-name data2 `
  --json
```

如果只想单独评测实体链接：

```powershell
D:\software\anaconda\python.exe retry\run_entity_linking_es.py eval `
  --mode both `
  --index-name data2 `
  --vector-model-dir retry\output\entity_linking_training\ner_finetuned_distilbert_mbert_e2 `
  --output-dir retry\output\entity_linking_eval `
  --json
```

`--vector-model-dir` 这里要传微调后的实体链接模型目录，而不是基础模型目录；否则 `vector_only` 指标和重建阶段使用的编码器不一致。

## 5. 新服务器从零开始

### 5.1 环境

建议：

```bash
conda create -n kg_retry python=3.10 -y
conda activate kg_retry
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r retry/requirements_server.txt
```

如果服务器有 CUDA，再按服务器 CUDA 版本安装 GPU 版 PyTorch。

### 5.2 本地 ES

```bash
docker run -d --name kg-elasticsearch \
  -p 9200:9200 \
  -e discovery.type=single-node \
  -e xpack.security.enabled=false \
  -e xpack.security.http.ssl.enabled=false \
  -e "ES_JAVA_OPTS=-Xms1g -Xmx1g" \
  docker.elastic.co/elasticsearch/elasticsearch:9.2.3
```

### 5.3 一条命令挂全流程

```bash
python retry/run_overnight_recovery.py
```

然后看：

`retry/output/overnight_recovery/state.json`

## 6. 现在仓库里可直接用于老师对比的对齐方法

已经确认可直接用于结果对比的：

1. `Raw LaBSE baseline`
2. `LaBSE + neighbor graph model`
3. `Raw BGE-M3 baseline`（构建成功后）

仓库里只有源码、没有现成权重的：

4. `LaBSE SSL`

所以老师要求的“对齐前后效果对比”目前最稳妥的交付组合就是：

- 对齐前：`Raw LaBSE`
- 对齐后：`final_model`
- 补充横向对比：`Raw BGE-M3`

## 7. 补充说明

- 这条恢复链默认走 CPU，可完整跑，只是会慢。
- 当前实体链接训练已经不依赖丢失的旧 `traindata`，而是走 `weak supervision`。
- 如果 `BGE-M3` 下载失败，夜间脚本不会影响主线结果；主线仍会继续完成。
