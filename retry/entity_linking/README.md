# 实体链接重建

这套代码放在 `retry/entity_linking/`，目标是先把服务器上丢失的“实体链接处理后数据”恢复出来，并且把流程整理成一套独立、可复跑的模块。

## 1. 当前确认的处理后格式

实体链接这一步的处理后数据不是 `DBP15K` 那套 `cleaned_ent_ids_* / triples_*`。

它对应的是两份 JSONL：

- `entity_words_zh.jsonl`
- `entity_words_en.jsonl`

每一行都保留原始 wiki 记录字段，并额外补上这些中间字段：

```json
{
  "label": "...",
  "zh_aliases": ["..."],
  "en_aliases": ["..."],
  "zh_description": "...",
  "en_description": "...",
  "_entity_words_zh": ["..."],
  "_entity_freq_zh": {"...": 1},
  "_entity_count_zh": 3,
  "_entity_words_en": ["..."],
  "_entity_freq_en": {"...": 1},
  "_entity_count_en": 2,
  "_entity_count_total": 5,
  "_entity_words_zh_vector": [0.01, -0.02, "..."],
  "_entity_words_en_vector": null
}
```

字段含义：

- `_entity_words_zh`：从中文描述里抽到的实体词
- `_entity_words_en`：从英文描述里抽到的实体词
- `_entity_freq_*`：当前项目里实际就是去重后的 `{词: 1}`
- `_entity_words_*_vector`：对抽出的实体词做逐项编码后再平均池化得到的 1024 维向量

## 2. 为什么这里做了双路径

你当前仓库里原始 wiki 数据还在，但原实验所需的 NER / encoder 模型并不完整，所以这里提供两条路径：

- `transformer`：尽量贴近原实验，前提是你补齐 NER 模型或编码模型
- `dictionary`：不依赖原 NER 模型，直接用全量别名字典从描述中回捞实体词，先把中间数据格式恢复出来

向量也一样：

- `transformer`：真实模型编码
- `none`：先只恢复字段结构，不生成向量
- `hash`：用稳定哈希向量做可运行回退，仅用于联调/验格式，不用于正式实验指标

## 3. 推荐运行方式

先做一个小样本 smoke test，只看格式：

```bash
python retry/run_entity_linking.py ^
  --extractor dictionary ^
  --vectorizer none ^
  --max-records 5 ^
  --overwrite
```

如果这一步正常，再跑全量格式恢复：

```bash
python retry/run_entity_linking.py ^
  --extractor dictionary ^
  --vectorizer none ^
  --overwrite
```

如果后面你补齐了模型，再跑更接近原实验的版本：

```bash
python retry/run_entity_linking.py ^
  --extractor transformer ^
  --transformer-ner-model path\\to\\ner_finetuned ^
  --vectorizer transformer ^
  --transformer-vector-model path\\to\\encoder_or_bert ^
  --overwrite
```

## 4. 输出位置

默认输出到：

- `retry/output/entity_linking/entity_words_zh.jsonl`
- `retry/output/entity_linking/entity_words_en.jsonl`
- `retry/output/entity_linking/entity_words_zh.stats.json`
- `retry/output/entity_linking/entity_words_en.stats.json`

## 5. 和下一步的关系

这一步恢复的是“实体链接中间数据”。

后面你要做的“由编号查实体、按 `DBP15K` 关系做关系检索、再送模型评测”，会接到另一套对齐数据：

- `data/processed/alignment/DBP15K/.../cleaned_ent_ids_*`
- `data/processed/alignment/DBP15K/.../triples_*`
- `data/processed/alignment/DBP15K/.../ref_ent_ids|test|valid`

也就是：

- 本目录先解决 `wiki JSONL -> entity_words_*.jsonl`
- 下一步再接 `DBP15K id -> entity / relation / triple` 的检索与评测
