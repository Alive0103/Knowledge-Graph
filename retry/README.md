# Retry Pipeline

`retry/` 是这份毕业设计仓库里整理出来的“实验恢复与重跑”主入口。目标不是只做 smoke，而是把下面这条链真正串起来：

`原始 wiki 数据 -> 实体链接弱监督重建 -> Elasticsearch 检索评测 -> DBP15K 编号查询/关系检索 -> 实体对齐评测 -> 老师汇报报告`

## 推荐入口

正式实验优先用这两步，不要再把 `run_full_pipeline.py` 当成唯一主入口：

1. 资源预检与自动恢复

```bash
python retry/run_prepare_experiment_assets.py \
  --dataset zh_en \
  --check-es \
  --prepare-bge-model \
  --json
```

2. 严格顺序全流程

```bash
python retry/run_rigorous_full_experiment.py \
  --dataset zh_en \
  --device cpu \
  --include-bge-m3
```

说明：

- `run_prepare_experiment_assets.py` 会检查原始 wiki 数据、`find.xlsx`、本地 ES、基础模型，并在 `data/processed/alignment/DBP15K/zh_en` 缺文件时自动恢复到 `retry/recovered/alignment/DBP15K/zh_en`
- `run_rigorous_full_experiment.py` 会顺序执行“实体链接训练 -> 实体链接重建 -> ES 建索引与评测 -> Raw LaBSE -> final_model -> Raw BGE-M3 -> 老师对比报告”
- 长步骤执行时会持续刷新 `state.json`，便于服务器上监控
- 如果 `raw_LaBSE_emb_*.pkl` 缺失，严格流程会自动下载 `LaBSE` 并现生成 raw baseline
- 如果 `final_model.pth` 缺失，严格流程会把 `alignment_final_model` 标成 `skipped`，而不是整条链直接失败

## 辅助入口

`retry/run_full_pipeline.py`

- 用于快速 smoke 或模块联调
- 适合先验证环境是否可运行
- 不作为正式实验的唯一推荐入口

`retry/run_overnight_recovery.py`

- 旧的接力式恢复脚本
- 适合“某个训练已经在跑，只想等它跑完再继续”的场景
- 新服务器从零开始时，优先用 `run_rigorous_full_experiment.py`

## 当前已验证结果

本地严格全流程结果已经完成，状态文件：

- `retry/output/rigorous_full_run_rigorous_full_audited_20260322_232533/state.json`

当前可直接引用的 `DBP15K zh_en test` 指标：

- `Raw LaBSE baseline`: `MRR=0.478`, `Hits@1=0.410`, `Hits@5=0.559`, `Hits@10=0.606`
- `LaBSE + neighbor graph model`: `MRR=0.690`, `Hits@1=0.621`, `Hits@5=0.773`, `Hits@10=0.810`
- `Raw BGE-M3 baseline`: `MRR=0.679`, `Hits@1=0.624`, `Hits@5=0.745`, `Hits@10=0.776`

当前实体链接 ES 评测结果：

- `text_only`: `MRR=0.6536`, `Hits@1=0.5676`, `Hits@5=0.7635`, `Hits@10=0.8063`
- `vector_only`: `MRR=0.0223`, `Hits@1=0.0158`, `Hits@5=0.0383`, `Hits@10=0.0428`

## 老师汇报产物

老师要求的“对齐前 vs 对齐后”报告会输出到：

- `retry/output/experiment_comparison/zh_en_test_comparison.json`
- `retry/output/experiment_comparison/zh_en_test_comparison.md`

其中主线结论应使用：

- 对齐前：`Raw LaBSE baseline`
- 对齐后：`LaBSE + neighbor graph model`

`BGE-M3` 当前作为额外的 `raw embedding baseline` 横向补充，不是“对齐后模型”。

## 当前仓库里可对比的方法

当前工作区内可以严谨地写进报告的：

- `Raw LaBSE baseline`
- `LaBSE + neighbor graph model`
- `Raw BGE-M3 baseline`

当前工作区内不要写成“已经具备”的：

- `LaBSE SSL`
  当前这份工作区没有可直接使用的源码/权重，除非你另行补齐并重新训练

## 常用查询命令

编号查实体：

```bash
python retry/run_alignment.py entity --kg 1 --id 6426
```

编号查关系：

```bash
python retry/run_alignment.py relation --kg 1 --id 106
```

按关系检索三元组：

```bash
python retry/run_alignment.py retrieve --kg 1 --relation-id 106 --entity-id 6426 --limit 5 --json
```

按关系名搜索：

```bash
python retry/run_alignment.py search-relations --kg 1 --query 使用 --limit 5 --json
```

## 服务器文档

完整的新服务器运行步骤见：

- `retry/SERVER_RUN_GUIDE.md`

如果你想单独整理一个“只上传原始数据和代码”的目录，可以执行：

```bash
python retry/build_server_raw_bundle.py
```

默认会生成：

- `server_upload_raw_only/`
