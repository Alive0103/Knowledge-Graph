# 统一运行指南

这份文档是当前 `retry/` 目录下唯一推荐的统一说明。后续如果要上服务器、重新训练、生成老师报告，优先看这份。

## 1. 当前两条主线

当前实验链分成两条：

1. 实体链接恢复链
2. 实体对齐重训与评测链

完整顺序是：

`原始 wiki 数据 -> 实体链接弱监督训练 -> 实体链接重建 -> ES 评测 -> DBP15K 对齐前基线 -> 邻居图模型重训/评测 -> 老师报告`

## 2. 现在新增的实体对齐重训入口

新的邻居图模型重训入口是：

`retry/run_alignment_training.py`

它会做这些事：

1. 自动确认 `DBP15K zh_en` 数据集完整
2. 检查 `raw_LaBSE_emb_1.pkl / raw_LaBSE_emb_2.pkl`
3. 如果缺少 raw LaBSE 向量，则自动补建
4. 按邻居图自监督训练 `LaBSE + neighbor graph model`
5. 每个 epoch 自动做 `valid / test` 评测
6. 保存 `best_model.pth` 和 `last_model.pth`

## 3. 新模型命名规则

为了不覆盖旧权重，新的重训模型统一保存到：

- `retry/output/alignment_training/labse_neighbor_retrained_<dataset>_<run_tag>/best_model.pth`
- `retry/output/alignment_training/labse_neighbor_retrained_<dataset>_<run_tag>/last_model.pth`

旧权重：

- `data/models/final_model.pth`

不会被覆盖，可以始终保留为“现成可复现结果”的后备模型。

## 4. 直接重训命令

本地 CPU 直接重训：

```powershell
D:\software\anaconda\python.exe retry\run_alignment_training.py `
  --dataset zh_en `
  --device cpu `
  --epochs 150
```

如果是服务器 GPU：

```bash
python retry/run_alignment_training.py \
  --dataset zh_en \
  --device cuda:0 \
  --epochs 150
```

## 5. 重训后评测命令

假设新模型路径为：

`retry/output/alignment_training/labse_neighbor_retrained_zh_en_<run_tag>/best_model.pth`

评测命令：

```powershell
D:\software\anaconda\python.exe retry\run_alignment.py --dataset zh_en eval `
  --mode final_model `
  --device cpu `
  --model-path retry\output\alignment_training\labse_neighbor_retrained_zh_en_<run_tag>\best_model.pth `
  --json
```

## 6. 一条命令跑完整链

现在严格全流程支持三种策略：

- `reuse`
  只用现有 `final_model.pth`
- `train_if_missing`
  只有在没有现成 `final_model.pth` 时才重训
- `retrain`
  无论是否已有旧权重，都重新训练一套新模型

推荐你正式重跑时用：

```powershell
D:\software\anaconda\python.exe retry\run_rigorous_full_experiment.py `
  --dataset zh_en `
  --device cpu `
  --alignment-model-strategy retrain `
  --alignment-epochs 150 `
  --include-bge-m3
```

这条链会自动完成：

1. 资源预检
2. 实体链接训练
3. 实体链接重建
4. ES 建索引和评测
5. Raw LaBSE baseline
6. 新的邻居图模型重训
7. 新模型评测
8. Raw BGE-M3 baseline
9. 老师对比报告输出

## 7. 如何监控训练

如果单独跑邻居图重训，重点看：

- `retry/output/alignment_training/labse_neighbor_retrained_<dataset>_<run_tag>/training.log`
- `retry/output/alignment_training/labse_neighbor_retrained_<dataset>_<run_tag>/history.json`
- `retry/output/alignment_training/labse_neighbor_retrained_<dataset>_<run_tag>/summary.json`

如果跑严格全流程，还要看：

- `retry/output/rigorous_full_run_<run_tag>/state.json`

## 8. 当前老师汇报材料在哪里

当前已经写好的老师汇报材料主要在这里：

- 当前实验对比报告：
  `retry/output/experiment_comparison/zh_en_test_comparison.md`
- 当前实验对比原始 JSON：
  `retry/output/experiment_comparison/zh_en_test_comparison.json`
- 周报/答辩直接念的口径稿：
  `retry/REPORT_WEEKLY_TALK_TRACK.md`
- 当前阶段老师汇报整合稿：
  `retry/TEACHER_REPORT_CURRENT.md`

## 9. 当前严谨口径

现在可以严谨地说：

- 已完成 `Raw LaBSE baseline`
- 已完成 `Raw BGE-M3 baseline`
- 已完成基于现有 `final_model.pth` 的 `LaBSE + neighbor graph model` 对齐后评测
- 已打通新的邻居图模型重训入口
- 新的从零重训任务已经可以启动并自动评测

现在不能混着说成一件事：

- “现有权重复现评测” 不等于 “已经完成了新的从零重训”

后续老师汇报时，要把这两件事严格区分。
