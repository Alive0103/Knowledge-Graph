# 给老师汇报的当前结果说明

## 1. 当前已经完成的工作

目前 `DBP15K zh_en` 这条实验链已经从数据恢复、查询工具、模型评测推进到了“可复现实验结果”的状态，已经完成的部分包括：

1. 合并并恢复了实体对齐所需的数据文件，补齐了实体、关系、三元组和对齐划分。
2. 实现了“编号查实体、编号查关系、按实体查三元组、按关系检索三元组”等查询工具。
3. 在 `retry/` 中整理并接通了统一的实体对齐训练与评测入口。
4. 完成了多种方法的同口径评测，包括原始文本向量基线、现有邻居图模型权重复现评测，以及本次从零重训后的新权重评测。

## 2. 当前可严谨汇报的实验结果

数据集：`DBP15K zh_en`  
评测划分：`test`  
测试集实体对数：`1206`

| 方法 | MRR | Hits@1 | Hits@5 | Hits@10 |
| --- | ---: | ---: | ---: | ---: |
| Raw LaBSE baseline | 0.478 | 0.410 | 0.559 | 0.606 |
| Raw BGE-M3 baseline | 0.679 | 0.624 | 0.745 | 0.776 |
| LaBSE + neighbor graph model（现有 `final_model.pth`） | 0.690 | 0.621 | 0.773 | 0.810 |
| LaBSE + neighbor graph model（本次重训 `best_model.pth`） | 0.696 | 0.631 | 0.773 | 0.810 |

## 3. “对齐前后”对比结论

如果老师重点看“对齐前后链接效果对比”，主线对比应该是：

- 对齐前：`Raw LaBSE baseline`
- 对齐后：`LaBSE + neighbor graph model`

其中有两组可用口径：

1. 基于现有旧权重 `final_model.pth`
   - `MRR +0.212`
   - `Hits@1 +0.211`
   - `Hits@5 +0.214`
   - `Hits@10 +0.204`
2. 基于本次从零重训得到的新权重 `best_model.pth`
   - `MRR +0.218`
   - `Hits@1 +0.221`
   - `Hits@5 +0.214`
   - `Hits@10 +0.204`

从结果上看，本次重训后的新权重相对于仓库内已有旧权重还有小幅提升：

- `MRR +0.006`
- `Hits@1 +0.010`
- `Hits@5 +0.000`
- `Hits@10 +0.000`

## 4. 本次重训的严谨说明

这次在 `retry/` 中完成的是一条独立的新训练链路，不会覆盖仓库里原有的 `data/models/final_model.pth`。

本次重训结果如下：

- 训练轮数：`150`
- 最优轮次：`147`
- 模型选择指标：`valid_hits@1`
- 最优验证集结果：`valid_mrr=0.686`，`valid_hits@1=0.623`
- 最优测试集结果：`test_mrr=0.696`，`test_hits@1=0.631`

因此现在可以严谨地区分两件事：

1. 已完成“现有旧权重”的复现评测。
2. 已完成“从零重训新权重”的训练和评测。

## 5. 当前还需要如实说明的限制

1. `BGE-M3` 目前完成的是原始文本向量基线评测，还没有完成 `BGE-M3 + graph` 的重训结果。
2. 如果要做 `BGE-M3 + graph`，需要重新生成 `BGE-M3` 实体向量，并重新适配邻居图模型输入后再训练。
3. 当前老师要求的“多种实体对齐方法对比”和“对齐前后效果参考”已经可以汇报，但若继续扩展方法数，可以把 `BGE-M3 + graph` 作为下一阶段工作。

## 6. 关键产物位置

- 旧权重：`data/models/final_model.pth`
- 新权重目录：`retry/output/alignment_training/labse_neighbor_retrained_zh_en_teacher_rerun_20260327_133549/`
- 新权重摘要：`retry/output/alignment_training/labse_neighbor_retrained_zh_en_teacher_rerun_20260327_133549/summary.json`
- 新权重训练日志：`retry/output/alignment_training/labse_neighbor_retrained_zh_en_teacher_rerun_20260327_133549/training.log`
- 新版结果对比：`retry/output/experiment_comparison/zh_en_test_comparison_retrained.md`
- 新版结果 JSON：`retry/output/experiment_comparison/zh_en_test_comparison_retrained.json`

## 7. 可直接对老师说的一段话

目前我已经把实体对齐实验链恢复到了可复现实验的状态。针对 `DBP15K zh_en`，我完成了原始 LaBSE 基线、原始 BGE-M3 基线、已有邻居图模型权重复现评测，以及本次从零重训后的新邻居图模型评测。从结果上看，若以 `Raw LaBSE baseline` 作为对齐前结果，邻居图增强后的模型在测试集上可以把 `MRR` 从 `0.478` 提升到 `0.690` 或 `0.696`，`Hits@1` 从 `0.410` 提升到 `0.621` 或 `0.631`，说明图结构增强对跨语言实体对齐是有效的。本次新重训权重相对于仓库已有旧权重还有小幅提升，因此现在不仅能做“对齐前后”的对比，也能把“旧权重复现”和“新权重重训”两组结果区分开来汇报。
