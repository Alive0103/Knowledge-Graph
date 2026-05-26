# DBP15K 查询与评测

`retry/alignment/` 把 `DBP15K` 的查询、关系检索和评测流程整理成了一套独立工具。

## 功能

- 编号查实体
- 编号查关系
- 查某实体相关三元组
- 按关系编号或关系名检索三元组
- 查询对齐实体
- 评测 raw embedding baseline
- 评测 `final_model.pth`

## 入口

```powershell
D:\software\anaconda\python.exe retry\run_alignment.py <subcommand> ...
```

## 常用命令

编号查实体：

```powershell
D:\software\anaconda\python.exe retry\run_alignment.py entity --kg 1 --id 4112
```

编号查关系：

```powershell
D:\software\anaconda\python.exe retry\run_alignment.py relation --kg 1 --id 106
```

查某实体相关三元组：

```powershell
D:\software\anaconda\python.exe retry\run_alignment.py triples --kg 1 --entity-id 4112 --limit 10
```

按关系编号检索：

```powershell
D:\software\anaconda\python.exe retry\run_alignment.py retrieve --kg 1 --relation-id 106 --limit 10
```

按关系文本检索：

```powershell
D:\software\anaconda\python.exe retry\run_alignment.py search-relations --kg 1 --query 海军
```

查对齐实体：

```powershell
D:\software\anaconda\python.exe retry\run_alignment.py alignment --kg 1 --id 4112 --split all
```

LaBSE raw baseline：

```powershell
D:\software\anaconda\python.exe retry\run_alignment.py eval --mode raw --split test
```

`final_model` 评测：

```powershell
D:\software\anaconda\python.exe retry\run_alignment.py eval --mode final_model --split test --device cpu
```

## BGE-M3 baseline

先生成实体名称向量：

```powershell
D:\software\anaconda\python.exe retry\run_alignment_embedding_baseline.py `
  --dataset zh_en `
  --model-name BAAI/bge-m3 `
  --device cpu
```

生成后会在 `data/processed/alignment/DBP15K/zh_en/` 下得到：

- `raw_BGE_M3_emb_1.pkl`
- `raw_BGE_M3_emb_2.pkl`
- `raw_BGE_M3_emb_metadata.json`

然后评测：

```powershell
D:\software\anaconda\python.exe retry\run_alignment.py eval `
  --mode raw `
  --embedding-name bge_m3 `
  --split test `
  --json
```

## BGE-M3 + graph model

现在 `retry/run_alignment_training.py` 已支持直接用 `BGE-M3` 从零重训 graph 模型，不需要再手动改输入维度。

训练示例：

```powershell
D:\software\anaconda\python.exe retry\run_alignment_training.py `
  --dataset zh_en `
  --embedding-name bge_m3 `
  --device cpu `
  --epochs 150
```

评测示例：

```powershell
D:\software\anaconda\python.exe retry\run_alignment.py eval `
  --mode final_model `
  --embedding-name bge_m3 `
  --model-path retry\output\alignment_training\bge_m3_neighbor_retrained_zh_en_<run_tag>\best_model.pth `
  --split test `
  --json
```

注意：

- 当前 `final_model.pth` 是基于 `LaBSE 768 维` 训练的。
- 因此 `BGE-M3` 现在只作为 `raw embedding baseline` 加入对比。
- 如果想做 `BGE-M3 + graph model`，需要把图模型输入维度改成 `BGE-M3` 的输出维度后重新训练。
