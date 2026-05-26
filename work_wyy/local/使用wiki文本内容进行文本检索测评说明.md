# 使用Wiki文本内容进行文本检索测评操作说明

## 概述

本文档说明如何将 `download_wiki_content.py` 下载的文本内容存储到ES索引，并使用模式2（`es_text_only` - 纯文本检索）对这些数据进行测评。

**重要说明：**
- 本方案**不生成向量**，只存储文本内容
- 使用ES的文本检索功能（multi_match查询）
- 用于测评添加wiki文本内容后，纯文本检索的效果提升

## 前置条件

1. 已完成wiki文本内容的下载（`work_wyy/data/find_wiki_content/` 目录下有txt文件）
2. ES服务已启动并可以访问
3. 已安装必要的Python依赖包

## 操作步骤

### 步骤1: 将文本内容存储到ES索引

使用 `store_wiki_content_to_es.py` 脚本将文本文件存储到ES索引。

#### 基本用法

```bash
# 使用默认配置（索引：config.py中的ES_INDEX_NAME，目录：work_wyy/data/find_wiki_content）
python store_wiki_content_to_es.py

# 指定内容目录
python store_wiki_content_to_es.py --content-dir /path/to/find_wiki_content

# 指定ES索引名称
python store_wiki_content_to_es.py --index data_config_5_all_except_msra

# 同时指定目录和索引
python store_wiki_content_to_es.py --content-dir work_wyy/data/find_wiki_content --index data_config_5_all_except_msra
```

#### 参数说明

- `--content-dir`: 文本文件目录路径（默认: `work_wyy/data/find_wiki_content`）
- `--index`: ES索引名称（默认: 使用 `config.py` 中的 `ES_INDEX_NAME`）
- `--match-existing`: 如果找到相同link的文档则更新，否则创建新文档（默认: True）
- `--no-match-existing`: 总是创建新文档，不更新现有文档

#### 功能说明

1. **解析文本文件**: 读取每个txt文件，提取：
   - 查询词（从文件头部的 `# 查询词:` 行）
   - 链接（从文件头部的 `# 链接:` 行）
   - 文本内容（文件主体部分）

2. **存储到ES**: 
   - 如果 `--match-existing` 为True（默认），会尝试根据link字段查找现有文档并更新
   - 如果找不到现有文档或 `--no-match-existing`，则创建新文档
   - 自动添加 `wiki_content` 字段到索引映射（如果不存在）
   - **注意：不生成向量，只存储文本内容**

3. **字段说明**:
   - `wiki_content`: 文本内容（用于文本检索）
   - `wiki_content_length`: 文本长度
   - `wiki_download_time`: 下载时间
   - `wiki_filename`: 文件名

#### 示例输出

```
======================================================================
将wiki文本内容存储到ES索引（文本检索模式）
======================================================================
内容目录: D:\work\毕设\知识图谱\Knowledge-Graph\work_wyy\data\find_wiki_content
目标索引: data_config_5_all_except_msra
注意: 只存储文本内容，不生成向量
======================================================================

开始处理目录: D:\work\毕设\知识图谱\Knowledge-Graph\work_wyy\data\find_wiki_content
目标索引: data_config_5_all_except_msra
匹配模式: 更新现有文档
注意: 只存储文本内容，不生成向量（用于文本检索测评）
======================================================================
找到 444 个txt文件
处理文件: 100%|████████████████████████████████████████| 444/444 [00:45<00:00,  9.82it/s]

======================================================================
处理完成统计
======================================================================
总文件数: 444
成功: 444 个
失败: 0 个
成功率: 100.0%

索引 data_config_5_all_except_msra 当前文档数: 1234
======================================================================
✓ 完成！
======================================================================
```

### 步骤2: 使用模式2进行文本检索测评

使用 `evaluate_with_wiki_content.py` 脚本对包含wiki文本内容的ES索引进行文本检索测评。

#### 基本用法

```bash
# 使用默认配置（索引：config.py中的ES_INDEX_NAME，find.xlsx：work_wyy/data/find.xlsx）
python evaluate_with_wiki_content.py

# 指定ES索引名称
python evaluate_with_wiki_content.py --index data_config_5_all_except_msra

# 指定find.xlsx文件路径
python evaluate_with_wiki_content.py --find-file /path/to/find.xlsx

# 同时指定索引和find文件
python evaluate_with_wiki_content.py --index data_config_5_all_except_msra --find-file work_wyy/data/find.xlsx
```

#### 参数说明

- `--find-file`: find.xlsx文件路径（默认: `work_wyy/data/find.xlsx`）
- `--index`: ES索引名称（默认: 使用 `config.py` 中的 `ES_INDEX_NAME`）
- `--output`: 输出报告文件路径（默认: `trainlog/evaluation_wiki_content_*.json`）

#### 功能说明

1. **读取查询数据**: 从 `find.xlsx` 读取查询词和正确链接

2. **执行模式2文本检索**: 使用 `es_text_only` 模式进行纯文本检索
   - 使用ES的 `multi_match` 查询
   - 搜索字段包括：
     - `label^3` (权重3)
     - `aliases_zh^2` (权重2)
     - `aliases_en^2` (权重2)
     - `descriptions_zh` (权重1)
     - `descriptions_en` (权重1)
     - `wiki_content^2` (权重2，新增的wiki文本内容)
   - 不使用向量检索
   - 不使用LLM重排序

3. **计算指标**: 
   - MRR (Mean Reciprocal Rank)
   - Hit@1, Hit@5, Hit@10

4. **生成报告**: 保存详细的测评结果到JSON文件

#### 示例输出

```
======================================================================
使用模式2（es_text_only）测评包含wiki文本内容的ES索引
======================================================================
find.xlsx文件: D:\work\毕设\知识图谱\Knowledge-Graph\work_wyy\data\find.xlsx
ES索引: data_config_5_all_except_msra
检索模式: es_text_only (模式2 - 纯文本检索)
======================================================================
从 D:\work\毕设\知识图谱\Knowledge-Graph\work_wyy\data\find.xlsx 读取了 444 个有效查询-链接对

开始测评，共 444 个查询...
注意: 使用纯文本检索模式，不涉及向量检索和LLM重排序
检索对比: 100%|████████████████████████████████████████| 444/444 [02:15<00:00,  3.28it/s]

======================================================================
测评结果
======================================================================
检索模式: es_text_only (模式2 - 纯文本检索)
ES索引: data_config_5_all_except_msra
总查询数: 444
MRR: 0.7234
Hit@1: 0.6568 (65.68%)
Hit@5: 0.7919 (79.19%)
Hit@10: 0.8189 (81.89%)
======================================================================

测评报告已保存到: D:\work\毕设\知识图谱\Knowledge-Graph\work_wyy\trainlog\evaluation_wiki_content_20251215_153045.json
======================================================================
```

## 完整操作流程示例

### 场景1: 使用默认索引

```bash
# 1. 将文本内容存储到默认ES索引
python store_wiki_content_to_es.py

# 2. 使用模式2测评默认索引的文本检索效果
python evaluate_with_wiki_content.py
```

### 场景2: 使用指定索引（推荐）

```bash
# 1. 将文本内容存储到指定索引
python store_wiki_content_to_es.py --index data_config_5_all_except_msra

# 2. 使用模式2测评指定索引的文本检索效果
python evaluate_with_wiki_content.py --index data_config_5_all_except_msra
```

### 场景3: 对比添加wiki内容前后的文本检索效果（推荐）

这是最常用的场景，用于评估添加wiki文本内容对检索效果的提升。

```bash
# 步骤1: 测评添加wiki内容前的文本检索效果
python evaluate_with_wiki_content.py --index data_config_5_all_except_msra --output trainlog/eval_before_wiki.json

# 步骤2: 添加wiki文本内容到ES索引
python store_wiki_content_to_es.py --index data_config_5_all_except_msra

# 步骤3: 测评添加wiki内容后的文本检索效果
python evaluate_with_wiki_content.py --index data_config_5_all_except_msra --output trainlog/eval_after_wiki.json

# 步骤4: 自动对比两个测评结果（推荐）
python compare_wiki_evaluation.py --before trainlog/eval_before_wiki.json --after trainlog/eval_after_wiki.json
```

**对比脚本输出示例：**
```
======================================================================
添加Wiki内容前后文本检索效果对比
======================================================================

【测评信息】
  添加前索引: data_config_5_all_except_msra
  添加前时间: 2025-12-15T10:00:00
  添加后索引: data_config_5_all_except_msra
  添加后时间: 2025-12-15T10:30:00
  查询总数: 444

【指标对比】
  指标            添加前        添加后        绝对提升      相对提升      
  ----------------------------------------------------------------------
  MRR             0.6500       0.7234        +0.0734       +11.28%      ↑
  Hit@1           0.5800       0.6568        +0.0768       +13.24%      ↑
  Hit@5           0.7200       0.7919        +0.0719       +9.99%       ↑
  Hit@10          0.7800       0.8189        +0.0389       +4.99%       ↑

  ----------------------------------------------------------------------

【总结】
  提升的指标数: 4/4
  MRR相对提升: +11.28%
  Hit@1相对提升: +13.24%
  ✓ 添加wiki内容后，文本检索效果有所提升
======================================================================
```

## 注意事项

1. **不生成向量**: 本方案只存储文本内容，不生成向量。如果需要向量检索，请使用其他脚本。

2. **索引映射**: 脚本会自动检查并添加 `wiki_content` 字段到索引映射。如果添加失败，会使用默认的text类型。

3. **文档匹配**: 默认情况下，脚本会尝试根据 `link` 字段匹配现有文档并更新。如果找不到匹配的文档，会创建新文档。

4. **模式2说明**: 模式2（`es_text_only`）是纯文本检索模式：
   - 使用ES的 `multi_match` 查询
   - 搜索多个字段（包括新增的 `wiki_content` 字段）
   - 不使用向量检索
   - 不使用LLM重排序
   - 适合测评添加更多文本内容后，纯文本检索的效果提升

5. **搜索字段权重**: 
   - `label`: 权重3（最重要）
   - `wiki_content`: 权重2（新增的wiki内容，较高权重）
   - `aliases_zh`, `aliases_en`: 权重2
   - `descriptions_zh`, `descriptions_en`: 权重1

6. **性能考虑**: 
   - 文本检索速度较快，不需要GPU
   - 不需要调用外部API
   - 适合快速测评文本检索效果

## 常见问题

### Q1: 如何确认文本内容已成功存储到ES？

A: 可以检查ES索引中的文档数量：
```bash
# 使用curl或Kibana查询
curl -X GET "localhost:9200/your_index_name/_count"
```

或者在Python中：
```python
from es_client import es
result = es.count(index='your_index_name')
print(f"文档数: {result['count']}")
```

### Q2: 如何查看存储的文本内容？

A: 可以查询ES索引中的文档：
```python
from es_client import es
resp = es.search(index='your_index_name', body={
    "query": {"match_all": {}},
    "size": 1
})
doc = resp['hits']['hits'][0]['_source']
print(doc.get('wiki_content', '')[:500])  # 打印前500个字符
```

### Q3: 测评结果中的MRR、Hit@K是什么意思？

A: 
- **MRR (Mean Reciprocal Rank)**: 平均倒数排名，值越大越好（范围0-1）
- **Hit@K**: 在前K个结果中找到正确结果的比例，值越大越好（范围0-1）

### Q4: 如何对比添加wiki内容前后的效果？

A: 有两种方式：

**方式1: 使用自动对比脚本（推荐）**
```bash
# 1. 测评添加前
python evaluate_with_wiki_content.py --index your_index --output trainlog/before.json

# 2. 添加wiki内容
python store_wiki_content_to_es.py --index your_index

# 3. 测评添加后
python evaluate_with_wiki_content.py --index your_index --output trainlog/after.json

# 4. 自动对比
python compare_wiki_evaluation.py --before trainlog/before.json --after trainlog/after.json
```

**方式2: 手动对比JSON文件**
直接打开两个JSON文件，对比 `metrics` 部分的数值。

自动对比脚本会显示：
- 各项指标的绝对提升和相对提升百分比
- 提升的指标数量
- 总结性评价

### Q5: 为什么选择模式2而不是模式5？

A: 
- **模式2（es_text_only）**: 纯文本检索，不涉及向量和LLM，适合测评添加文本内容后的效果
- **模式5（vector_with_llm）**: 向量检索+LLM重排序，更复杂，适合最终的综合测评

本方案专注于测评**文本检索**的效果，因此使用模式2。

### Q6: wiki_content字段的搜索权重为什么是2？

A: wiki_content包含完整的wiki页面文本，信息量较大，因此给予较高权重（2），但低于label的权重（3），因为label是实体的核心标识。

## 相关文件

- `store_wiki_content_to_es.py`: 将文本内容存储到ES的脚本（不生成向量）
- `evaluate_with_wiki_content.py`: 使用模式2进行文本检索测评的脚本
- `compare_wiki_evaluation.py`: 对比添加wiki前后测评结果的脚本（新增）
- `download_wiki_content.py`: 下载wiki文本内容的脚本（如果还未下载）
- `search_vllm.py`: 包含模式2文本检索逻辑的主脚本

## 技术支持

如有问题，请检查：
1. ES服务是否正常运行
2. 文本文件格式是否正确（包含元数据头部）
3. 索引映射是否正确（wiki_content字段是否存在）
4. find.xlsx文件格式是否正确
