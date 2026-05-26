# NER实体识别模块

## 📚 文档导航

- **[训练与使用文档.md](./训练与使用文档.md)** - **完整训练与使用指南**（推荐阅读）
  - 训练数据详细说明
  - 训练效果评估（F1-Score: 98.26%）
  - 与向量检索系统集成方案
  - 后续优化建议

## 🔄 重要更新（2025-12-08）

- **向量生成已更新**: 所有向量生成代码（`search_vllm.py`、`vector2ES.py`）现在**优先使用微调后的NER模型**进行向量生成
- **统一向量生成模块**: 新增 `vector_model.py` 模块，统一管理向量生成，支持从微调后的NER模型中提取encoder
- **自动化流水线**: 新增 `auto_pipeline.py` 脚本，自动化执行：训练 → 测试 → 向量化 → 存入ES → 正式测试

## 概述

本模块实现了基于BERT的命名实体识别（NER）功能，用于从文本中提取军事领域相关的实体词（如"宙斯盾作战系统"、"AN/SPY-13D相控阵雷达"、"阿利·伯克级驱逐舰"等）。该模块使用Chinese-RoBERTa作为基础模型，支持在军事领域NER数据上进行微调以提升识别效果。

**训练效果**: F1-Score **98.26%**，支持**27种实体类型**，训练数据**3,399条**。

## 文件说明

### 核心代码文件

1. **`ner_extract_entities.py`** - NER实体提取核心模块
   - **作用**: 从文本中提取实体词的核心功能模块
   - **主要函数**:
     - `load_ner_model()`: 加载NER模型（优先使用微调模型，如果不存在则使用基础模型）
     - `extract_entities_from_text()`: 从文本中提取实体词列表
     - `get_entity_words_from_text()`: 提取实体词并返回词频字典格式
   - **模型路径**: 自动定位到 `./../model/chinese-roberta-wwm-ext-large` 或 `./../model/ner_finetuned`（相对于脚本所在目录）
   - **使用方式**: 可独立使用，也可被其他模块导入调用

2. **`finetune_ner_model.py`** - NER模型微调脚本
   - **作用**: 在军事领域NER数据上微调Chinese-RoBERTa模型
   - **功能**:
     - 支持多种数据格式（JSON数组、BIO格式、CCKS格式）
     - 自动从多个数据目录加载数据（traindata、ccks_ner、nlp_datasets）
     - 将字符级标签对齐到token级标签（处理BERT子词切分问题）
     - 动态构建标签映射（支持多种实体类型）
     - 使用HuggingFace Trainer进行模型训练
     - 支持训练/验证/测试集划分
     - 自动保存最佳模型和标签映射信息
   - **数据目录**: `./../data/`（相对于脚本所在目录）
     - `traindata/`: JSON数组格式的训练数据
     - `ccks_ner/militray/`: CCKS军事领域NER数据
     - `nlp_datasets/ner/msra/`: MSRA BIO格式数据
   - **输出路径**: `./../model/ner_finetuned`（相对于脚本所在目录）
   - **训练参数**:
     - 最大序列长度: 512
     - 批次大小: 8
     - 学习率: 2e-5
     - 训练轮数: 5 epochs
     - 自动使用GPU（如果可用）

3. **`diagnose_ner_model.py`** - NER模型诊断脚本
   - **作用**: 分析微调后的NER模型效果，诊断问题
   - **功能**:
     - 分析训练数据的统计信息（样本数、实体数、实体类型分布等）
     - 检查标签对齐的准确性
     - 测试模型预测结果（显示每个token的预测标签和概率）
     - 提供改进建议

4. **`check_prerequisites.py`** - 前置条件检查脚本
   - **作用**: 在训练前检查所有必要的条件和数据
   - **功能**:
     - 检查基础模型是否存在且完整
     - 检查训练数据是否存在且格式正确
     - 检查CCKS数据是否可访问
     - 检查Python依赖是否已安装
     - 检查CUDA是否可用
     - 统计数据信息（样本数、实体类型等）
   - **使用**: 在运行微调脚本前先运行此脚本

5. **`data_loader.py`** - 数据加载模块
   - **作用**: 统一处理多种格式的NER数据加载
   - **功能**:
     - 支持JSON数组格式
     - 支持BIO格式（MSRA格式）
     - 支持CCKS JSON格式
     - 支持CCKS BIO格式
     - 自动检测文件编码（UTF-8, GBK, GB2312等）
     - 从多个数据目录自动加载数据

## 技术思路

### 1. 模型架构

- **基础模型**: Chinese-RoBERTa-wwm-ext-large
- **任务类型**: Token分类（Token Classification）
- **标签格式**: BIO标注格式
  - `O`: 非实体
  - `B-ENTITY`: 实体开始
  - `I-ENTITY`: 实体内部

### 2. 标签对齐方法

由于BERT使用WordPiece分词，会将中文文本切分成多个子词（subword），需要将字符级标签对齐到token级标签。本模块使用以下方法：

1. **字符级标签创建**: 根据训练数据中的实体位置（start, end），创建字符级标签序列
2. **Token匹配**: 对每个token，在文本中搜索其对应的字符位置
3. **标签映射**: 将找到位置的字符标签映射到token
4. **特殊处理**: 
   - 特殊token（[CLS], [SEP], [PAD]）标记为O
   - 子词token（以##开头）继承主词的标签
   - 如果找不到精确匹配，使用当前位置的标签

### 3. 训练流程

1. **数据加载**: 从JSONL文件加载训练数据，每条数据包含：
   ```json
   {
     "content": "文本内容",
     "result_list": [
       {"text": "实体文本", "start": 开始位置, "end": 结束位置}
     ]
   }
   ```

2. **标签对齐**: 将字符级标签转换为token级标签

3. **模型训练**: 使用HuggingFace Trainer进行训练
   - 自动划分验证集（如果没有提供）
   - 每个epoch结束后评估并保存最佳模型
   - 支持混合精度训练（FP16）

4. **模型保存**: 保存微调后的模型和tokenizer到指定目录

### 4. 实体提取流程

1. **文本Tokenization**: 使用tokenizer将文本转换为token序列
2. **模型预测**: 对每个token预测标签（O/B-ENTITY/I-ENTITY）
3. **实体合并**: 将连续的B-ENTITY和I-ENTITY标签合并为完整实体
4. **去重**: 去除重复的实体

## 使用说明

### 快速开始（使用基础模型）

如果暂时跳过微调，直接使用基础模型进行NER提取：

```python
from ner.ner_extract_entities import extract_entities_from_text

# 提取实体
text = "阿利·伯克级驱逐舰装备有宙斯盾作战系统和AN/SPY-13D相控阵雷达。"
entities = extract_entities_from_text(text)
print(entities)  # ['阿利·伯克级驱逐舰', '宙斯盾作战系统', 'AN/SPY-13D相控阵雷达']
```

### 完整流程（包含微调）

#### 步骤1: 准备训练数据

脚本支持从多个数据目录自动加载数据，支持多种数据格式：

**1. traindata目录（JSON数组格式）**
- 路径：`./../data/traindata/`
- 格式：
```json
[
  {
    "text": "文本内容",
    "entities": [
      {
        "start": 0,
        "end": 5,
        "text": "实体文本",
        "type": "实体类型"
      }
    ],
    "sample_id": 1
  }
]
```
- 文件命名：`*_ner_train.json`（训练）、`*_ner_dev.json`（验证）

**2. ccks_ner目录（CCKS格式）**
- 路径：`./../data/ccks_ner/militray/PreModel_Encoder_CRF/ccks_8_data_v2/`
- 格式：单个JSON文件，包含text和entities字段
- 文件：`train/*.json`（训练）、`validate_data.json`（验证）

**3. nlp_datasets目录（BIO格式）**
- 路径：`./../data/nlp_datasets/ner/msra/`
- 格式：每行"字符\t标签"（MSRA格式）
- 文件：`msra_train_bio.txt`（训练）、`msra_test_bio.txt`（验证）

**支持的实体类型**：
- 火炮、军工企业、军用舰艇、军事组织、枪械
- 军用航空器、军用车辆、武器系统、导弹、信息系统
- 地缘政治实体、军事系统、无人机、弹药、军事地点
- 以及其他在数据中出现的实体类型（会自动识别和合并）

**注意**：脚本会自动从所有存在的目录加载数据并合并，无需手动配置。

#### 步骤2: 微调模型

```bash
cd work_wyy/ner
python finetune_ner_model.py
```

**训练参数可在脚本中修改**：
- `NUM_EPOCHS`: 训练轮数（默认5）
- `BATCH_SIZE`: 批次大小（默认8）
- `LEARNING_RATE`: 学习率（默认2e-5）
- `MAX_LENGTH`: 最大序列长度（默认512）

**新功能**：
- 支持多种数据格式（JSON数组、BIO格式、CCKS格式）
- 自动从多个数据目录加载数据（traindata、ccks_ner、nlp_datasets）
- 自动识别数据中的所有实体类型
- 动态构建BIO标签映射（支持多种实体类型）
- 自动合并多个训练文件
- 保存标签映射信息到模型目录

#### 步骤3: 使用微调后的模型

微调完成后，`extract_entities_from_text()` 会自动优先使用微调后的模型。

### 诊断模型效果

如果模型效果不佳，可以运行诊断脚本：

```bash
cd work_wyy/ner
python diagnose_ner_model.py
```

这会显示：
- 训练数据的统计信息
- 标签对齐的准确性
- 模型预测的详细输出

### 检查前置条件

在训练前，运行前置条件检查：

```bash
cd work_wyy/ner
python check_prerequisites.py
```

这会检查所有必要的条件和数据，确保训练可以正常进行。

## 效果说明

### 未微调模型

- **效果**: 较差，可能无法准确识别军事领域实体
- **原因**: 基础模型未在军事领域数据上训练
- **适用场景**: 快速测试或通用实体识别

### 微调后模型

- **效果**: 显著提升，能够准确识别各种军事实体
- **识别类型**: 
  - 军工企业（如"德国莱茵金属防务公司"）
  - 军事组织（如"美国海军"）
  - 武器装备（如"M777型155毫米超轻牵引榴弹炮"）
  - 导弹（如"战斧巡航导弹"）
  - 军用航空器（如"F-22战机"）
  - 信息系统（如"宙斯盾作战系统"）
  - 舰艇（如"阿利·伯克级驱逐舰"）

### 性能指标

- **训练时间**: 
  - 有GPU: 30分钟-几小时（取决于数据量）
  - 无GPU: 数小时-一天
- **推理速度**: 
  - 有GPU: 约100-200条/秒
  - 无GPU: 约10-20条/秒
- **准确率**: 取决于训练数据质量和数量（建议至少1000+条数据）

## 注意事项

1. **模型路径**: 
   - 确保 `./../model/chinese-roberta-wwm-ext-large` 存在（相对于脚本所在目录）
   - 代码使用相对路径，如果路径不对，检查文件位置

2. **训练数据**: 
   - 确保 `./../data/nerdata/train.txt` 存在（相对于脚本所在目录）
   - 数据格式必须正确（JSONL格式，每行一个JSON对象）

3. **GPU支持**: 
   - 如果有GPU，训练和推理会自动使用GPU加速
   - 没有GPU也可以运行，但速度较慢

4. **内存要求**: 
   - 建议至少8GB内存
   - 使用GPU时建议至少6GB显存
   - 如果内存不足，可以减小 `BATCH_SIZE` 或 `MAX_LENGTH`

5. **数据格式**: 
   - 支持多种数据格式（JSON数组、BIO格式、CCKS格式）
   - 自动检测文件编码（UTF-8, GBK, GB2312等）
   - 如果遇到编码错误，代码会自动尝试其他编码

## 故障排除

### 模型加载失败

- 运行 `python check_ner_path.py` 检查模型路径
- 确保模型文件完整（config.json, pytorch_model.bin, tokenizer.json等）

### NER提取失败

- 系统会自动回退到词频统计方法
- 检查日志中的错误信息
- 如果基础模型加载失败，检查模型文件是否完整

### 训练失败

- 检查训练数据格式是否正确
- 确保训练数据文件存在
- 检查是否有足够的磁盘空间保存模型
- 如果显存不足，减小 `BATCH_SIZE`

### 编码错误

- 如果遇到 `UnicodeDecodeError`，代码会自动尝试多种编码
- 如果所有编码都失败，会在日志中记录错误并跳过该文件
- 检查日志文件 `ner_finetune.log` 查看详细错误信息

## 文件结构

```
work_wyy/
├── ner/
│   ├── ner_extract_entities.py      # NER实体提取核心模块
│   ├── finetune_ner_model.py        # NER模型微调脚本
│   ├── diagnose_ner_model.py        # NER模型诊断脚本
│   ├── check_prerequisites.py       # 前置条件检查脚本
│   ├── data_loader.py               # 数据加载模块（统一处理多种格式）
│   ├── README.md                     # 本文档
│   └── 训练与使用文档.md            # 完整训练与使用指南
├── vector/
│   └── vector2ES.py                  # 向量化数据并存入ES（已更新使用微调模型）
├── vector_model.py                  # 统一向量生成模块（支持微调后的模型）
├── auto_pipeline.py                 # 自动化流水线脚本
├── search_vllm.py                  # 向量检索系统（已更新使用微调模型）
├── model/
│   ├── chinese-roberta-wwm-ext-large/  # 基础模型
│   └── ner_finetuned/                 # 微调后的NER模型（用于NER和向量生成）
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer.json
│       └── label_mapping.json         # 标签映射文件
└── data/
    ├── traindata/                     # 训练数据目录（JSON格式）
    │   ├── *_ner_train.json          # 训练文件
    │   └── *_ner_dev.json            # 验证文件
    ├── ccks_ner/                     # CCKS NER数据集
    │   └── militray/                 # 军事领域数据
    │       └── PreModel_Encoder_CRF/
    │           └── ccks_8_data_v2/
    │               ├── train/        # 训练数据（JSON文件）
    │               └── validate_data.json  # 验证数据
    └── nlp_datasets/                 # NLP数据集
        └── ner/
            └── msra/                 # MSRA数据集（已禁用）
                ├── msra_train_bio.txt  # 训练数据（BIO格式）
                └── msra_test_bio.txt   # 测试数据（BIO格式）
```

## 更新日志

- **2025-12-03**: 
  - 初始版本，支持NER实体提取和微调
  - 添加标签对齐验证功能
  - 添加模型诊断功能
  - 统一文档说明
- **2025-12-03 (更新)**:
  - 支持多种数据格式（JSON数组、BIO格式、CCKS格式）
  - 支持从多个数据目录自动加载数据（traindata、ccks_ner、nlp_datasets）
  - 动态构建标签映射，支持多种实体类型
  - 添加数据集获取指南文档
- **2025-12-08 (重构)**:
  - 重构代码，将数据加载逻辑提取到 `data_loader.py` 模块
  - 添加 `check_prerequisites.py` 前置条件检查脚本
  - 删除重复文件（`check_entity_types.py`, `verify_label_alignment.py`）
  - 完善编码错误处理，自动检测文件编码
  - 优化代码结构，提高可维护性
- **2025-12-08 (文档完善)**:
  - 创建完整的训练与使用文档，包含训练数据说明、效果评估、集成方案
  - 训练完成：F1-Score 98.26%，支持27种实体类型
  - 添加与 `search_vllm.py` 向量检索系统的集成指南
- **2025-12-08 (向量生成更新)**:
  - **重要**: 所有向量生成代码已更新，优先使用微调后的NER模型
  - 新增 `vector_model.py` 统一向量生成模块，支持从NER模型中提取encoder用于向量生成
  - 更新 `search_vllm.py` 和 `vector/vector2ES.py` 使用新的向量生成模块
  - 新增 `auto_pipeline.py` 自动化流水线脚本
  - 删除临时文档（数据格式分析、数据集获取指南）

