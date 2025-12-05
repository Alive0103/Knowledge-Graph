# NER实体识别模块

## 概述

本模块实现了基于BERT的命名实体识别（NER）功能，用于从文本中提取军事领域相关的实体词（如"宙斯盾作战系统"、"AN/SPY-13D相控阵雷达"、"阿利·伯克级驱逐舰"等）。该模块使用Chinese-RoBERTa作为基础模型，支持在军事领域NER数据上进行微调以提升识别效果。

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
     - 加载NER训练数据（JSONL格式，包含实体位置标注）
     - 将字符级标签对齐到token级标签（处理BERT子词切分问题）
     - 使用HuggingFace Trainer进行模型训练
     - 支持训练/验证/测试集划分
     - 自动保存最佳模型
   - **训练数据路径**: `./../data/nerdata/train.txt`（相对于脚本所在目录）
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

4. **`verify_label_alignment.py`** - 标签对齐验证脚本
   - **作用**: 验证字符级标签到token级标签的映射是否正确
   - **功能**:
     - 检查训练数据中的实体位置是否正确映射到token标签
     - 统计标签对齐的准确率
     - 显示错误对齐的示例

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

确保训练数据位于：`./../data/nerdata/train.txt`（相对于脚本所在目录）

数据格式要求：
```json
{
  "content": "文本内容",
  "result_list": [
    {"text": "实体文本", "start": 开始位置, "end": 结束位置, "url": "链接（可选）"}
  ],
  "prompt": "实体类型（可选）"
}
```

#### 步骤2: 微调模型

```bash
cd work_wyy/ner
python finetune_ner_model.py
```

训练参数可在脚本中修改：
- `NUM_EPOCHS`: 训练轮数（默认5）
- `BATCH_SIZE`: 批次大小（默认8）
- `LEARNING_RATE`: 学习率（默认2e-5）

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

### 验证标签对齐

在训练前，可以验证标签对齐是否正确：

```bash
cd work_wyy/ner
python verify_label_alignment.py
```

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

5. **标签对齐**: 
   - 标签对齐是NER训练的关键步骤
   - 如果对齐不准确，模型学习的是错误的标签
   - 建议在训练前运行 `verify_label_alignment.py` 检查对齐准确性

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

### 标签对齐错误

- 运行 `verify_label_alignment.py` 检查对齐准确性
- 如果对齐错误率高，可能需要优化对齐方法

## 文件结构

```
work_wyy/
├── ner/
│   ├── ner_extract_entities.py      # NER实体提取核心模块
│   ├── finetune_ner_model.py        # NER模型微调脚本
│   ├── diagnose_ner_model.py        # NER模型诊断脚本
│   ├── verify_label_alignment.py    # 标签对齐验证脚本
│   └── README.md                     # 本文档
├── model/
│   ├── chinese-roberta-wwm-ext-large/  # 基础NER模型
│   └── ner_finetuned/                 # 微调后的模型（训练后生成）
└── data/
    └── nerdata/
        ├── train.txt                  # 训练数据
        ├── dev.txt                    # 验证数据（可选）
        └── test.txt                   # 测试数据（可选）
```

## 更新日志

- **2025-12-03**: 
  - 初始版本，支持NER实体提取和微调
  - 添加标签对齐验证功能
  - 添加模型诊断功能
  - 统一文档说明

