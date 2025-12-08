# 知识图谱实体链接系统 - 完整文档

## 📚 文档导航

本系统包含以下主要模块：

1. **[NER模型训练与使用](#ner模型训练与使用)** - 命名实体识别模型
2. **[向量检索系统](#向量检索系统)** - 5种检索方案对比
3. **[自动化流水线](#自动化流水线)** - 一键运行完整流程

---

## 🎯 NER模型训练与使用

### 快速开始

```bash
cd work_wyy/ner

# 1. 检查前置条件
python check_prerequisites.py

# 2. 训练模型
python finetune_ner_model.py

# 3. 测试模型
python diagnose_ner_model.py

# 4. 使用模型提取实体
python -c "from ner_extract_entities import extract_entities_from_text; print(extract_entities_from_text('阿利·伯克级驱逐舰装备有宙斯盾作战系统'))"
```

### 详细文档

- **基础文档**: [ner/README.md](./ner/README.md) - 快速入门和基本使用
- **完整文档**: [ner/训练与使用文档.md](./ner/训练与使用文档.md) - 详细训练指南、性能评估、集成方案

### 核心功能

- ✅ 支持27种军事领域实体类型识别
- ✅ 自动加载多种数据格式（JSON, BIO, CCKS格式）
- ✅ 使用微调后的Chinese-RoBERTa-Large模型
- ✅ F1-Score: 98.26%

---

## 🔍 向量检索系统

### 快速开始

```bash
cd work_wyy

# 评估所有5种检索方案（推荐）
python search_vllm.py

# 评估单个方案
python search_vllm.py --vector-only      # 方案1: 纯向量检索
python search_vllm.py --es-text-only     # 方案2: 纯ES文本搜索
python search_vllm.py --llm-only          # 方案3: 纯LLM判断
python search_vllm.py --vector-llm-always # 方案4: 向量+LLM（始终重排序）
python search_vllm.py --vector-llm       # 方案5: 向量+LLM（智能混合，推荐）
```

### 详细文档

- **检索方案对比**: [检索方案对比说明.md](./检索方案对比说明.md) - 5种方案的详细说明、优缺点、使用方法

### 5种检索方案

| 方案 | 模式名称 | 向量检索 | ES文本搜索 | LLM判断 | 特点 |
|------|---------|---------|-----------|---------|------|
| 方案1 | `vector_only` | ✅ | ❌ | ❌ | 纯向量检索，速度快 |
| 方案2 | `es_text_only` | ❌ | ✅ | ❌ | 纯文本搜索，不依赖向量 |
| 方案3 | `llm_only` | ❌ | ✅ | ✅ | 纯LLM语义判断 |
| 方案4 | `vector_with_llm_always` | ✅ | ❌ | ✅ | 始终使用LLM重排序 |
| 方案5 | `vector_with_llm` | ✅ | ✅ | ✅ | **智能混合模式（推荐）** |

**推荐使用**: 方案5（智能混合模式）- 自适应策略，兼顾效率和精度

---

## 🚀 自动化流水线

### 一键运行完整流程

```bash
cd work_wyy
python auto_pipeline.py
```

### 流水线步骤

1. **检查前置条件** - 验证环境、数据、模型
2. **训练NER模型** - 自动训练微调模型
3. **测试NER模型** - 验证模型效果
4. **提取实体词并向量化** - 使用NER模型从每条数据的描述中提取实体词，并用微调模型向量化
5. **向量化并存入ES** - 将处理后的数据存入ES（包含实体词向量）
6. **运行正式测试** - 评估所有5种检索方案

### 详细说明

自动化脚本会：
- ✅ 自动检查所有前置条件
- ✅ 训练NER模型（如果模型不存在）
- ✅ 测试模型效果
- ✅ 提示向量化步骤（需要手动配置数据源）
- ✅ 运行完整的检索系统评估（所有5种方案）

**注意**: 向量化步骤需要手动配置数据源路径，脚本会提示您完成。

---

## 📁 项目结构

```
work_wyy/
├── README.md                          # 本文档（主文档）
│
├── ner/                               # NER模型模块
│   ├── README.md                      # NER基础文档
│   ├── 训练与使用文档.md              # NER完整训练指南
│   ├── finetune_ner_model.py          # 模型训练脚本
│   ├── diagnose_ner_model.py          # 模型诊断脚本
│   ├── ner_extract_entities.py        # 实体提取核心模块
│   ├── check_prerequisites.py         # 前置条件检查
│   └── data_loader.py                 # 数据加载模块
│
├── vector/                            # 向量化模块
│   └── vector2ES.py                    # 向量化并存入ES
│
├── search_vllm.py                     # 向量检索系统（主程序）
├── vector_model.py                    # 统一向量生成模块
├── auto_pipeline.py                   # 自动化流水线
│
├── 检索方案对比说明.md                # 检索方案详细文档
│
├── model/                             # 模型目录
│   ├── chinese-roberta-wwm-ext-large/ # 基础模型
│   └── ner_finetuned/                 # 微调后的NER模型
│
└── data/                              # 数据目录
    ├── traindata/                     # 训练数据
    ├── ccks_ner/                      # CCKS数据集
    └── find.xlsx                      # 评测数据集
```

---

## 🔧 环境要求

### Python依赖

```bash
pip install torch transformers zhipuai elasticsearch pandas tqdm numpy
```

### 模型要求

- **基础模型**: `chinese-roberta-wwm-ext-large` (需下载到 `model/` 目录)
- **微调模型**: 运行 `finetune_ner_model.py` 自动生成

### 外部服务

- **Elasticsearch**: 用于向量存储和检索
- **智谱AI API**: 用于LLM重排序（需要API Key）

---

## 📊 性能指标

### NER模型性能

- **F1-Score**: 98.26%
- **Accuracy**: 98.22%
- **支持实体类型**: 27种军事领域实体

### 检索系统性能

运行 `python search_vllm.py` 查看所有5种方案的详细对比结果。

---

## 🎓 使用流程建议

### 首次使用

1. **准备环境**
   ```bash
   # 安装依赖
   pip install -r requirements.txt  # 如果有的话
   
   # 下载基础模型（如果还没有）
   # 模型应放在 model/chinese-roberta-wwm-ext-large/
   ```

2. **运行自动化流水线**
   ```bash
   cd work_wyy
   python auto_pipeline.py
   ```

3. **查看结果**
   - NER模型保存在: `model/ner_finetuned/`
   - 检索评估报告: `evaluation_summary_*.json`

### 日常使用

- **提取实体**: 使用 `ner/ner_extract_entities.py`
- **检索实体**: 使用 `search_vllm.py`（默认评估所有方案）
- **重新训练**: 运行 `ner/finetune_ner_model.py`

---

## 📝 更新日志

### 2025-12-08

- ✅ 实现5种检索方案对比
- ✅ 创建检索方案对比说明文档
- ✅ 更新自动化流水线脚本
- ✅ 整合所有文档到主README

### 2025-12-07

- ✅ 实现智能混合检索模式
- ✅ 优化向量生成模块
- ✅ 完善NER模型文档

---

## 🔗 相关文档

- [NER模型训练与使用文档](./ner/训练与使用文档.md)
- [检索方案对比说明](./检索方案对比说明.md)
- [NER基础文档](./ner/README.md)

---

## ❓ 常见问题

### Q: 如何选择检索方案？

**A**: 推荐使用方案5（智能混合模式），它会自动判断使用向量检索还是ES文本搜索，兼顾效率和精度。

### Q: 向量化步骤为什么需要手动？

**A**: 因为数据源路径需要根据实际情况配置，脚本会提示您完成。

### Q: 如何查看详细的评估结果？

**A**: 运行 `search_vllm.py` 后会生成 `evaluation_summary_*.json` 文件，包含所有方案的详细对比。

### Q: NER模型训练需要多长时间？

**A**: 根据数据量和硬件配置，通常需要10-30分钟。

---

## 📧 联系方式

如有问题，请查看相关模块的详细文档或检查日志文件。

---

**文档版本**: v1.0  
**最后更新**: 2025-12-08

