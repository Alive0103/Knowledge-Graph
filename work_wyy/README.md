# 知识图谱实体链接系统

## 📚 快速导航

- **[完整使用指南](./完整使用指南.md)** ⭐ **推荐阅读** - 从头开始的完整使用教程
- [NER模型训练与使用](#ner模型训练与使用)
- [向量检索系统](#向量检索系统)
- [自动化流水线](#自动化流水线)

---

## 🚀 快速开始

### 第一次运行（完整流程）

```bash
# 1. 进入项目目录
cd work_wyy

# 2. 运行自动化流水线（一键完成所有步骤）
python auto_pipeline.py
```

### 后续运行

```bash
# 如果模型已训练，可以从提取实体词开始
python auto_pipeline.py --from extract

# 或者只运行检索测试
python auto_pipeline.py --from final_test
```

**详细步骤请参考**: [完整使用指南](./完整使用指南.md)

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
```

### 核心功能

- ✅ 支持30种实体类型识别（27种军事领域 + 3种通用NER）
- ✅ 自动加载多种数据格式（JSON, BIO, CCKS格式, JSONL）
- ✅ 使用微调后的Chinese-RoBERTa-Large模型
- ✅ F1-Score: 98.26%

### 详细文档

- **基础文档**: [ner/README.md](./ner/README.md) - 快速入门
- **完整文档**: [ner/训练与使用文档.md](./ner/训练与使用文档.md) - 详细训练指南、性能评估

### 训练数据

系统会自动加载所有可用数据：
- ✅ traindata目录（6组训练/验证文件）
- ✅ ccks_ner（400个JSON文件 + 5个fold的BIO数据）
- ✅ train.txt（JSONL格式）
- ✅ MSRA数据（通用NER）

**总训练数据量**: 约 51,338 条

---

## 🔍 向量检索系统

### 快速开始

```bash
cd work_wyy

# 评估所有5种检索方案（推荐）
python search_vllm.py

# 评估单个方案
python search_vllm.py --vector-llm  # 方案5：智能混合模式（推荐）
```

### 5种检索方案

| 方案 | 模式名称 | 特点 | 推荐度 |
|------|---------|------|--------|
| 方案1 | `vector_only` | 纯向量检索，速度快 | ⭐⭐⭐ |
| 方案2 | `es_text_only` | 纯文本搜索，不依赖向量 | ⭐⭐ |
| 方案3 | `llm_only` | 纯LLM语义判断 | ⭐⭐⭐ |
| 方案4 | `vector_with_llm_always` | 始终使用LLM重排序 | ⭐⭐⭐⭐ |
| 方案5 | `vector_with_llm` | **智能混合模式** | ⭐⭐⭐⭐⭐ **推荐** |

**推荐使用**: 方案5（智能混合模式）- 自适应策略，兼顾效率和精度

### 详细文档

- **检索方案对比**: [检索方案对比说明.md](./检索方案对比说明.md) - 5种方案的详细说明

---

## 🚀 自动化流水线

### 一键运行完整流程

```bash
cd work_wyy
python auto_pipeline.py
```

### 流水线步骤

1. **check** - 检查前置条件（环境、数据、模型）
2. **train** - 训练NER模型（自动加载所有数据）
3. **test** - 测试NER模型（验证效果，失败则终止）
4. **extract** - 提取实体词并向量化（使用NER模型）
5. **vectorize** - 向量化并存入ES（生成所有向量字段）
6. **final_test** - 运行正式测试（评估所有5种检索方案）

### 灵活执行

```bash
# 从指定阶段开始
python auto_pipeline.py --from test        # 从测试开始
python auto_pipeline.py --from extract    # 从提取实体词开始
python auto_pipeline.py --from final_test # 只运行最终测试

# 跳过指定阶段
python auto_pipeline.py --skip check      # 跳过前置检查
python auto_pipeline.py --skip train      # 跳过训练

# 列出所有阶段
python auto_pipeline.py --list-steps
```

### 日志文件

所有日志保存在 `trainlog/` 目录：
- 控制台输出日志
- 详细执行日志
- 评估报告（JSON格式）

---

## 📁 项目结构

```
work_wyy/
├── README.md                          # 本文档（主文档）
├── 完整使用指南.md                    # ⭐ 完整使用教程
├── 检索方案对比说明.md                # 检索方案详细文档
│
├── ner/                               # NER模型模块
│   ├── README.md                      # NER基础文档
│   ├── 训练与使用文档.md              # NER完整训练指南
│   ├── finetune_ner_model.py          # 模型训练脚本
│   ├── diagnose_ner_model.py          # 模型诊断脚本
│   ├── ner_extract_entities.py        # 实体提取核心模块
│   ├── check_prerequisites.py         # 前置条件检查
│   └── data_loader.py                 # 数据加载模块（支持所有数据源）
│
├── vector/                            # 向量化模块
│   └── vector2ES.py                   # 向量化并存入ES
│
├── data/                              # 数据目录
│   ├── traindata/                     # NER训练数据
│   ├── ccks_ner/                      # CCKS数据集
│   ├── nlp_datasets/                  # MSRA通用NER数据
│   ├── train.txt                      # JSONL格式训练数据
│   ├── zh_wiki_v2.jsonl              # 中文维基数据
│   └── en_wiki_v3.jsonl              # 英文维基数据
│
├── search_vllm.py                     # 向量检索系统（主程序）
├── vector_model.py                    # 统一向量生成模块
├── auto_pipeline.py                   # 自动化流水线
├── es_client.py                       # Elasticsearch客户端配置
│
├── model/                             # 模型目录
│   ├── chinese-roberta-wwm-ext-large/ # 基础模型
│   └── ner_finetuned/                 # 微调后的NER模型（训练后生成）
│
└── trainlog/                          # 日志目录（自动创建）
    ├── auto_pipeline_*.log            # 流水线日志
    ├── extract_entity_words_*.log     # 实体提取日志
    ├── search_vllm_console_*.log      # 检索系统日志
    └── evaluation_*.json              # 评估报告
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

- **Elasticsearch**: 用于向量存储和检索（配置在 `es_client.py`）
- **智谱AI API**: 用于LLM重排序（配置在 `search_vllm.py`）

---

## 📊 性能指标

### NER模型性能

- **F1-Score**: 98.26%
- **Accuracy**: 98.22%
- **支持实体类型**: 30种（27种军事领域 + 3种通用NER）
- **训练数据量**: 约 51,338 条

### 检索系统性能

运行 `python search_vllm.py` 查看所有5种方案的详细对比结果。

---

## 📝 使用流程建议

### 首次使用

1. **准备环境**
   ```bash
   # 安装依赖
   pip install torch transformers zhipuai elasticsearch pandas tqdm numpy
   
   # 下载基础模型（如果还没有）
   # 模型应放在 model/chinese-roberta-wwm-ext-large/
   ```

2. **配置服务**
   - 编辑 `es_client.py` 配置Elasticsearch连接
   - 编辑 `search_vllm.py` 配置智谱AI API Key

3. **运行自动化流水线**
   ```bash
   cd work_wyy
   python auto_pipeline.py
   ```

4. **查看结果**
   - NER模型保存在: `model/ner_finetuned/`
   - 检索评估报告: `trainlog/evaluation_summary_*.json`
   - 所有日志: `trainlog/` 目录

### 日常使用

- **提取实体**: 使用 `ner/ner_extract_entities.py`
- **检索实体**: 使用 `search_vllm.py`（默认评估所有方案）
- **重新训练**: 运行 `ner/finetune_ner_model.py` 或 `auto_pipeline.py --from train`

---

## ❓ 常见问题

### Q: 如何选择检索方案？

**A**: 推荐使用方案5（智能混合模式），它会自动判断使用向量检索还是ES文本搜索，兼顾效率和精度。

### Q: 训练需要多长时间？

**A**: 根据数据量和硬件配置：
- GPU：约30-60分钟（数据量约51,000条）
- CPU：约2-4小时

### Q: 如何查看详细的评估结果？

**A**: 运行 `search_vllm.py` 后会生成 `trainlog/evaluation_summary_*.json` 文件，包含所有方案的详细对比。

### Q: 实体提取效果不好怎么办？

**A**: 
- 检查训练数据质量
- 尝试增加训练轮数
- 查看 `ner/训练与使用文档.md` 中的优化建议

---

## 🔗 相关文档

- **[完整使用指南](./完整使用指南.md)** ⭐ **推荐阅读**
- [NER训练与使用文档](./ner/训练与使用文档.md)
- [检索方案对比说明](./检索方案对比说明.md)
- [NER基础文档](./ner/README.md)

---

## 📝 更新日志

### 2025-12-08

- ✅ 启用所有可用训练数据（fold0-4、train.txt、MSRA）
- ✅ 数据量从3,399条增加到51,338条（增长15倍）
- ✅ 实现5种检索方案对比
- ✅ 创建完整使用指南
- ✅ 整理文档，删除不必要的文件
- ✅ 优化自动化流水线，支持灵活执行

### 2025-12-07

- ✅ 实现智能混合检索模式
- ✅ 优化向量生成模块
- ✅ 完善NER模型文档

---

**文档版本**: v2.0  
**最后更新**: 2025-12-08
