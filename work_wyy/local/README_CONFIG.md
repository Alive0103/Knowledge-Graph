# 配置文件说明

## 概述

`config.py` 是统一配置文件，管理所有路径、ES连接、数据源开关等配置。

## 主要配置项

### 1. 路径配置

- `WORK_DIR`: 工作目录（work_wyy）
- `DATA_DIR`: 数据目录
- `MODEL_DIR`: 模型目录
- `BASE_MODEL_PATH`: 基础模型路径
- `FINETUNED_MODEL_PATH`: 微调模型路径
- `TRAINLOG_DIR`: 日志目录

### 2. Elasticsearch 配置

#### ES连接模式

```python
ES_MODE = 'local'  # 'local' 或 'aliyun'
```

- **local**: 本地ES（Docker环境，通过服务名 `elasticsearch:9200` 访问）
- **aliyun**: 阿里云ES（需要认证）

#### 本地ES配置

```python
ES_LOCAL_CONFIG = {
    'url': 'http://elasticsearch:9200',
    'request_timeout': 180,
    'max_retries': 3,
    'retry_on_timeout': True,
    'basic_auth': None,  # 本地ES通常不需要认证
    'headers': None,
    'http_compress': False
}
```

#### 阿里云ES配置

```python
ES_ALIYUN_CONFIG = {
    'url': 'http://kgcode-xw7.public.cn-hangzhou.es-serverless.aliyuncs.com:9200',
    'username': 'kgcode-xw7',
    'password': 'Ln216812_',
    # ... 其他配置
}
```

### 3. 数据源开关配置

所有数据源都可以通过开关控制是否启用：

```python
DATA_SOURCE_SWITCHES = {
    # traindata目录（主要训练数据）
    'traindata': True,
    
    # ccks_ner目录（CCKS军事领域数据）
    'ccks_json': True,        # ccks_8_data_v2/train/*.json (400个文件)
    'ccks_validate': True,     # validate_data.json（如果有标注）
    'ccks_fold0': True,        # fold0/train (BIO格式)
    'ccks_fold1': True,        # fold1/train (BIO格式)
    'ccks_fold2': True,        # fold2/train (BIO格式)
    'ccks_fold3': True,        # fold3/train (BIO格式)
    'ccks_fold4': True,        # fold4/train (BIO格式)
    
    # train.txt（JSONL格式训练数据）
    'train_txt': True,
    
    # MSRA通用NER数据
    'msra_train': True,       # msra_train_bio.txt
    'msra_test': True,        # msra_test_bio.txt（作为验证集）
}
```

### 4. 模型训练配置

```python
TRAINING_CONFIG = {
    'max_length': 512,
    'batch_size': 8,
    'learning_rate': 2e-5,
    'num_epochs': 5,
    'warmup_steps': 100,
    'weight_decay': 0.01,
    'logging_steps': 50,
    'save_strategy': 'epoch',
    'save_total_limit': 3,
    'fp16': True,
}
```

### 5. 向量化配置

```python
VECTOR_DIMS = 1024
VECTOR_BATCH_SIZE = 64
USE_FINETUNED_FOR_VECTORIZATION = True
```

### 6. 实体提取配置

```python
MIN_TEXT_LENGTH = 2
MIN_ENTITY_LENGTH_ZH = 2
MIN_ENTITY_LENGTH_EN = 3
```

### 7. 检索系统配置

```python
ZHIPUAI_API_KEY = "your_api_key"
VECTOR_SEARCH_TOP_K = 30
LLM_RERANK_TOP_K = 30
```

## 使用方法

### 在代码中使用配置

```python
from config import (
    ES_INDEX_NAME,
    DATA_SOURCE_SWITCHES,
    BASE_MODEL_PATH,
    create_es_client
)

# 使用ES客户端
es = create_es_client()

# 检查数据源是否启用
if DATA_SOURCE_SWITCHES['traindata']:
    # 加载traindata数据
    pass

# 使用模型路径
model_path = BASE_MODEL_PATH
```

### 修改配置

直接编辑 `config.py` 文件即可修改配置。例如：

1. **切换ES连接方式**：
   ```python
   ES_MODE = 'local'  # 改为 'aliyun' 使用阿里云ES
   ```

2. **禁用某个数据源**：
   ```python
   DATA_SOURCE_SWITCHES = {
       'traindata': True,
       'msra_train': False,  # 禁用MSRA训练数据
       # ...
   }
   ```

3. **修改训练参数**：
   ```python
   TRAINING_CONFIG = {
       'num_epochs': 10,  # 增加训练轮数
       'batch_size': 16,  # 增大批次
       # ...
   }
   ```

## 注意事项

1. **路径配置**：所有路径都是相对于 `WORK_DIR`（work_wyy）的绝对路径，确保在不同工作目录下运行都能正确找到文件。

2. **ES连接**：切换 `ES_MODE` 后，系统会自动使用对应的连接配置。

3. **数据源开关**：禁用某个数据源后，训练时该数据源不会被加载，但不会影响已训练的模型。

4. **配置导入**：如果无法导入配置（例如在其他目录运行），代码会使用默认值，但建议始终在 `local` 目录下运行脚本。

