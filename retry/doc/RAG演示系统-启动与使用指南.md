# RAG 演示系统 — 启动与使用指南

> 基于 Yuxi 框架改造，展示实体链接、跨语言实体对齐与知识图谱数据

---

## 一、系统架构

```
浏览器 (localhost:3000)
  │
  ├─ /              首页（项目介绍 + 导航卡片）
  ├─ /chat          对话查询（直接调 kg-rag API）
  ├─ /database      知识图谱浏览（图谱可视化、实体详情、查询）
  └─ /dashboard     实验结果展示（指标表格 + ECharts 图表）
        │
        ▼
  Vite Dev Server ──proxy──▶ FastAPI 后端 (localhost:5050)
                                │
                  ┌─────────────┼─────────────┐
                  ▼             ▼             ▼
             PostgreSQL    Elasticsearch   Knowledge-Graph
           (localhost:5432) (localhost:9200)  (retry/kg_rag)
```

### 技术栈

| 层 | 技术 |
|---|---|
| 前端 | Vue 3.5 + Ant Design Vue + AntV G6 + ECharts |
| 后端 | FastAPI + SQLAlchemy (async) + kg_rag_bridge |
| 数据库 | PostgreSQL 16 (Docker) |
| 搜索引擎 | Elasticsearch 9.x (Docker) |
| 知识图谱 | Knowledge-Graph/retry/kg_rag 模块 |

---

## 二、前置条件

1. **Docker Desktop** 已安装并运行
2. **Node.js 20+**（推荐通过 nvm 管理）
   ```bash
   nvm install 22.16.0
   nvm use 22.16.0
   ```
3. **Python 3.12+**（Anaconda 或系统 Python 均可）
4. **uv 包管理器**（已安装到 `.venv`）

---

## 三、启动步骤

### 3.1 启动基础服务（Docker）

```bash
cd D:\work\毕设\知识图谱\Yuxi

# PostgreSQL（首次运行会自动创建数据库）
docker compose up -d postgres

# Elasticsearch（如果尚未运行）
# 项目中已有 kg-elasticsearch 容器，确认运行状态：
docker ps | grep elasticsearch
# 如果没有运行：
docker start kg-elasticsearch
```

验证：
```bash
# PostgreSQL
docker exec postgres pg_isready -U postgres -d yuxi_know

# Elasticsearch（应返回 5795 条文档）
curl http://localhost:9200/data2/_count
```

### 3.2 启动后端

```bash
cd D:\work\毕设\知识图谱\Yuxi\backend

# 设置环境变量并启动
# Windows PowerShell:
$env:PYTHONPATH="package"
$env:SILICONFLOW_API_KEY="demo-placeholder"
$env:LITE_MODE="true"
$env:POSTGRES_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/yuxi_know"
$env:REDIS_URL="redis://localhost:6379/0"
.venv\Scripts\python.exe server\main.py

# 或 Git Bash / WSL:
PYTHONPATH=package \
SILICONFLOW_API_KEY=demo-placeholder \
LITE_MODE=true \
POSTGRES_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/yuxi_know" \
REDIS_URL="redis://localhost:6379/0" \
.venv/Scripts/python.exe server/main.py
```

看到以下输出即启动成功：
```
Yuxi backend startup complete
INFO: Uvicorn running on http://0.0.0.0:5050
```

验证：
```bash
curl http://localhost:5050/api/system/health
# 应返回 {"status":"ok","message":"服务正常运行","version":"0.6.0"}
```

### 3.3 启动前端

新开一个终端：
```bash
cd D:\work\毕设\知识图谱\Yuxi\web

# 确保使用 Node 22+
nvm use 22.16.0

# 启动开发服务器
npx vite --port 3000
```

看到以下输出即启动成功：
```
VITE v7.3.1  ready in xxx ms
➜  Local:   http://localhost:3000/
```

### 3.4 打开浏览器

访问 **http://localhost:3000**

---

## 四、页面功能说明

### 4.1 首页 `/`

项目介绍页，包含三张导航卡片：
- **对话查询** → 跳转到 `/chat`
- **知识图谱浏览** → 跳转到 `/database`
- **实验结果** → 跳转到 `/dashboard`

### 4.2 实验结果 `/dashboard`

展示完整的实验评测指标，数据来源于 `Knowledge-Graph/retry/output/` 下的 JSON 结果文件。

包含四个区域：
1. **概览卡片** — NER F1、最佳 MRR、最佳 Hits@1、数据集信息
2. **实体链接评测** — NER 训练结果 + Text-only vs Vector-only 检索对比表与柱状图
3. **实体对齐方法对比** — 四种方法（Raw LaBSE / LaBSE+Graph / Raw BGE-M3 / BGE-M3+Graph）的 MRR、Hits@1/5/10 对比表与分组柱状图
4. **对齐前后效果对比** — LaBSE 线和 BGE-M3 线的提升量对比表与柱状图

### 4.3 知识图谱浏览 `/database`

点击 **"军事装备知识图谱"** 进入知识库详情页，包含以下标签页：

- **图谱 (graph)** — 交互式知识图谱可视化
  - 搜索框输入实体名，如 "装甲车辆"、"M1 Abrams"
  - 点击节点查看详情面板
  - 可调整最大节点数和搜索深度
- **查询 (query)** — 结构化查询测试
  - 支持实体查询、关系查询、三元组查询、混合查询
  - 返回实体列表、关系、三元组、对齐扩展、证据块
- **评测 (evaluation)** — RAG 评测功能

点击实体可跳转到 **实体详情页** `/database/:id/entity/:key`，展示：
- 实体基本信息（名称、ID、所属图谱）
- 关联关系与样例三元组
- 跨语言对齐实体（含分数和证据）
- 实体链接证据块

### 4.4 对话查询 `/chat`

简化的 RAG 对话界面：
- 输入实体名或自然语言查询
- 系统调用 kg-rag API 返回结构化结果
- 实体名可点击跳转到详情页

示例查询：
- `装甲车辆`
- `M1 Abrams`
- `导弹`
- `T-72`

---

## 五、环境变量说明

在 `Yuxi/.env` 中配置（已预设）：

| 变量 | 说明 | 默认值 |
|---|---|---|
| `LITE_MODE` | 精简模式，跳过 Milvus/Neo4j/LightRAG | `true` |
| `SILICONFLOW_API_KEY` | LLM 提供商 API Key（演示占位） | `demo-placeholder` |
| `POSTGRES_URL` | PostgreSQL 连接串 | `postgresql+asyncpg://postgres:postgres@localhost:5432/yuxi_know` |
| `KG_RAG_REPO_ROOT` | Knowledge-Graph 项目根目录 | 自动推断 |
| `KG_RAG_RESULTS_DIR` | 实验结果 JSON 目录 | `retry/output/complete_supervised_retrain_...` |
| `KG_RAG_COMPARISON_DIR` | 对齐对比 JSON 目录 | `retry/output/experiment_comparison` |

---

## 六、常见问题

### Q: 后端启动报 "No model provider available"
**A:** 确保设置了 `SILICONFLOW_API_KEY=demo-placeholder`（或任意非空值）。

### Q: 后端启动报 psycopg ProactorEventLoop 警告
**A:** 确保通过 `python server/main.py` 启动（而非 `python -m uvicorn`），`main.py` 顶部会自动设置 `WindowsSelectorEventLoopPolicy`。

### Q: 前端报 "Vite requires Node.js version 20.19+"
**A:** 运行 `nvm use 22.16.0`（或任何 22+ 版本）。

### Q: `/api/knowledge/databases` 返回空列表
**A:** kg-rag 知识库记录可能未创建。通过以下 SQL 插入：
```sql
docker exec postgres psql -U postgres -d yuxi_know -c "
INSERT INTO knowledge_bases (db_id, name, description, kb_type, additional_params, share_config, created_at, updated_at)
VALUES (
  'kb_kgrag_demo_001',
  '军事装备知识图谱',
  '基于 DBP15K zh_en 的军事装备领域中英文知识图谱',
  'kg-rag',
  '{\"kg_repo_root\": \"D:/work/毕设/知识图谱/Knowledge-Graph\", \"es_url\": \"http://localhost:9200\", \"es_index_name\": \"data2\", \"dbp15k_dataset\": \"zh_en\", \"enable_alignment_expansion\": true}'::jsonb,
  '{\"is_shared\": true, \"accessible_departments\": []}'::jsonb,
  NOW(), NOW()
) ON CONFLICT (db_id) DO NOTHING;
"
```
插入后需重启后端。

### Q: Elasticsearch 数据未导入
**A:** 运行以下命令导入实体数据：
```bash
cd Knowledge-Graph/retry
python run_entity_linking_es.py index \
  --processed-dir output/entity_linking_transformer_distilbert_mbert_rigorous_full_overnight_complete_20260331_001_labse \
  --es-url http://localhost:9200 \
  --index-name data2
```

### Q: 图谱可视化无数据
**A:** 确保 Elasticsearch 运行且 `data2` 索引有数据（`curl http://localhost:9200/data2/_count` 应大于 0）。

---

## 七、停止服务

```bash
# 停止前端：在前端终端按 Ctrl+C

# 停止后端：在后端终端按 Ctrl+C

# 停止 Docker 服务（可选）
cd D:\work\毕设\知识图谱\Yuxi
docker compose stop postgres
docker stop kg-elasticsearch
```

---

## 八、目录结构概览

```
Yuxi/
├── .env                          # 环境变量配置
├── backend/
│   ├── server/
│   │   ├── main.py               # 后端入口
│   │   ├── routers/
│   │   │   ├── knowledge_router.py  # 知识库 API
│   │   │   ├── dashboard_router.py  # 实验结果 API（新增 experiment-results 端点）
│   │   │   └── graph_router.py      # 图谱 API
│   │   └── utils/
│   │       └── auth_middleware.py    # 认证（演示模式已绕过）
│   └── package/yuxi/
│       └── integrations/
│           └── kg_rag_bridge.py     # Knowledge-Graph 桥接层
├── web/
│   ├── src/
│   │   ├── views/
│   │   │   ├── HomeView.vue          # 首页（改造）
│   │   │   ├── SimpleChatView.vue    # 对话查询（新建）
│   │   │   ├── DashboardView.vue     # 实验结果（改造）
│   │   │   ├── DataBaseView.vue      # 知识库列表
│   │   │   ├── DataBaseInfoView.vue  # 知识库详情（图谱/查询/评测）
│   │   │   └── KgEntityDetailView.vue # 实体详情
│   │   ├── components/dashboard/
│   │   │   ├── ExperimentOverviewCards.vue  # 概览卡片（新建）
│   │   │   ├── EntityLinkingResults.vue     # 实体链接评测（新建）
│   │   │   ├── AlignmentComparison.vue      # 对齐方法对比（新建）
│   │   │   └── AlignmentDelta.vue           # 对齐前后对比（新建）
│   │   ├── router/index.js           # 路由（精简）
│   │   └── layouts/AppLayout.vue     # 导航栏（改造）
│   └── .env                          # 前端代理配置
└── docker-compose.yml
```
