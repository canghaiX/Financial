# Financial Agentic-RAG

基于 LangGraph 搭建的金融/法律领域多跳 Agentic-RAG 项目。

项目目标是把本地法律 PDF 语料构建成可检索知识库，并通过 LangGraph 编排“问题改写 -> 多跳检索 -> 证据检查 -> 答案生成”的可观测 Agent 工作流。

## 目录结构

```text
.
├── configs/                 # 模型、检索、索引、LangGraph 图配置
├── data/                    # 原始数据、清洗数据、评测集
├── docs/                    # 项目设计文档、实验记录
├── logs/                    # 运行日志
├── notebooks/               # 探索性分析和索引调试
├── pdf/                     # 已下载的法律 PDF 数据
├── prompts/                 # 图节点使用的提示词
├── scripts/                 # 数据处理、索引构建、评测脚本
├── src/financial_agentic_rag/
│   ├── agents/              # 对外 Agent 封装，调用 LangGraph app
│   ├── chains/              # 节点内部可复用 LCEL / LangChain 链
│   ├── document_loaders/    # PDF/DOCX 加载与解析
│   ├── evaluation/          # 检索与问答评测
│   ├── graphs/              # LangGraph 核心
│   │   ├── builder.py       # StateGraph 构建与 compile 入口
│   │   ├── state.py         # GraphState 状态定义
│   │   ├── nodes/           # query_rewrite / retrieve / grade / answer 节点
│   │   ├── edges/           # 条件边和路由函数
│   │   └── checkpoints/     # checkpoint 配置或适配器
│   ├── indexing/            # 切分、嵌入、索引构建
│   ├── llms/                # LLM 和 Embedding 适配
│   ├── memory/              # 会话记忆、长期记忆或检查点存储扩展
│   ├── retrievers/          # 向量、关键词、混合、多跳检索
│   ├── runtime/             # 图运行入口、事件流、配置加载
│   ├── schemas/             # Pydantic 状态和数据结构
│   ├── tools/               # Agent 可调用工具
│   └── utils/               # 通用工具函数
├── storage/                 # 向量库、文档库、缓存
└── tests/                   # 单元测试与集成测试
```

## LangGraph 工作流

```text
START
  -> query_rewrite
  -> retrieve
  -> evidence_check
  -> route_after_check
       ├── enough_evidence -> generate_answer
       └── need_more       -> retrieve
  -> END
```

核心状态定义在 `src/financial_agentic_rag/graphs/state.py`：

- `question`：用户原始问题
- `rewritten_queries`：改写后的检索查询
- `documents`：召回文档片段
- `evidence`：经过证据检查的候选依据
- `needs_more_retrieval`：是否继续多跳检索
- `answer`：最终回答

## 数据

当前法律文档位于 `pdf/`，清单文件为 `pdf/manifest.csv`。

## MinerU 清洗与索引

第一版使用 MinerU 解析 PDF，生成可溯源 chunk，再构建本地 FAISS 索引。默认不接 pgvector，原因是当前数据量只有 103 份 PDF，单机 FAISS 更轻，`chunks.jsonl + docstore` 已能满足文档、章节、页码溯源。

处理流程：

```bash
python scripts/parse_pdfs_with_mineru.py
python scripts/build_chunks.py
python scripts/build_index.py
```

默认使用 GPU 1 运行 MinerU，可在 `configs/retrieval_config.yaml` 的 `mineru.cuda_visible_devices` 调整。当前项目默认假设 GPU 0 用于本地 vLLM 服务。
MinerU 3.x 的命令行入口是 `mineru`；如果你使用旧版并只有 `magic-pdf`，把 `configs/retrieval_config.yaml` 里的 `mineru.command` 改回 `magic-pdf` 即可。
本地解析需要 MinerU pipeline 依赖，如果看到 `ai pipeline dependencies ... Install mineru[pipeline]`，运行：

```bash
pip install -U "mineru[pipeline]>=3.2.3"
```

首次运行本地 MinerU 前建议先下载 pipeline 模型，国内环境优先使用 ModelScope：

```bash
conda install -n financial-rag -c conda-forge libgl libglib -y
pip install six
mineru-models-download -s modelscope -m pipeline
```

当前项目默认使用 `backend: pipeline` 和 `MINERU_MODEL_SOURCE=local`。如果切回 `hybrid-auto-engine`，还需要额外下载 VLM 模型。

当前检索默认使用 BGE vLLM 服务：`bge-m3` 做向量粗召回，`bge-reranker-v2-m3` 做最终重排。先分别启动两个服务：

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=fork \
vllm serve /data/models/bge-m3 \
  --runner pooling \
  --convert embed \
  --served-model-name bge-m3 \
  --host 0.0.0.0 \
  --port 8001 \
  --dtype auto \
  --gpu-memory-utilization 0.10 \
  --max-model-len 8192
```

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=fork \
vllm serve /data/models/bge-reranker-v2-m3 \
  --runner pooling \
  --convert classify \
  --served-model-name bge-reranker-v2-m3 \
  --host 0.0.0.0 \
  --port 8002 \
  --dtype auto \
  --gpu-memory-utilization 0.10 \
  --max-model-len 8192
```

健康检查：

```bash
curl http://127.0.0.1:8001/v1/models
curl http://127.0.0.1:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"bge-m3","input":["危险化学品安全管理义务"]}'
curl http://127.0.0.1:8002/v1/models
curl http://127.0.0.1:8002/rerank \
  -H "Content-Type: application/json" \
  -d '{"model":"bge-reranker-v2-m3","query":"危险化学品安全管理义务","documents":["企业应当建立安全管理制度","无关文本"],"top_n":1}'
```

切换到 BGE 后，必须重新构建 FAISS 索引：

```bash
python scripts/build_index.py
```

本地 vLLM 服务示例：

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /data/models/Qwen3-14B \
  --served-model-name Qwen3-14B \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --gpu-memory-utilization 0.7 \
  --max-model-len 16384 \
  --tool-call-parser hermes
```

项目中的 LLM 配置位于 `configs/model_config.yaml`，默认连接：

```text
http://127.0.0.1:8000/v1
model: Qwen3-14B
max_model_len: 16384
tool_call_parser: hermes
```

`graphs/nodes/generate_answer.py` 会通过 OpenAI-compatible 接口调用该 vLLM 服务；如果你修改端口或模型名，同步改 `.env` 或 `configs/model_config.yaml` 即可。因为 vLLM 已占用 GPU 0，MinerU 默认使用 `MINERU_CUDA_VISIBLE_DEVICES=1`。

清洗输出：

- `data/processed/mineru/`：MinerU 原始解析结果
- `data/processed/chunks.jsonl`：RAG chunk 主文件
- `data/processed/rejected_chunks.jsonl`：缺少页码或缺少解析结果等不可索引记录
- `data/processed/markdown/`：人工质检用 Markdown
- `storage/vectorstore/`：FAISS 索引
- `storage/docstore/`：chunk 原文和元数据

chunk 设计要点：

- 每个 chunk 都包含 `document_id`、`document_title`、`chapter_title`、`page_start`、`page_end`。
- 表格 chunk 会合并表格前后上下文，优先保留 MinerU 表格 HTML，同时生成 Markdown 表格。
- 跨页或连续表格会重复表头；如果无法补齐表头，会标记 `metadata.table_header_missing=true`。
- 页眉、页脚、页码、目录点线和孤立标点会被过滤；法律条文编号会保留。

## 常用入口

```bash
python scripts/build_index.py
python scripts/run_agent.py "根据《危险化学品安全法》和相关安全管理规定，危险化学品生产、储存企业在重大危险源管理、事故应急处置和违法责任承担方面分别有哪些义务和后果？"
```

目前目录和接口已预留，后续实现时优先从 `graphs/builder.py` 和 `graphs/nodes/` 开始补节点逻辑。
