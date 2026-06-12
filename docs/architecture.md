# LangGraph Architecture

本项目采用 LangGraph 编排 Agentic-RAG。图中的每个节点都是一个可独立测试的函数，所有节点通过 `GraphState` 传递状态。

1. Query Rewrite: 将用户问题改写为可检索查询。
2. Retrieve: 结合向量检索、关键词检索和多跳扩展召回证据。
3. Evidence Check: 判断证据是否足以回答问题。
4. Answer: 基于证据生成带引用的答案。

## Graph

```text
START -> query_rewrite -> retrieve -> evidence_check
                                    ^              |
                                    |              v
                                    +---------- retrieve
                                                   |
                                                   v
                                                generate_answer -> END
```

## Implementation

- `src/financial_agentic_rag/graphs/state.py`: 图状态。
- `src/financial_agentic_rag/graphs/builder.py`: `StateGraph` 构建入口。
- `src/financial_agentic_rag/graphs/nodes/`: 节点函数。
- `src/financial_agentic_rag/graphs/edges/`: 条件边和路由函数。
- `src/financial_agentic_rag/runtime/app.py`: 图运行时入口。

## Indexing Pipeline

```text
pdf/*.pdf
  -> scripts/parse_pdfs_with_mineru.py
  -> data/processed/mineru/
  -> scripts/build_chunks.py
  -> data/processed/chunks.jsonl
  -> scripts/build_index.py
  -> storage/vectorstore/ + storage/docstore/
```

MinerU 负责 PDF 版面解析和表格识别。`build_chunks.py` 负责噪声过滤、章节继承、表格上下文合并、表头补齐和页码溯源。

当前向量库选择本地 FAISS。暂不接 pgvector，因为项目仍是本地原型且数据量小；后续服务化、多用户并发、复杂元数据过滤或增量写入需求明确后，再通过 `retrievers/vectorstore.py` 增加 pgvector backend。
