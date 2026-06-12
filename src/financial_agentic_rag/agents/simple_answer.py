from __future__ import annotations

from financial_agentic_rag.agents.common import complete_answer, format_chunks
from financial_agentic_rag.graphs.state import GraphState
from financial_agentic_rag.llms.client import LLMClientError
from financial_agentic_rag.tools.retrieval import run_retrieval_tool
from financial_agentic_rag.tracing import append_trace_event, summarize_chunks


def simple_answer(state: GraphState) -> GraphState:
    question = state.get("question", "")
    result = run_retrieval_tool("semantic_search", question)
    chunks = result["chunks"]
    messages = [
        {
            "role": "system",
            "content": "你是法律 RAG 助手。请只基于给定证据回答，并在末尾列出来源。",
        },
        {
            "role": "user",
            "content": f"问题：{question}\n\n证据：\n{format_chunks(chunks)}\n\n请给出简洁中文答案。",
        },
    ]
    try:
        answer, deltas = complete_answer(messages, bool(state.get("stream_answer")))
    except LLMClientError as exc:
        answer = f"已完成本地检索，但调用 Qwen 失败：{exc}"
        next_state = {
            **state,
            "answer": answer,
            "retrieved_chunks": chunks,
            "errors": [*state.get("errors", []), str(exc)],
        }
        return {
            **next_state,
            "trace_events": append_trace_event(
                next_state,
                "simple_answer",
                "answer_failed",
                {
                    "tool": result["tool"],
                    "query": result["query"],
                    "retrieved_count": len(chunks),
                    "chunks": summarize_chunks(chunks),
                    "error": str(exc),
                },
            ),
        }
    next_state = {
        **state,
        "answer": answer,
        "answer_deltas": [*state.get("answer_deltas", []), *deltas],
        "retrieved_chunks": chunks,
        "verified_evidence": chunks,
        "tool_history": [*state.get("tool_history", []), {k: result[k] for k in ("tool", "query", "warning")}],
    }
    return {
        **next_state,
        "trace_events": append_trace_event(
            next_state,
            "simple_answer",
            "answer_generated",
            {
                "tool": result["tool"],
                "query": result["query"],
                "retrieved_count": len(chunks),
                "chunks": summarize_chunks(chunks),
                "answer_chars": len(answer),
                "streamed": bool(deltas),
            },
        ),
    }
