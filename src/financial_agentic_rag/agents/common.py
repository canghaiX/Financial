from __future__ import annotations

from typing import Any

from financial_agentic_rag.config import resolve_project_path
from financial_agentic_rag.llms.client import chat_completion, stream_chat_completion


DEFAULT_ROUTER_PROMPT = """你是法律 Agentic-RAG 的 router。
请判断用户问题属于 simple 或 multi_hop。
只返回 JSON：{"query_type":"simple|multi_hop","reason":"...","confidence":0.0}
只有单一事实、定义、适用范围、发布日期、施行日期、制定机关等一次语义检索可回答的问题才是 simple。
其他问题都返回 multi_hop。"""


def read_prompt(relative_path: str, fallback: str) -> str:
    path = resolve_project_path(relative_path)
    if not path.exists():
        return fallback
    return path.read_text(encoding="utf-8")


def default_plan(question: str, feedback: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    queries = as_list(feedback.get("suggested_queries")) if feedback else None
    tools = as_list(feedback.get("suggested_tools")) if feedback else None
    return [
        {
            "step_id": str(feedback.get("step_id") if feedback else "step_1") if feedback else "step_1",
            "sub_question": (queries or [question])[0],
            "tool": (tools or ["hybrid_search"])[0],
            "query": (queries or [question])[0],
            "reason": "默认使用混合检索召回相关法律条文。",
        }
    ]


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def normalize_tool(tool: Any) -> str:
    value = str(tool or "hybrid_search")
    if value not in {"semantic_search", "keyword_search", "hybrid_search"}:
        return "hybrid_search"
    return value


def normalize_plan_steps(steps: Any, question: str) -> list[dict[str, Any]]:
    normalized = []
    for index, step in enumerate(as_list(steps), start=1):
        item = step if isinstance(step, dict) else {"sub_question": str(step)}
        sub_question = str(item.get("sub_question") or item.get("query") or question)
        query = str(item.get("query") or sub_question or question)
        normalized.append(
            {
                "step_id": str(item.get("step_id") or f"step_{index}"),
                "sub_question": sub_question,
                "tool": normalize_tool(item.get("tool")),
                "query": query,
                "reason": str(item.get("reason") or "检索该子问题所需的法律证据。"),
            }
        )
    return normalized or default_plan(question)


def dedupe_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        if chunk_id and chunk_id not in by_id:
            by_id[chunk_id] = chunk
    return list(by_id.values())


def format_chunks(chunks: list[dict[str, Any]], max_items: int = 10, max_chars: int = 1200) -> str:
    lines = []
    for index, chunk in enumerate(chunks[:max_items], start=1):
        text = str(chunk.get("text", "")).strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        source = (
            f"{chunk.get('document_title', chunk.get('document_id', '未知文档'))}"
            f" / {chunk.get('chapter_title', '未知章节')}"
            f" / 第{chunk.get('page_start', '?')}-{chunk.get('page_end', '?')}页"
        )
        lines.append(f"[证据{index}] {source}\n{text}")
    return "\n\n".join(lines) if lines else "无"


def complete_answer(messages: list[dict[str, str]], stream_answer: bool) -> tuple[str, list[str]]:
    if not stream_answer:
        answer = chat_completion(messages)
        return answer, []
    deltas: list[str] = []
    for delta in stream_chat_completion(messages):
        deltas.append(delta)
    return "".join(deltas), deltas
