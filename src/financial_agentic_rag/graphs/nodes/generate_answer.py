from financial_agentic_rag.graphs.state import GraphState
from financial_agentic_rag.llms.client import LLMClientError, chat_completion


def _format_evidence(evidence: list[dict], max_items: int = 8) -> str:
    blocks = []
    for index, item in enumerate(evidence[:max_items], start=1):
        source = (
            f"{item.get('document_title', item.get('document_id', '未知文档'))}"
            f" / {item.get('chapter_title', '未知章节')}"
            f" / 第{item.get('page_start', '?')}-{item.get('page_end', '?')}页"
        )
        text = str(item.get("text", "")).strip()
        if len(text) > 1800:
            text = text[:1800] + "..."
        blocks.append(f"[证据{index}] {source}\n{text}")
    return "\n\n".join(blocks)


def generate_answer(state: GraphState) -> GraphState:
    """Generate the final answer from checked evidence."""

    evidence = state.get("evidence", [])
    if not evidence:
        final_answer = "当前证据不足，无法基于本地文档给出可靠答案。"
    else:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个法律领域 Agentic-RAG 助手。请严格基于给定证据回答，"
                    "不要编造未出现在证据中的法条或结论。回答末尾列出引用来源，"
                    "包含文档名、章节和页码。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"问题：{state.get('question', '')}\n\n"
                    f"检索证据：\n{_format_evidence(evidence)}\n\n"
                    "请给出简洁、可溯源的中文答案。"
                ),
            },
        ]
        try:
            final_answer = chat_completion(messages)
        except LLMClientError as exc:
            final_answer = f"已检索到相关证据，但调用本地 Qwen3-14B 失败：{exc}"
            return {
                **state,
                "answer": final_answer,
                "errors": [*state.get("errors", []), str(exc)],
            }

    return {
        **state,
        "answer": final_answer,
    }
