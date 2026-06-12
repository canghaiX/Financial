from __future__ import annotations

from financial_agentic_rag.agents.common import complete_answer, format_chunks
from financial_agentic_rag.graphs.state import GraphState
from financial_agentic_rag.llms.client import LLMClientError
from financial_agentic_rag.tracing import append_trace_event, summarize_chunks


def synthesizer(state: GraphState) -> GraphState:
    question = state.get("question", "")
    evidence = state.get("verified_evidence") or state.get("retrieved_chunks", [])
    feedback = state.get("verifier_feedback", {})
    insufficiency_note = "" if feedback.get("is_sufficient") else "注意：证据可能不足，请明确说明不确定性。\n"
    messages = [
        {
            "role": "system",
            "content": "你是法律 RAG synthesizer。请严格基于证据回答，末尾列出文档名、章节、页码来源。",
        },
        {
            "role": "user",
            "content": f"{insufficiency_note}问题：{question}\n\n证据：\n{format_chunks(evidence)}\n\n请生成最终中文答案。",
        },
    ]
    try:
        answer, deltas = complete_answer(messages, bool(state.get("stream_answer")))
    except LLMClientError as exc:
        answer = f"无法调用 Qwen 生成最终答案。已召回 {len(evidence)} 个证据块，错误：{exc}"
        next_state = {**state, "answer": answer, "evidence": evidence, "errors": [*state.get("errors", []), str(exc)]}
        return {
            **next_state,
            "trace_events": append_trace_event(
                next_state,
                "synthesizer",
                "synthesis_failed",
                {
                    "evidence_count": len(evidence),
                    "is_sufficient": bool(feedback.get("is_sufficient")),
                    "error": str(exc),
                },
            ),
        }
    next_state = {
        **state,
        "answer": answer,
        "answer_deltas": [*state.get("answer_deltas", []), *deltas],
        "evidence": evidence,
    }
    return {
        **next_state,
        "trace_events": append_trace_event(
            next_state,
            "synthesizer",
            "answer_synthesized",
            {
                "evidence_count": len(evidence),
                "is_sufficient": bool(feedback.get("is_sufficient")),
                "answer_chars": len(answer),
                "streamed": bool(deltas),
                "chunks": summarize_chunks(evidence),
            },
        ),
    }
