from __future__ import annotations

from financial_agentic_rag.agents.common import (
    default_plan,
    format_chunks,
    normalize_plan_steps,
    read_prompt,
)
from financial_agentic_rag.graphs.state import GraphState
from financial_agentic_rag.llms.client import LLMClientError, chat_completion
from financial_agentic_rag.llms.json_utils import parse_json_object
from financial_agentic_rag.tracing import append_trace_event


def _limit_steps_for_fast_mode(steps: list[dict], has_pending_tasks: bool) -> list[dict]:
    return steps[:2] if has_pending_tasks else steps[:4]


def planner(state: GraphState) -> GraphState:
    question = state.get("question", "")
    feedback = state.get("verifier_feedback", {})
    pending_tasks = state.get("pending_tasks", [])
    task_verifications = state.get("task_verifications", [])
    messages = [
        {
            "role": "system",
            "content": read_prompt(
                "prompts/planner.md",
                (
                    "你是法律 Agentic-RAG planner。请返回 JSON 对象，格式："
                    '{"steps":[{"step_id":"step_1","sub_question":"...","tool":"semantic_search|keyword_search|hybrid_search",'
                    '"query":"...","reason":"..."}]}。不要使用 knowledge_graph_search。'
                ),
            ),
        },
        {
            "role": "user",
            "content": (
                f"原问题：{question}\n"
                f"当前轮次：{state.get('iteration', 0) + 1}/{state.get('max_iterations', 2)}\n"
                f"待补充任务（优先只围绕这些任务重新规划）：{pending_tasks}\n"
                f"逐任务验证结果：{task_verifications}\n"
                f"上轮反馈：{feedback}\n"
                f"历史工具：{state.get('tool_history', [])}\n"
                f"已召回证据摘要：{format_chunks(state.get('retrieved_chunks', []), max_items=5, max_chars=400)}"
            ),
        },
    ]
    try:
        content = chat_completion(messages)
        parsed = parse_json_object(content, {"steps": default_plan(question, feedback)})
        steps = normalize_plan_steps(
            parsed.get("steps") if isinstance(parsed.get("steps"), list) else default_plan(question, feedback),
            question,
        )
        steps = _limit_steps_for_fast_mode(steps, bool(pending_tasks))
        used_fallback = not isinstance(parsed.get("steps"), list)
    except LLMClientError as exc:
        fallback_feedback = pending_tasks[0] if pending_tasks else feedback
        steps = normalize_plan_steps(default_plan(question, fallback_feedback), question)
        steps = _limit_steps_for_fast_mode(steps, bool(pending_tasks))
        next_state = {**state, "plan_steps": steps, "errors": [*state.get("errors", []), str(exc)]}
        return {
            **next_state,
            "trace_events": append_trace_event(
                next_state,
                "planner",
                "plan_failed_fallback",
                {"steps": steps, "error": str(exc), "used_fallback": True},
            ),
        }
    next_state = {**state, "plan_steps": steps}
    return {
        **next_state,
        "trace_events": append_trace_event(
            next_state,
            "planner",
            "plan_created",
            {
                "steps": steps,
                "used_fallback": used_fallback,
                "feedback": feedback,
                "pending_tasks": pending_tasks,
                "max_steps": 2 if pending_tasks else 4,
            },
        ),
    }
