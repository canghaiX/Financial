from __future__ import annotations

from financial_agentic_rag.agents.common import DEFAULT_ROUTER_PROMPT, read_prompt
from financial_agentic_rag.graphs.state import GraphState
from financial_agentic_rag.llms.client import LLMClientError, chat_completion
from financial_agentic_rag.llms.json_utils import parse_json_object
from financial_agentic_rag.tracing import append_trace_event


def router(state: GraphState) -> GraphState:
    question = state.get("question", "")
    messages = [
        {"role": "system", "content": read_prompt("prompts/router.md", DEFAULT_ROUTER_PROMPT)},
        {"role": "user", "content": f"用户问题：{question}"},
    ]
    errors = state.get("errors", [])
    raw_query_type = "multi_hop"
    reason = "router defaulted to multi_hop"
    confidence = 0.0
    try:
        content = chat_completion(messages)
        parsed = parse_json_object(content, {})
        if parsed.get("query_type") == "simple":
            raw_query_type = "simple"
        elif parsed.get("query_type") == "multi_hop":
            raw_query_type = "multi_hop"
        else:
            errors = [*errors, "router returned invalid JSON or unknown query_type; defaulted to multi_hop"]
        reason = str(parsed.get("reason") or reason)
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
    except LLMClientError as exc:
        errors = [*errors, str(exc)]
        reason = f"router LLM failed, defaulted to multi_hop: {exc}"

    query_type = raw_query_type if raw_query_type == "simple" else "multi_hop"
    next_state = {
        **state,
        "query_type": query_type,
        "route_type": query_type,
        "iteration": state.get("iteration", 0),
        "max_iterations": state.get("max_iterations", 2) or 2,
        "plan_steps": state.get("plan_steps", []),
        "executed_steps": state.get("executed_steps", []),
        "retrieved_chunks": state.get("retrieved_chunks", []),
        "verified_evidence": state.get("verified_evidence", []),
        "tool_history": state.get("tool_history", []),
        "trace_events": state.get("trace_events", []),
        "stream_answer": state.get("stream_answer", False),
        "answer_deltas": state.get("answer_deltas", []),
        "errors": errors,
    }
    return {
        **next_state,
        "trace_events": append_trace_event(
            next_state,
            "router",
            "route_decision",
            {
                "question": question,
                "query_type": query_type,
                "route_type": query_type,
                "reason": reason,
                "confidence": confidence,
            },
        ),
    }
