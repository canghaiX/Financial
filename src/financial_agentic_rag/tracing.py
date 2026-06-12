from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from financial_agentic_rag.config import resolve_project_path
from financial_agentic_rag.graphs.state import GraphState
from financial_agentic_rag.utils.io import write_json


def new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid4().hex[:8]}"


def summarize_chunks(chunks: list[dict[str, Any]], max_items: int = 8, preview_chars: int = 220) -> list[dict[str, Any]]:
    summaries = []
    for chunk in chunks[:max_items]:
        text = str(chunk.get("text", "")).strip().replace("\n", " ")
        if len(text) > preview_chars:
            text = text[:preview_chars] + "..."
        summaries.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "document_title": chunk.get("document_title"),
                "chapter_title": chunk.get("chapter_title"),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "score": chunk.get("score"),
                "retrieval_tool": chunk.get("retrieval_tool"),
                "text_preview": text,
            }
        )
    return summaries


def append_trace_event(
    state: GraphState,
    node: str,
    event_type: str,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    iteration = state.get("iteration", 0)
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node": node,
        "event_type": event_type,
        "iteration": iteration,
        "round": iteration,
        "payload": payload,
    }
    return [*state.get("trace_events", []), event]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _truncate_text(text: Any, max_chars: int = 600) -> str:
    value = str(text or "").strip()
    if len(value) > max_chars:
        return value[:max_chars] + "..."
    return value


def _normalize_event(event: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(event)
    iteration = normalized.get("iteration", normalized.get("round", 0))
    normalized["iteration"] = iteration
    normalized["round"] = normalized.get("round", iteration)
    payload = dict(normalized.get("payload") or {})
    if normalized.get("node") == "verifier":
        payload["suggested_tools"] = _as_list(payload.get("suggested_tools"))
        payload["task_verifications"] = [
            {
                **item,
                "suggested_tools": _as_list(item.get("suggested_tools")),
            }
            if isinstance(item, dict)
            else item
            for item in _as_list(payload.get("task_verifications"))
        ]
    normalized["payload"] = payload
    return normalized


def _format_list(items: list[Any], empty: str = "无") -> str:
    if not items:
        return f"- {empty}"
    return "\n".join(f"- {item}" for item in items)


def _render_planner(payload: dict[str, Any]) -> list[str]:
    lines = ["### Planner"]
    steps = _as_list(payload.get("steps"))
    if not steps:
        lines.append("- 子问题：无")
        return lines
    for step in steps:
        if not isinstance(step, dict):
            lines.append(f"- 子问题：{step}")
            continue
        lines.extend(
            [
                f"- `{step.get('step_id', 'step')}` 子问题：{step.get('sub_question', '')}",
                f"  - 工具：`{step.get('tool', '')}`",
                f"  - 查询：{step.get('query', '')}",
                f"  - 原因：{step.get('reason', '')}",
            ]
        )
    if payload.get("used_fallback"):
        lines.append("- 使用 fallback plan：是")
    return lines


def _render_executer(payload: dict[str, Any]) -> list[str]:
    lines = [
        "### Executer",
        f"- 并行执行：{payload.get('parallel', False)}",
        f"- worker 数：{payload.get('worker_count', 0)}",
        f"- 累计召回块数：{payload.get('retrieved_total', 0)}",
    ]
    steps = _as_list(payload.get("steps"))
    if not steps:
        lines.append("- 工具调用：无")
        return lines
    for step in steps:
        if not isinstance(step, dict):
            lines.append(f"- 工具调用：{step}")
            continue
        lines.extend(
            [
                f"- `{step.get('step_id', 'step')}` 工具：`{step.get('tool', '')}`",
                f"  - 查询：{step.get('query', '')}",
                f"  - 召回块数：{step.get('result_count', 0)}",
                f"  - warning：{step.get('warning') or '无'}",
                f"  - error：{step.get('error') or '无'}",
            ]
        )
    return lines


def _render_verifier(payload: dict[str, Any]) -> list[str]:
    lines = [
        "### Verifier",
        f"- 证据是否充分：{payload.get('is_sufficient')}",
        f"- 判断原因：{payload.get('reason', '')}",
        "- 还需要找的证据块：",
        _format_list(_as_list(payload.get("missing_evidence"))),
        "- 建议查询：",
        _format_list(_as_list(payload.get("suggested_queries"))),
        "- 建议工具：",
        _format_list(_as_list(payload.get("suggested_tools"))),
    ]
    task_verifications = [item for item in _as_list(payload.get("task_verifications")) if isinstance(item, dict)]
    if task_verifications:
        lines.append("- 子任务验证：")
        for item in task_verifications:
            lines.extend(
                [
                    f"  - `{item.get('step_id', 'step')}` 子问题：{item.get('sub_question', '')}",
                    f"    - 是否满足：{item.get('is_sufficient')}",
                    f"    - 原因：{item.get('reason', '')}",
                    f"    - 缺少证据：{', '.join(str(v) for v in _as_list(item.get('missing_evidence'))) or '无'}",
                    f"    - 建议查询：{', '.join(str(v) for v in _as_list(item.get('suggested_queries'))) or '无'}",
                    f"    - 建议工具：{', '.join(str(v) for v in _as_list(item.get('suggested_tools'))) or '无'}",
                ]
            )
    return lines


def _render_router(payload: dict[str, Any]) -> list[str]:
    return [
        "## Router",
        f"- query_type：`{payload.get('query_type', payload.get('route_type', ''))}`",
        f"- reason：{payload.get('reason', '')}",
        f"- confidence：{payload.get('confidence', '')}",
    ]


def _render_answer_node(node: str, payload: dict[str, Any]) -> list[str]:
    title = "Simple Answer" if node == "simple_answer" else "Synthesizer"
    lines = [f"### {title}"]
    if node == "simple_answer":
        lines.extend(
            [
                f"- 工具：`{payload.get('tool', '')}`",
                f"- 查询：{payload.get('query', '')}",
                f"- 召回块数：{payload.get('retrieved_count', 0)}",
            ]
        )
    else:
        lines.extend(
            [
                f"- 使用证据数：{payload.get('evidence_count', 0)}",
                f"- 证据是否充分：{payload.get('is_sufficient')}",
            ]
        )
    lines.extend(
        [
            f"- 答案字符数：{payload.get('answer_chars', 0)}",
            f"- 流式输出：{payload.get('streamed', False)}",
        ]
    )
    if payload.get("error"):
        lines.append(f"- error：{payload['error']}")
    return lines


def _display_round(event: dict[str, Any]) -> int:
    iteration = int(event.get("round", event.get("iteration", 0)) or 0)
    if event.get("node") == "planner":
        return iteration + 1
    return max(iteration, 1)


def render_trace_markdown(payload: dict[str, Any]) -> str:
    events = [_normalize_event(event) for event in payload.get("events", [])]
    lines = [
        f"# Agentic-RAG Trace `{payload.get('run_id', '')}`",
        "",
        "## Overview",
        f"- 问题：{payload.get('question', '')}",
        f"- query_type：`{payload.get('query_type', payload.get('route_type', ''))}`",
        f"- 总轮次：{payload.get('iterations', 0)}/{payload.get('max_iterations', 2)}",
        f"- 错误数：{len(payload.get('errors', []))}",
        f"- JSON Trace：{payload.get('trace_path', '')}",
        "",
        "## Final Answer Preview",
        _truncate_text(payload.get("answer", ""), max_chars=800) or "无",
        "",
    ]
    errors = payload.get("errors", [])
    if errors:
        lines.extend(["## Errors", _format_list(errors), ""])

    router_events = [event for event in events if event.get("node") == "router"]
    if router_events:
        lines.extend(_render_router(router_events[-1].get("payload", {})))
        lines.append("")

    round_events = [event for event in events if event.get("node") != "router"]
    rounds = sorted({_display_round(event) for event in round_events})
    for round_number in rounds:
        lines.extend([f"## Round {round_number}", ""])
        for event in [item for item in round_events if _display_round(item) == round_number]:
            node = event.get("node")
            payload_data = event.get("payload", {})
            if node == "planner":
                lines.extend(_render_planner(payload_data))
            elif node == "executer":
                lines.extend(_render_executer(payload_data))
            elif node == "verifier":
                lines.extend(_render_verifier(payload_data))
            elif node in {"simple_answer", "synthesizer"}:
                lines.extend(_render_answer_node(node, payload_data))
            else:
                lines.extend([f"### {node}", f"- event_type：{event.get('event_type', '')}"])
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_trace(state: GraphState, traces_dir: str | Path = "logs/traces") -> Path:
    run_id = state.get("run_id") or new_run_id()
    trace_path = resolve_project_path(traces_dir) / f"{run_id}.json"
    markdown_path = trace_path.with_suffix(".md")
    events = [_normalize_event(event) for event in state.get("trace_events", [])]
    payload = {
        "run_id": run_id,
        "trace_path": str(trace_path),
        "trace_markdown_path": str(markdown_path),
        "question": state.get("question", ""),
        "query_type": state.get("query_type", ""),
        "route_type": state.get("route_type", ""),
        "iterations": state.get("iteration", 0),
        "max_iterations": state.get("max_iterations", 2),
        "answer": state.get("answer", ""),
        "errors": state.get("errors", []),
        "events": events,
    }
    write_json(trace_path, payload)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_trace_markdown(payload), encoding="utf-8")
    return trace_path
