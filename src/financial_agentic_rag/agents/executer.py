from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from typing import Any

from financial_agentic_rag.agents.common import dedupe_chunks
from financial_agentic_rag.config import load_yaml
from financial_agentic_rag.graphs.state import GraphState
from financial_agentic_rag.tools.retrieval import run_retrieval_tool
from financial_agentic_rag.tracing import append_trace_event, summarize_chunks


LEGAL_NAME_RE = re.compile(r"《[^》]{2,40}》|[\u4e00-\u9fff]{2,30}法|[\u4e00-\u9fff]{2,30}条例")
ARTICLE_RE = re.compile(r"第[一二三四五六七八九十百千万零〇0-9]+[章节条编款项]?")
KEYWORD_RE = re.compile(r"重大危险源|行政处罚|刑事责任|民事责任|法律责任|事故|应急|义务|责任|处罚|备案|登记|培训|安全管理|生产|储存")


def _unique_join(parts: list[str]) -> str:
    seen = set()
    values = []
    for part in parts:
        value = str(part or "").strip()
        if value and value not in seen:
            seen.add(value)
            values.append(value)
    return " ".join(values)


def _extract_terms(*texts: str) -> list[str]:
    joined = " ".join(str(text or "") for text in texts)
    terms = []
    for pattern in (LEGAL_NAME_RE, ARTICLE_RE, KEYWORD_RE):
        terms.extend(pattern.findall(joined))
    return list(dict.fromkeys(str(term).strip("《》") for term in terms if str(term).strip()))


def _build_effective_query(step: dict[str, Any], question: str) -> tuple[str, str]:
    original_query = str(step.get("query") or step.get("sub_question") or question)
    sub_question = str(step.get("sub_question") or "")
    terms = _extract_terms(question, sub_question, original_query)
    return original_query, _unique_join([original_query, sub_question, *terms])


def _step_is_pending(step: dict[str, Any], pending_tasks: list[dict[str, Any]]) -> bool:
    step_id = step.get("step_id")
    return any(task.get("step_id") == step_id for task in pending_tasks)


def _top_k_for_tool(tool: str, config: dict[str, Any], is_pending: bool) -> int:
    retrieval_cfg = config.get("retrieval", {})
    return int(retrieval_cfg.get("final_top_k") or retrieval_cfg.get("top_k", 5))


def _clean_chunks(chunks: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        text = str(chunk.get("text", "")).strip()
        if not chunk_id or not text:
            continue
        candidate = dict(chunk)
        existing = by_id.get(str(chunk_id))
        if existing is None or float(candidate.get("score", 0) or 0) > float(existing.get("score", 0) or 0):
            by_id[str(chunk_id)] = candidate
    return sorted(by_id.values(), key=lambda item: float(item.get("score", 0)), reverse=True)[:top_k]


def _execute_step(step: dict[str, Any], question: str, config: dict[str, Any], pending_tasks: list[dict[str, Any]]) -> dict[str, Any]:
    tool = str(step.get("tool") or "hybrid_search")
    original_query, effective_query = _build_effective_query(step, question)
    top_k = _top_k_for_tool(tool, config, _step_is_pending(step, pending_tasks))
    try:
        result = run_retrieval_tool(tool, effective_query, top_k=top_k)
        chunks = _clean_chunks(result["chunks"], top_k)
        warning = result["warning"]
        if not chunks:
            warning = (warning + " " if warning else "") + "No usable chunks after filtering; planner may need to rewrite query."
        return {
            "step": step,
            "chunks": chunks,
            "task_result": {
                "step_id": step.get("step_id"),
                "sub_question": step.get("sub_question", ""),
                "tool": result["tool"],
                "requested_tool": result["requested_tool"],
                "query": effective_query,
                "original_query": original_query,
                "effective_query": effective_query,
                "top_k": top_k,
                "result_count": len(chunks),
                "chunks": chunks,
                "warning": warning,
                "error": "",
            },
            "history": {
                "step_id": step.get("step_id"),
                "requested_tool": result["requested_tool"],
                "tool": result["tool"],
                "query": effective_query,
                "original_query": original_query,
                "effective_query": effective_query,
                "top_k": top_k,
                "result_count": len(chunks),
                "warning": warning,
                "error": "",
            },
            "executed_step": {
                **step,
                "query": effective_query,
                "original_query": original_query,
                "effective_query": effective_query,
                "top_k": top_k,
                "result_count": len(chunks),
                "warning": warning,
                "error": "",
            },
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 - one failed subtask should not break the whole graph.
        error = f"{type(exc).__name__}: {exc}"
        return {
            "step": step,
            "chunks": [],
            "task_result": {
                "step_id": step.get("step_id"),
                "sub_question": step.get("sub_question", ""),
                "tool": tool,
                "requested_tool": tool,
                "query": effective_query,
                "original_query": original_query,
                "effective_query": effective_query,
                "top_k": top_k,
                "result_count": 0,
                "chunks": [],
                "warning": "",
                "error": error,
            },
            "history": {
                "step_id": step.get("step_id"),
                "requested_tool": tool,
                "tool": tool,
                "query": effective_query,
                "original_query": original_query,
                "effective_query": effective_query,
                "top_k": top_k,
                "result_count": 0,
                "warning": "",
                "error": error,
            },
            "executed_step": {
                **step,
                "query": effective_query,
                "original_query": original_query,
                "effective_query": effective_query,
                "top_k": top_k,
                "result_count": 0,
                "warning": "",
                "error": error,
            },
            "error": error,
        }


def executer(state: GraphState) -> GraphState:
    config = load_yaml("configs/retrieval_config.yaml")
    retrieved = list(state.get("retrieved_chunks", []))
    tool_history = list(state.get("tool_history", []))
    executed_steps = list(state.get("executed_steps", []))
    task_results = list(state.get("task_results", []))
    plan_steps = list(state.get("plan_steps", []))
    pending_tasks = list(state.get("pending_tasks", []))
    worker_count = min(len(plan_steps), 4) if plan_steps else 0
    round_results: list[dict[str, Any]] = []
    if plan_steps:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_index = {
                executor.submit(_execute_step, step, state.get("question", ""), config, pending_tasks): index
                for index, step in enumerate(plan_steps)
            }
            indexed_results: list[tuple[int, dict[str, Any]]] = []
            for future in as_completed(future_to_index):
                indexed_results.append((future_to_index[future], future.result()))
        round_results = [result for _, result in sorted(indexed_results, key=lambda item: item[0])]

    errors = list(state.get("errors", []))
    for result in round_results:
        retrieved.extend(result["chunks"])
        tool_history.append(result["history"])
        executed_steps.append(result["executed_step"])
        task_results.append(result["task_result"])
        if result["error"]:
            step_id = result["executed_step"].get("step_id", "unknown_step")
            errors.append(f"executer step {step_id} failed: {result['error']}")

    deduped = dedupe_chunks(retrieved)
    round_executed_steps = [result["executed_step"] for result in round_results]
    round_task_results = [result["task_result"] for result in round_results]
    next_state = {
        **state,
        "iteration": state.get("iteration", 0) + 1,
        "retrieved_chunks": deduped,
        "documents": deduped,
        "executed_steps": executed_steps,
        "task_results": task_results,
        "tool_history": tool_history,
        "errors": errors,
    }
    return {
        **next_state,
        "trace_events": append_trace_event(
            next_state,
            "executer",
            "tools_executed",
            {
                "parallel": True,
                "worker_count": worker_count,
                "steps": round_executed_steps,
                "task_results": [
                    {**task, "chunks": summarize_chunks(task.get("chunks", []))}
                    for task in round_task_results
                ],
                "tool_history": tool_history,
                "retrieved_total": len(deduped),
                "chunks": summarize_chunks(deduped),
            },
        ),
    }
