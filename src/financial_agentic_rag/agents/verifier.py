from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from financial_agentic_rag.agents.common import as_list, dedupe_chunks, format_chunks, read_prompt
from financial_agentic_rag.graphs.state import GraphState
from financial_agentic_rag.llms.client import chat_completion
from financial_agentic_rag.llms.json_utils import parse_json_object
from financial_agentic_rag.tracing import append_trace_event, summarize_chunks


def _normalize_task_verification(task: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
    is_sufficient = bool(parsed.get("is_sufficient"))
    return {
        "step_id": task.get("step_id"),
        "sub_question": task.get("sub_question", ""),
        "query": task.get("query", ""),
        "tool": task.get("tool", ""),
        "is_sufficient": is_sufficient,
        "reason": str(parsed.get("reason", "")),
        "missing_evidence": [] if is_sufficient else [str(item) for item in as_list(parsed.get("missing_evidence"))],
        "suggested_queries": [] if is_sufficient else [str(item) for item in as_list(parsed.get("suggested_queries"))],
        "suggested_tools": [] if is_sufficient else [str(item) for item in as_list(parsed.get("suggested_tools"))],
        "checked_chunk_count": len(task.get("chunks", [])),
    }


def _pending_task_from_verification(verification: dict[str, Any]) -> dict[str, Any]:
    suggested_queries = verification.get("suggested_queries") or [verification.get("query") or verification.get("sub_question")]
    suggested_tools = verification.get("suggested_tools") or ["hybrid_search"]
    return {
        "step_id": verification.get("step_id"),
        "sub_question": verification.get("sub_question"),
        "missing_evidence": verification.get("missing_evidence", []),
        "suggested_queries": suggested_queries,
        "suggested_tools": suggested_tools,
        "reason": verification.get("reason", ""),
    }


def _verify_one_task(question: str, task: dict[str, Any]) -> dict[str, Any]:
    chunks = task.get("chunks", [])
    if task.get("error"):
        return _normalize_task_verification(
            task,
            {
                "is_sufficient": False,
                "reason": f"该子任务执行失败：{task.get('error')}",
                "missing_evidence": ["需要重新检索该子任务的相关证据块"],
                "suggested_queries": [task.get("query") or task.get("sub_question") or question],
                "suggested_tools": [task.get("tool") or "hybrid_search"],
            },
        )
    if not chunks:
        return _normalize_task_verification(
            task,
            {
                "is_sufficient": False,
                "reason": "该子任务没有召回到证据。",
                "missing_evidence": ["需要检索该子任务对应的法律条文或事实依据"],
                "suggested_queries": [task.get("query") or task.get("sub_question") or question],
                "suggested_tools": ["hybrid_search"],
            },
        )

    messages = [
        {
            "role": "system",
            "content": read_prompt(
                "prompts/verifier.md",
                (
                    "你是法律 RAG 子任务 verifier。请只判断当前子任务的候选证据是否足够。"
                    "够回答即可，不要求穷尽所有细节。返回 JSON。"
                ),
            ),
        },
        {
            "role": "user",
            "content": (
                f"原问题：{question}\n"
                f"子任务：{task.get('sub_question', '')}\n"
                f"检索 query：{task.get('query', '')}\n\n"
                f"候选证据：\n{format_chunks(chunks)}"
            ),
        },
    ]
    parsed = parse_json_object(chat_completion(messages), {})
    return _normalize_task_verification(task, parsed)


def _failed_task_verification(question: str, task: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return _normalize_task_verification(
        task,
        {
            "is_sufficient": False,
            "reason": f"该子任务 verifier 调用失败：{type(exc).__name__}: {exc}",
            "missing_evidence": ["需要重新验证该子任务的候选证据"],
            "suggested_queries": [task.get("query") or task.get("sub_question") or question],
            "suggested_tools": [task.get("tool") or "hybrid_search"],
        },
    )


def _verify_tasks_parallel(question: str, tasks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str], int]:
    worker_count = min(len(tasks), 3) if tasks else 0
    if not tasks:
        return [], [], worker_count
    indexed_results: list[tuple[int, dict[str, Any]]] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_item = {
            executor.submit(_verify_one_task, question, task): (index, task)
            for index, task in enumerate(tasks)
        }
        for future in as_completed(future_to_item):
            index, task = future_to_item[future]
            try:
                verification = future.result()
            except Exception as exc:  # noqa: BLE001 - keep other task verifications alive.
                verification = _failed_task_verification(question, task, exc)
                errors.append(f"verifier step {task.get('step_id', 'unknown_step')} failed: {type(exc).__name__}: {exc}")
            indexed_results.append((index, verification))
    return [result for _, result in sorted(indexed_results, key=lambda item: item[0])], errors, worker_count


def verifier(state: GraphState) -> GraphState:
    question = state.get("question", "")
    is_final_round = state.get("iteration", 0) >= state.get("max_iterations", 2)
    task_results = state.get("task_results", [])
    current_step_ids = {step.get("step_id") for step in state.get("plan_steps", [])}
    current_tasks = [
        task for task in task_results
        if not current_step_ids or task.get("step_id") in current_step_ids
    ]
    chunks = state.get("retrieved_chunks", [])
    if not current_tasks:
        next_state = {
            **state,
            "verifier_feedback": {
                "is_sufficient": False,
                "reason": "没有召回到证据。",
                "missing_evidence": ["需要检索相关法律条文"],
                "suggested_queries": [question],
                "suggested_tools": ["hybrid_search"],
            },
            "task_verifications": [],
            "pending_tasks": [
                {
                    "step_id": "step_1",
                    "sub_question": question,
                    "missing_evidence": ["需要检索相关法律条文"],
                    "suggested_queries": [question],
                    "suggested_tools": ["hybrid_search"],
                    "reason": "没有可验证的子任务结果。",
                }
            ],
        }
        return {
            **next_state,
            "trace_events": append_trace_event(
                next_state,
                "verifier",
                "evidence_checked",
                next_state["verifier_feedback"],
            ),
        }
    current_verifications, verification_errors, worker_count = _verify_tasks_parallel(question, current_tasks)
    if is_final_round:
        current_verifications = [
            {
                **item,
                "is_sufficient": True,
                "reason": (
                    item.get("reason", "")
                    + " 已达到最大检索轮次，交由 synthesizer 基于现有证据回答并说明证据边界。"
                ).strip(),
                "missing_evidence": [],
                "suggested_queries": [],
                "suggested_tools": [],
            }
            if task.get("chunks") else item
            for item, task in zip(current_verifications, current_tasks)
        ]
    errors = [*state.get("errors", []), *verification_errors]

    previous_verifications = [
        item for item in state.get("task_verifications", [])
        if item.get("is_sufficient") and item.get("step_id") not in current_step_ids
    ]
    task_verifications = [*previous_verifications, *current_verifications]
    pending_tasks = [
        _pending_task_from_verification(item)
        for item in task_verifications
        if not item.get("is_sufficient")
    ]
    is_sufficient = bool(task_verifications) and not pending_tasks
    verified_step_ids = {
        item.get("step_id") for item in task_verifications if item.get("is_sufficient")
    }
    verified_evidence = dedupe_chunks(
        [
            chunk
            for task in task_results
            if task.get("step_id") in verified_step_ids
            for chunk in task.get("chunks", [])
        ]
    )
    missing_evidence = [
        missing
        for item in pending_tasks
        for missing in item.get("missing_evidence", [])
    ]
    suggested_queries = [
        query
        for item in pending_tasks
        for query in item.get("suggested_queries", [])
    ]
    suggested_tools = [
        tool
        for item in pending_tasks
        for tool in item.get("suggested_tools", [])
    ]
    feedback = {
        "is_sufficient": is_sufficient,
        "reason": "所有子任务证据均已满足。" if is_sufficient else "部分子任务证据仍不充分。",
        "missing_evidence": missing_evidence,
        "suggested_queries": suggested_queries,
        "suggested_tools": suggested_tools,
        "task_verifications": task_verifications,
        "pending_tasks": pending_tasks,
    }
    if errors != state.get("errors", []):
        feedback["reason"] = f"部分 verifier 调用失败，按已有证据继续：{errors[-1]}"

    next_state = {
        **state,
        "verifier_feedback": feedback,
        "task_verifications": task_verifications,
        "pending_tasks": pending_tasks,
        "verified_evidence": verified_evidence if is_sufficient else state.get("verified_evidence", []),
        "errors": errors,
    }
    return {
        **next_state,
        "trace_events": append_trace_event(
            next_state,
            "verifier",
            "evidence_checked",
            {
                **feedback,
                "parallel": True,
                "worker_count": worker_count,
                "verified_task_count": len(current_verifications),
                "task_verifications": [
                    {
                        **item,
                        "checked_chunks": summarize_chunks(
                            next(
                                (task.get("chunks", []) for task in task_results if task.get("step_id") == item.get("step_id")),
                                [],
                            )
                        ),
                    }
                    for item in task_verifications
                ],
                "checked_chunks": summarize_chunks(chunks),
            },
        ),
    }
