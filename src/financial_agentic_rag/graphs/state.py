from typing import Any, TypedDict


class GraphState(TypedDict, total=False):
    """State shared by all LangGraph nodes."""

    question: str
    run_id: str
    trace_events: list[dict[str, Any]]
    trace_path: str
    stream_answer: bool
    answer_deltas: list[str]
    query_type: str
    route_type: str
    iteration: int
    max_iterations: int
    plan_steps: list[dict[str, Any]]
    executed_steps: list[dict[str, Any]]
    task_results: list[dict[str, Any]]
    task_verifications: list[dict[str, Any]]
    pending_tasks: list[dict[str, Any]]
    retrieved_chunks: list[dict[str, Any]]
    verified_evidence: list[dict[str, Any]]
    verifier_feedback: dict[str, Any]
    tool_history: list[dict[str, Any]]
    rewritten_queries: list[str]
    documents: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    retrieval_round: int
    needs_more_retrieval: bool
    answer: str
    errors: list[str]
