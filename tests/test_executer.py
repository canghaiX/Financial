import importlib

executer_agent = importlib.import_module("financial_agentic_rag.agents.executer")


def test_executer_runs_all_plan_steps_in_one_round(monkeypatch) -> None:
    calls = []

    def fake_tool(tool_name, query, top_k=None):
        calls.append((tool_name, query, top_k))
        return {
            "requested_tool": tool_name,
            "tool": tool_name,
            "query": query,
            "warning": "",
            "chunks": [{"chunk_id": query, "text": query}],
        }

    monkeypatch.setattr(executer_agent, "run_retrieval_tool", fake_tool)
    monkeypatch.setattr(
        executer_agent,
        "load_yaml",
        lambda path: {"retrieval": {"top_k": 5, "rerank_top_k": 4}},
    )

    state = executer_agent.executer(
        {
            "question": "复杂问题",
            "iteration": 2,
            "plan_steps": [
                {"step_id": "s1", "tool": "semantic_search", "query": "问题1"},
                {"step_id": "s2", "tool": "keyword_search", "query": "问题2"},
                {"step_id": "s3", "tool": "hybrid_search", "query": "问题3"},
            ],
        }
    )

    assert [call[0] for call in sorted(calls)] == ["hybrid_search", "keyword_search", "semantic_search"]
    assert any("问题1" in query for _, query, _ in calls)
    assert state["iteration"] == 3
    assert len(state["retrieved_chunks"]) == 3
    assert len(state["executed_steps"]) == 3
    assert len(state["task_results"]) == 3
    assert state["task_results"][0]["step_id"] == "s1"
    assert state["task_results"][0]["result_count"] == 1
    assert state["task_results"][0]["original_query"] == "问题1"
    assert "问题1" in state["task_results"][0]["effective_query"]
    assert state["task_results"][0]["top_k"] == 5
    assert state["trace_events"][-1]["payload"]["parallel"] is True
    assert state["trace_events"][-1]["payload"]["worker_count"] == 3


def test_executer_dedupes_chunks_across_parallel_steps(monkeypatch) -> None:
    def fake_tool(tool_name, query, top_k=None):
        return {
            "requested_tool": tool_name,
            "tool": tool_name,
            "query": query,
            "warning": "",
            "chunks": [{"chunk_id": "same", "text": query}],
        }

    monkeypatch.setattr(executer_agent, "run_retrieval_tool", fake_tool)
    monkeypatch.setattr(executer_agent, "load_yaml", lambda path: {"retrieval": {"top_k": 5, "rerank_top_k": 4}})

    state = executer_agent.executer(
        {
            "question": "复杂问题",
            "plan_steps": [
                {"step_id": "s1", "tool": "semantic_search", "query": "问题1"},
                {"step_id": "s2", "tool": "keyword_search", "query": "问题2"},
            ],
        }
    )

    assert len(state["retrieved_chunks"]) == 1
    assert state["retrieved_chunks"][0]["chunk_id"] == "same"


def test_executer_keeps_successful_results_when_one_step_fails(monkeypatch) -> None:
    def fake_tool(tool_name, query, top_k=None):
        if "失败问题" in query:
            raise RuntimeError("boom")
        return {
            "requested_tool": tool_name,
            "tool": tool_name,
            "query": query,
            "warning": "",
            "chunks": [{"chunk_id": "ok", "text": "成功证据"}],
        }

    monkeypatch.setattr(executer_agent, "run_retrieval_tool", fake_tool)
    monkeypatch.setattr(executer_agent, "load_yaml", lambda path: {"retrieval": {"top_k": 5, "rerank_top_k": 4}})

    state = executer_agent.executer(
        {
            "question": "复杂问题",
            "plan_steps": [
                {"step_id": "s1", "tool": "semantic_search", "query": "成功问题"},
                {"step_id": "s2", "tool": "keyword_search", "query": "失败问题"},
            ],
        }
    )

    assert state["retrieved_chunks"][0]["chunk_id"] == "ok"
    assert state["retrieved_chunks"][0]["text"] == "成功证据"
    assert len(state["executed_steps"]) == 2
    assert len(state["task_results"]) == 2
    assert state["task_results"][1]["error"] == "RuntimeError: boom"
    assert state["executed_steps"][1]["error"] == "RuntimeError: boom"
    assert state["tool_history"][1]["error"] == "RuntimeError: boom"
    assert state["errors"] == ["executer step s2 failed: RuntimeError: boom"]
    assert state["trace_events"][-1]["payload"]["steps"][1]["error"] == "RuntimeError: boom"


def test_executer_uses_tool_specific_and_pending_top_k(monkeypatch) -> None:
    calls = []

    def fake_tool(tool_name, query, top_k=None):
        calls.append((tool_name, top_k))
        return {
            "requested_tool": tool_name,
            "tool": tool_name,
            "query": query,
            "warning": "",
            "chunks": [{"chunk_id": f"{tool_name}-{top_k}", "text": "证据"}],
        }

    monkeypatch.setattr(executer_agent, "run_retrieval_tool", fake_tool)
    monkeypatch.setattr(executer_agent, "load_yaml", lambda path: {"retrieval": {"top_k": 5, "rerank_top_k": 4}})

    state = executer_agent.executer(
        {
            "question": "复杂问题",
            "pending_tasks": [{"step_id": "s3"}],
            "plan_steps": [
                {"step_id": "s1", "tool": "semantic_search", "query": "语义"},
                {"step_id": "s2", "tool": "keyword_search", "query": "关键词"},
                {"step_id": "s3", "tool": "hybrid_search", "query": "补充"},
            ],
        }
    )

    assert sorted(calls) == [("hybrid_search", 5), ("keyword_search", 5), ("semantic_search", 5)]
    assert [task["top_k"] for task in state["task_results"]] == [5, 5, 5]


def test_executer_filters_empty_chunks_and_keeps_retriever_order(monkeypatch) -> None:
    def fake_tool(tool_name, query, top_k=None):
        return {
            "requested_tool": tool_name,
            "tool": tool_name,
            "query": query,
            "warning": "",
            "chunks": [
                {"chunk_id": "empty", "text": "", "score": 10.0, "document_title": "中华人民共和国危险化学品安全法"},
                {"chunk_id": None, "text": "无 id", "score": 9.0, "document_title": "中华人民共和国危险化学品安全法"},
                {
                    "chunk_id": "related",
                    "text": "重大危险源 管控 备案 义务",
                    "score": 0.1,
                    "document_title": "中华人民共和国危险化学品安全法",
                    "chapter_title": "重大危险源",
                },
                {
                    "chunk_id": "unrelated",
                    "text": "其他内容",
                    "score": 0.2,
                    "document_title": "中华人民共和国食品安全法",
                    "chapter_title": "其他",
                },
            ],
        }

    monkeypatch.setattr(executer_agent, "run_retrieval_tool", fake_tool)
    monkeypatch.setattr(executer_agent, "load_yaml", lambda path: {"retrieval": {"top_k": 5, "rerank_top_k": 4}})

    state = executer_agent.executer(
        {
            "question": "《危险化学品安全法》重大危险源管控义务是什么？",
            "plan_steps": [
                {
                    "step_id": "s1",
                    "tool": "hybrid_search",
                    "sub_question": "重大危险源管控义务",
                    "query": "重大危险源 管控",
                }
            ],
        }
    )

    chunks = state["task_results"][0]["chunks"]
    assert [chunk["chunk_id"] for chunk in chunks] == ["unrelated", "related"]
