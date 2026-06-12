import importlib

from financial_agentic_rag.llms.client import LLMClientError

planner_agent = importlib.import_module("financial_agentic_rag.agents.planner")
verifier_agent = importlib.import_module("financial_agentic_rag.agents.verifier")


def test_planner_normalizes_multiple_steps(monkeypatch) -> None:
    monkeypatch.setattr(
        planner_agent,
        "chat_completion",
        lambda messages: (
            '{"steps":['
            '{"step_id":"s1","sub_question":"查义务","tool":"semantic_search","query":"安全义务","reason":"归纳义务"},'
            '{"step_id":"s2","sub_question":"查责任","tool":"keyword_search","query":"法律责任 处罚","reason":"精确术语"}'
            ']}'
        ),
    )

    state = planner_agent.planner({"question": "复杂法律问题"})

    assert [step["step_id"] for step in state["plan_steps"]] == ["s1", "s2"]
    assert state["plan_steps"][0]["tool"] == "semantic_search"
    assert state["plan_steps"][1]["tool"] == "keyword_search"


def test_planner_limits_first_round_to_four_steps(monkeypatch) -> None:
    monkeypatch.setattr(
        planner_agent,
        "chat_completion",
        lambda messages: (
            '{"steps":['
            '{"step_id":"s1","sub_question":"一","tool":"hybrid_search","query":"一","reason":"r"},'
            '{"step_id":"s2","sub_question":"二","tool":"hybrid_search","query":"二","reason":"r"},'
            '{"step_id":"s3","sub_question":"三","tool":"hybrid_search","query":"三","reason":"r"},'
            '{"step_id":"s4","sub_question":"四","tool":"hybrid_search","query":"四","reason":"r"},'
            '{"step_id":"s5","sub_question":"五","tool":"hybrid_search","query":"五","reason":"r"}'
            ']}'
        ),
    )

    state = planner_agent.planner({"question": "复杂法律问题"})

    assert [step["step_id"] for step in state["plan_steps"]] == ["s1", "s2", "s3", "s4"]


def test_planner_limits_pending_round_to_two_steps(monkeypatch) -> None:
    monkeypatch.setattr(
        planner_agent,
        "chat_completion",
        lambda messages: (
            '{"steps":['
            '{"step_id":"s1","sub_question":"一","tool":"hybrid_search","query":"一","reason":"r"},'
            '{"step_id":"s2","sub_question":"二","tool":"hybrid_search","query":"二","reason":"r"},'
            '{"step_id":"s3","sub_question":"三","tool":"hybrid_search","query":"三","reason":"r"}'
            ']}'
        ),
    )

    state = planner_agent.planner({"question": "复杂法律问题", "pending_tasks": [{"step_id": "s1"}]})

    assert [step["step_id"] for step in state["plan_steps"]] == ["s1", "s2"]


def test_planner_uses_pending_task_for_fallback(monkeypatch) -> None:
    def raise_error(messages):
        raise LLMClientError("down")

    monkeypatch.setattr(planner_agent, "chat_completion", raise_error)

    state = planner_agent.planner(
        {
            "question": "复杂法律问题",
            "pending_tasks": [
                {
                    "step_id": "s2",
                    "sub_question": "查责任",
                    "suggested_queries": ["事故责任 条款"],
                    "suggested_tools": ["keyword_search"],
                }
            ],
        }
    )

    assert state["plan_steps"][0]["step_id"] == "s2"
    assert state["plan_steps"][0]["query"] == "事故责任 条款"
    assert state["plan_steps"][0]["tool"] == "keyword_search"


def test_verifier_marks_all_tasks_sufficient(monkeypatch) -> None:
    monkeypatch.setattr(
        verifier_agent,
        "chat_completion",
        lambda messages: (
            '{"is_sufficient":true,"reason":"证据已覆盖核心条款，缺少检查频率不影响回答",'
            '"missing_evidence":["检查频率细节"],'
            '"suggested_queries":["继续查频率"],'
            '"suggested_tools":["hybrid_search"]}'
        ),
    )

    state = verifier_agent.verifier(
        {
            "question": "复杂法律问题",
            "plan_steps": [{"step_id": "s1"}],
            "task_results": [
                {
                    "step_id": "s1",
                    "sub_question": "查义务",
                    "tool": "semantic_search",
                    "query": "安全义务",
                    "chunks": [{"chunk_id": "c1", "text": "证据"}],
                    "error": "",
                }
            ],
            "retrieved_chunks": [{"chunk_id": "c1", "text": "证据"}],
        }
    )

    assert state["verifier_feedback"]["is_sufficient"] is True
    assert state["pending_tasks"] == []
    assert state["verified_evidence"] == [{"chunk_id": "c1", "text": "证据"}]
    assert state["task_verifications"][0]["missing_evidence"] == []
    assert state["task_verifications"][0]["suggested_queries"] == []
    assert state["task_verifications"][0]["suggested_tools"] == []


def test_verifier_prompt_tells_model_to_avoid_over_strict_checks(monkeypatch) -> None:
    captured = {}

    def fake_chat(messages):
        captured["system"] = messages[0]["content"]
        return '{"is_sufficient":true,"reason":"够回答","missing_evidence":[],"suggested_queries":[],"suggested_tools":[]}'

    monkeypatch.setattr(verifier_agent, "chat_completion", fake_chat)

    verifier_agent.verifier(
        {
            "question": "复杂法律问题",
            "plan_steps": [{"step_id": "s1"}],
            "task_results": [
                {
                    "step_id": "s1",
                    "sub_question": "查义务",
                    "tool": "semantic_search",
                    "query": "安全义务",
                    "chunks": [{"chunk_id": "c1", "text": "证据"}],
                    "error": "",
                }
            ],
            "retrieved_chunks": [{"chunk_id": "c1", "text": "证据"}],
        }
    )

    assert "不要追求法规百科式完整覆盖" in captured["system"]
    assert "够回答即可" in captured["system"]


def test_verifier_creates_pending_tasks_for_unsatisfied_subtasks(monkeypatch) -> None:
    def fake_chat(messages):
        content = messages[-1]["content"]
        if "查义务" in content:
            return '{"is_sufficient":true,"reason":"足够","missing_evidence":[],"suggested_queries":[],"suggested_tools":[]}'
        return (
            '{"is_sufficient":false,"reason":"缺少责任条款",'
            '"missing_evidence":["事故责任证据块"],'
            '"suggested_queries":["事故责任 条款"],'
            '"suggested_tools":"keyword_search"}'
        )

    monkeypatch.setattr(verifier_agent, "chat_completion", fake_chat)

    state = verifier_agent.verifier(
        {
            "question": "复杂法律问题",
            "plan_steps": [{"step_id": "s1"}, {"step_id": "s2"}],
            "task_results": [
                {
                    "step_id": "s1",
                    "sub_question": "查义务",
                    "tool": "semantic_search",
                    "query": "安全义务",
                    "chunks": [{"chunk_id": "c1", "text": "义务证据"}],
                    "error": "",
                },
                {
                    "step_id": "s2",
                    "sub_question": "查责任",
                    "tool": "keyword_search",
                    "query": "法律责任",
                    "chunks": [{"chunk_id": "c2", "text": "责任证据不足"}],
                    "error": "",
                },
            ],
            "retrieved_chunks": [
                {"chunk_id": "c1", "text": "义务证据"},
                {"chunk_id": "c2", "text": "责任证据不足"},
            ],
        }
    )

    assert state["verifier_feedback"]["is_sufficient"] is False
    assert len(state["task_verifications"]) == 2
    assert state["pending_tasks"] == [
        {
            "step_id": "s2",
            "sub_question": "查责任",
            "missing_evidence": ["事故责任证据块"],
            "suggested_queries": ["事故责任 条款"],
            "suggested_tools": ["keyword_search"],
            "reason": "缺少责任条款",
        }
    ]


def test_verifier_final_round_sends_existing_evidence_to_synthesizer(monkeypatch) -> None:
    monkeypatch.setattr(
        verifier_agent,
        "chat_completion",
        lambda messages: (
            '{"is_sufficient":false,"reason":"还缺细节",'
            '"missing_evidence":["细节"],"suggested_queries":["细节"],"suggested_tools":["hybrid_search"]}'
        ),
    )

    state = verifier_agent.verifier(
        {
            "question": "复杂法律问题",
            "iteration": 2,
            "max_iterations": 2,
            "plan_steps": [{"step_id": "s1"}],
            "task_results": [
                {
                    "step_id": "s1",
                    "sub_question": "查义务",
                    "tool": "semantic_search",
                    "query": "安全义务",
                    "chunks": [{"chunk_id": "c1", "text": "证据"}],
                    "error": "",
                }
            ],
            "retrieved_chunks": [{"chunk_id": "c1", "text": "证据"}],
        }
    )

    assert state["verifier_feedback"]["is_sufficient"] is True
    assert state["pending_tasks"] == []


def test_verifier_parallel_trace_and_stable_order(monkeypatch) -> None:
    calls = []

    def fake_chat(messages):
        content = messages[-1]["content"]
        calls.append(content)
        if "第二个任务" in content:
            return '{"is_sufficient":false,"reason":"缺第二项证据","missing_evidence":["第二项"],"suggested_queries":["第二项 查询"],"suggested_tools":["keyword_search"]}'
        return '{"is_sufficient":true,"reason":"足够","missing_evidence":[],"suggested_queries":[],"suggested_tools":[]}'

    monkeypatch.setattr(verifier_agent, "chat_completion", fake_chat)

    state = verifier_agent.verifier(
        {
            "question": "复杂法律问题",
            "plan_steps": [{"step_id": "s1"}, {"step_id": "s2"}, {"step_id": "s3"}],
            "task_results": [
                {"step_id": "s1", "sub_question": "第一个任务", "tool": "semantic_search", "query": "q1", "chunks": [{"chunk_id": "c1", "text": "证据"}]},
                {"step_id": "s2", "sub_question": "第二个任务", "tool": "keyword_search", "query": "q2", "chunks": [{"chunk_id": "c2", "text": "证据"}]},
                {"step_id": "s3", "sub_question": "第三个任务", "tool": "hybrid_search", "query": "q3", "chunks": [{"chunk_id": "c3", "text": "证据"}]},
            ],
            "retrieved_chunks": [
                {"chunk_id": "c1", "text": "证据"},
                {"chunk_id": "c2", "text": "证据"},
                {"chunk_id": "c3", "text": "证据"},
            ],
        }
    )

    assert len(calls) == 3
    assert [item["step_id"] for item in state["task_verifications"]] == ["s1", "s2", "s3"]
    assert state["pending_tasks"][0]["step_id"] == "s2"
    payload = state["trace_events"][-1]["payload"]
    assert payload["parallel"] is True
    assert payload["worker_count"] == 3
    assert payload["verified_task_count"] == 3


def test_verifier_keeps_other_results_when_one_task_verification_fails(monkeypatch) -> None:
    def fake_verify(question, task):
        if task["step_id"] == "s2":
            raise RuntimeError("verifier boom")
        return {
            "step_id": task["step_id"],
            "sub_question": task["sub_question"],
            "query": task["query"],
            "tool": task["tool"],
            "is_sufficient": True,
            "reason": "足够",
            "missing_evidence": [],
            "suggested_queries": [],
            "suggested_tools": [],
            "checked_chunk_count": len(task["chunks"]),
        }

    monkeypatch.setattr(verifier_agent, "_verify_one_task", fake_verify)

    state = verifier_agent.verifier(
        {
            "question": "复杂法律问题",
            "plan_steps": [{"step_id": "s1"}, {"step_id": "s2"}],
            "task_results": [
                {"step_id": "s1", "sub_question": "查义务", "tool": "semantic_search", "query": "q1", "chunks": [{"chunk_id": "c1", "text": "证据"}]},
                {"step_id": "s2", "sub_question": "查责任", "tool": "keyword_search", "query": "q2", "chunks": [{"chunk_id": "c2", "text": "证据"}]},
            ],
            "retrieved_chunks": [
                {"chunk_id": "c1", "text": "证据"},
                {"chunk_id": "c2", "text": "证据"},
            ],
        }
    )

    assert state["task_verifications"][0]["is_sufficient"] is True
    assert state["task_verifications"][1]["is_sufficient"] is False
    assert state["pending_tasks"][0]["step_id"] == "s2"
    assert state["errors"] == ["verifier step s2 failed: RuntimeError: verifier boom"]
