import importlib

simple_answer_agent = importlib.import_module("financial_agentic_rag.agents.simple_answer")


def test_simple_answer_uses_semantic_search(monkeypatch) -> None:
    calls = []

    def fake_tool(tool_name, query, top_k=None):
        calls.append((tool_name, query))
        return {"tool": tool_name, "query": query, "warning": "", "chunks": []}

    monkeypatch.setattr(simple_answer_agent, "run_retrieval_tool", fake_tool)
    monkeypatch.setattr(simple_answer_agent, "complete_answer", lambda messages, stream_answer: ("答案", []))

    state = simple_answer_agent.simple_answer({"question": "危险化学品安全法的适用范围是什么？"})

    assert calls == [("semantic_search", "危险化学品安全法的适用范围是什么？")]
    assert state["answer"] == "答案"
    assert state["trace_events"][-1]["payload"]["tool"] == "semantic_search"
