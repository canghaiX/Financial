import importlib

common_agent = importlib.import_module("financial_agentic_rag.agents.common")


def test_complete_answer_streams_when_enabled(monkeypatch) -> None:
    monkeypatch.setattr(common_agent, "stream_chat_completion", lambda messages: iter(["你", "好"]))

    answer, deltas = common_agent.complete_answer([{"role": "user", "content": "hi"}], stream_answer=True)

    assert answer == "你好"
    assert deltas == ["你", "好"]


def test_complete_answer_uses_non_streaming_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(common_agent, "chat_completion", lambda messages: "完整答案")

    answer, deltas = common_agent.complete_answer([{"role": "user", "content": "hi"}], stream_answer=False)

    assert answer == "完整答案"
    assert deltas == []
