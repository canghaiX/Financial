import importlib

from financial_agentic_rag.graphs.edges.agentic import route_after_router, route_after_verifier
from financial_agentic_rag.graphs.nodes import agentic
from financial_agentic_rag.llms.client import LLMClientError

router_agent = importlib.import_module("financial_agentic_rag.agents.router")


def test_router_routes_simple_question(monkeypatch) -> None:
    monkeypatch.setattr(
        router_agent,
        "chat_completion",
        lambda messages: '{"query_type":"simple","reason":"单一事实","confidence":0.9}',
    )

    state = router_agent.router({"question": "危险化学品安全法的适用范围是什么？"})

    assert state["query_type"] == "simple"
    assert state["route_type"] == "simple"
    assert route_after_router(state) == "simple_answer"


def test_router_routes_multi_hop_question(monkeypatch) -> None:
    monkeypatch.setattr(
        router_agent,
        "chat_completion",
        lambda messages: '{"query_type":"multi_hop","reason":"需要比较","confidence":0.8}',
    )

    state = router_agent.router({"question": "危险化学品安全法和民法典在企业责任方面有什么区别？"})

    assert state["query_type"] == "multi_hop"
    assert state["route_type"] == "multi_hop"
    assert route_after_router(state) == "planner"


def test_router_defaults_to_multi_hop_on_invalid_json(monkeypatch) -> None:
    monkeypatch.setattr(router_agent, "chat_completion", lambda messages: "不是 JSON")

    state = router_agent.router({"question": "危险化学品安全法的适用范围是什么？"})

    assert state["query_type"] == "multi_hop"
    assert state["errors"] == ["router returned invalid JSON or unknown query_type; defaulted to multi_hop"]
    assert route_after_router(state) == "planner"


def test_router_defaults_to_multi_hop_on_llm_error(monkeypatch) -> None:
    def raise_error(messages):
        raise LLMClientError("vllm down")

    monkeypatch.setattr(router_agent, "chat_completion", raise_error)

    state = router_agent.router({"question": "危险化学品安全法的适用范围是什么？"})

    assert state["query_type"] == "multi_hop"
    assert state["errors"] == ["vllm down"]
    assert route_after_router(state) == "planner"


def test_router_defaults_to_two_iterations_for_fast_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        router_agent,
        "chat_completion",
        lambda messages: '{"query_type":"multi_hop","reason":"需要规划","confidence":0.8}',
    )

    state = router_agent.router({"question": "复杂问题"})

    assert state["max_iterations"] == 2


def test_router_respects_explicit_max_iterations(monkeypatch) -> None:
    monkeypatch.setattr(
        router_agent,
        "chat_completion",
        lambda messages: '{"query_type":"multi_hop","reason":"需要规划","confidence":0.8}',
    )

    state = router_agent.router({"question": "复杂问题", "max_iterations": 4})

    assert state["max_iterations"] == 4


def test_verifier_routes_to_planner_when_evidence_insufficient() -> None:
    state = {"verifier_feedback": {"is_sufficient": False}, "iteration": 2, "max_iterations": 5}
    assert route_after_verifier(state) == "planner"


def test_verifier_routes_to_synthesizer_at_round_limit() -> None:
    state = {"verifier_feedback": {"is_sufficient": False}, "iteration": 5, "max_iterations": 5}
    assert route_after_verifier(state) == "synthesizer"


def test_verifier_routes_to_synthesizer_when_sufficient() -> None:
    state = {"verifier_feedback": {"is_sufficient": True}, "iteration": 1, "max_iterations": 5}
    assert route_after_verifier(state) == "synthesizer"


def test_agentic_compatibility_layer_exports_router() -> None:
    assert agentic.router is router_agent.router
