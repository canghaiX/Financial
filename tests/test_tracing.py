import json
import importlib

from financial_agentic_rag.tracing import append_trace_event, summarize_chunks, write_trace

router_agent = importlib.import_module("financial_agentic_rag.agents.router")


def test_append_trace_event_records_node_and_payload() -> None:
    events = append_trace_event(
        {"question": "测试", "iteration": 1},
        "planner",
        "plan_created",
        {"steps": []},
    )

    assert len(events) == 1
    assert events[0]["node"] == "planner"
    assert events[0]["event_type"] == "plan_created"
    assert events[0]["iteration"] == 1
    assert events[0]["round"] == 1
    assert events[0]["payload"] == {"steps": []}


def test_summarize_chunks_uses_preview_not_full_text() -> None:
    long_text = "法" * 400
    summary = summarize_chunks(
        [
            {
                "chunk_id": "c1",
                "document_title": "测试法",
                "chapter_title": "第一章",
                "page_start": 1,
                "page_end": 2,
                "score": 0.9,
                "retrieval_tool": "hybrid_search",
                "text": long_text,
            }
        ],
        preview_chars=20,
    )

    assert summary[0]["text_preview"].endswith("...")
    assert len(summary[0]["text_preview"]) < len(long_text)
    assert "text" not in summary[0]


def test_router_writes_trace_event(monkeypatch) -> None:
    monkeypatch.setattr(
        router_agent,
        "chat_completion",
        lambda messages: '{"query_type":"simple","reason":"单一事实","confidence":0.9}',
    )

    state = router_agent.router({"question": "危险化学品安全法的适用范围是什么？", "run_id": "test-run"})

    assert state["trace_events"]
    assert state["trace_events"][-1]["node"] == "router"
    assert state["trace_events"][-1]["event_type"] == "route_decision"


def test_write_trace_writes_json(tmp_path) -> None:
    path = write_trace(
        {
            "run_id": "test-run",
            "question": "测试问题",
            "route_type": "simple",
            "iteration": 1,
            "max_iterations": 5,
            "answer": "测试答案",
            "errors": [],
            "trace_events": [{"node": "router"}],
        },
        traces_dir=tmp_path,
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["run_id"] == "test-run"
    assert data["question"] == "测试问题"
    assert data["trace_markdown_path"] == str(path.with_suffix(".md"))
    assert data["events"] == [{"node": "router", "iteration": 0, "round": 0, "payload": {}}]
    assert path.with_suffix(".md").exists()


def test_write_trace_writes_readable_markdown_summary(tmp_path) -> None:
    path = write_trace(
        {
            "run_id": "trace-rich",
            "question": "复杂法律问题",
            "query_type": "multi_hop",
            "route_type": "multi_hop",
            "iteration": 1,
            "max_iterations": 5,
            "answer": "最终答案",
            "errors": [],
            "trace_events": [
                {
                    "node": "router",
                    "event_type": "route_decision",
                    "iteration": 0,
                    "payload": {"query_type": "multi_hop", "reason": "需要多跳", "confidence": 0.8},
                },
                {
                    "node": "planner",
                    "event_type": "plan_created",
                    "iteration": 0,
                    "payload": {
                        "steps": [
                            {
                                "step_id": "step_1",
                                "sub_question": "子问题一",
                                "tool": "semantic_search",
                                "query": "查询一",
                                "reason": "定位义务条款",
                            }
                        ]
                    },
                },
                {
                    "node": "executer",
                    "event_type": "tools_executed",
                    "iteration": 1,
                    "payload": {
                        "parallel": True,
                        "worker_count": 2,
                        "retrieved_total": 3,
                        "steps": [
                            {
                                "step_id": "step_1",
                                "tool": "semantic_search",
                                "query": "查询一",
                                "result_count": 3,
                                "warning": "",
                                "error": "",
                            }
                        ],
                    },
                },
                {
                    "node": "verifier",
                    "event_type": "evidence_checked",
                    "iteration": 1,
                    "payload": {
                        "is_sufficient": False,
                        "reason": "缺少责任条款",
                        "missing_evidence": ["事故责任证据块"],
                        "suggested_queries": ["事故责任 条款"],
                        "suggested_tools": "keyword_search",
                        "task_verifications": [
                            {
                                "step_id": "step_1",
                                "sub_question": "子问题一",
                                "is_sufficient": False,
                                "reason": "缺少责任条款",
                                "missing_evidence": ["事故责任证据块"],
                                "suggested_queries": ["事故责任 条款"],
                                "suggested_tools": "keyword_search",
                            }
                        ],
                    },
                },
            ],
        },
        traces_dir=tmp_path,
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    markdown = path.with_suffix(".md").read_text(encoding="utf-8")

    assert data["events"][-1]["payload"]["suggested_tools"] == ["keyword_search"]
    assert "## Round 1" in markdown
    assert "### Planner" in markdown
    assert "### Executer" in markdown
    assert "### Verifier" in markdown
    assert "子问题：子问题一" in markdown
    assert "工具：`semantic_search`" in markdown
    assert "还需要找的证据块" in markdown
    assert "事故责任证据块" in markdown
    assert "建议工具" in markdown
    assert "子任务验证" in markdown
    assert "keyword_search" in markdown
