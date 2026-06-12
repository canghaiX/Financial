"""Run the LangGraph Agentic-RAG workflow for a single question."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from financial_agentic_rag.runtime.app import create_app
from financial_agentic_rag.tracing import new_run_id, write_trace


def _node_label(node_name: str) -> str:
    labels = {
        "router": "router 判断问题类型",
        "simple_answer": "simple_answer 检索并生成答案",
        "planner": "planner 规划子任务",
        "executer": "executer 调用检索工具",
        "verifier": "verifier 检查证据",
        "synthesizer": "synthesizer 合成最终答案",
    }
    return labels.get(node_name, node_name)


def main() -> None:
    app = create_app()
    question = " ".join(sys.argv[1:]).strip() or "危险化学品安全法的适用范围是什么？"
    inputs = {"question": question, "max_iterations": 5, "run_id": new_run_id(), "stream_answer": True}
    result = {}
    printed_answer = False
    print(f"Question: {question}\n")
    for update in app.stream(inputs, stream_mode="updates"):
        for node_name, node_state in update.items():
            result = node_state
            print(f"[{_node_label(node_name)}]")
            if node_name in {"simple_answer", "synthesizer"} and node_state.get("answer"):
                print("\nAnswer:")
                deltas = node_state.get("answer_deltas", [])
                if deltas:
                    for delta in deltas:
                        print(delta, end="", flush=True)
                    print()
                else:
                    print(node_state["answer"])
                printed_answer = True
    trace_path = write_trace(result)
    result["trace_path"] = str(trace_path)
    result["trace_markdown_path"] = str(trace_path.with_suffix(".md"))
    if not printed_answer:
        print("\nAnswer:")
        print(result.get("answer", result))
    print(f"\nTrace JSON: {trace_path}")
    print(f"Trace Markdown: {trace_path.with_suffix('.md')}")


if __name__ == "__main__":
    main()
