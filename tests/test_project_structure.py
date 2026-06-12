from pathlib import Path


def test_core_directories_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative_path in [
        "configs",
        "prompts",
        "pdf",
        "src/financial_agentic_rag",
        "src/financial_agentic_rag/graphs/nodes",
        "src/financial_agentic_rag/graphs/edges",
        "storage/vectorstore",
        "storage/docstore",
    ]:
        assert (root / relative_path).exists()


def test_langgraph_builder_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "src/financial_agentic_rag/graphs/builder.py").exists()
