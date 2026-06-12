from __future__ import annotations

from functools import lru_cache
from threading import RLock
from typing import Any

from financial_agentic_rag.config import load_yaml
from financial_agentic_rag.retrievers.vectorstore import LocalRetriever


SUPPORTED_RETRIEVAL_TOOLS = {
    "semantic_search",
    "keyword_search",
    "hybrid_search",
    "knowledge_graph_search",
}


_RETRIEVER_LOCK = RLock()
_SEMANTIC_LOCK = RLock()


@lru_cache(maxsize=1)
def _get_cached_retriever() -> LocalRetriever:
    config = load_yaml("configs/retrieval_config.yaml")
    return LocalRetriever(config)


def clear_retriever_cache() -> None:
    """Clear cached retriever state, mainly for tests or index rebuilds."""

    with _RETRIEVER_LOCK:
        _get_cached_retriever.cache_clear()


def run_retrieval_tool(tool_name: str, query: str, top_k: int | None = None) -> dict[str, Any]:
    """Run one retrieval tool and return chunks plus execution metadata."""

    with _RETRIEVER_LOCK:
        retriever = _get_cached_retriever()
    requested_tool = tool_name
    warning = ""
    if tool_name not in SUPPORTED_RETRIEVAL_TOOLS:
        warning = f"Unknown tool '{tool_name}', fallback to hybrid_search."
        tool_name = "hybrid_search"

    if tool_name == "semantic_search":
        with _SEMANTIC_LOCK:
            chunks = retriever.semantic_search(query, top_k=top_k)
    elif tool_name == "keyword_search":
        chunks = retriever.keyword_search(query, top_k=top_k)
    elif tool_name == "hybrid_search":
        with _SEMANTIC_LOCK:
            chunks = retriever.hybrid_search(query, top_k=top_k)
    else:
        chunks = []
        warning = "knowledge_graph_search is reserved but not implemented in v1."

    return {
        "requested_tool": requested_tool,
        "tool": tool_name,
        "query": query,
        "chunks": chunks,
        "warning": warning,
    }
