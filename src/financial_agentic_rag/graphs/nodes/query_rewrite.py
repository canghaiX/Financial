from financial_agentic_rag.graphs.state import GraphState


def query_rewrite(state: GraphState) -> GraphState:
    """Rewrite the user question into retrieval queries."""

    question = state.get("question", "")
    return {
        **state,
        "rewritten_queries": state.get("rewritten_queries") or [question],
    }

