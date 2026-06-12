from financial_agentic_rag.graphs.state import GraphState


def evidence_check(state: GraphState) -> GraphState:
    """Decide whether retrieved documents are enough to answer."""

    has_documents = bool(state.get("documents"))
    round_limit_reached = state.get("retrieval_round", 0) >= 2
    return {
        **state,
        "evidence": state.get("documents", []),
        "needs_more_retrieval": not has_documents and not round_limit_reached,
    }

