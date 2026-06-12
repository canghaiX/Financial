from financial_agentic_rag.graphs.state import GraphState


def route_after_evidence_check(state: GraphState) -> str:
    """Route to another retrieval hop or final answer generation."""

    if state.get("needs_more_retrieval"):
        return "retrieve"
    return "generate_answer"
