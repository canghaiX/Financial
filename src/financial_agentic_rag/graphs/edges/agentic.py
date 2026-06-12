from financial_agentic_rag.graphs.state import GraphState


def route_after_router(state: GraphState) -> str:
    return "simple_answer" if state.get("query_type") == "simple" else "planner"


def route_after_verifier(state: GraphState) -> str:
    feedback = state.get("verifier_feedback", {})
    if feedback.get("is_sufficient"):
        return "synthesizer"
    if state.get("iteration", 0) >= state.get("max_iterations", 2):
        return "synthesizer"
    return "planner"
