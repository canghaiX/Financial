from langgraph.graph import END, START, StateGraph

from financial_agentic_rag.graphs.edges.agentic import route_after_router, route_after_verifier
from financial_agentic_rag.graphs.nodes.agentic import (
    executer,
    planner,
    router,
    simple_answer,
    synthesizer,
    verifier,
)
from financial_agentic_rag.graphs.state import GraphState


def build_graph():
    """Build and compile the Agentic-RAG LangGraph app."""

    graph = StateGraph(GraphState)
    graph.add_node("router", router)
    graph.add_node("simple_answer", simple_answer)
    graph.add_node("planner", planner)
    graph.add_node("executer", executer)
    graph.add_node("verifier", verifier)
    graph.add_node("synthesizer", synthesizer)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "simple_answer": "simple_answer",
            "planner": "planner",
        },
    )
    graph.add_edge("simple_answer", END)
    graph.add_edge("planner", "executer")
    graph.add_edge("executer", "verifier")
    graph.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {
            "synthesizer": "synthesizer",
            "planner": "planner",
        },
    )
    graph.add_edge("synthesizer", END)
    return graph.compile()
