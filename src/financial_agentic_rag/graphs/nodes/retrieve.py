from financial_agentic_rag.graphs.state import GraphState
from financial_agentic_rag.config import load_yaml
from financial_agentic_rag.retrievers.vectorstore import LocalRetriever


def retrieve(state: GraphState) -> GraphState:
    """Retrieve evidence candidates for the current queries."""

    config = load_yaml("configs/retrieval_config.yaml")
    retriever = LocalRetriever(config)
    queries = state.get("rewritten_queries") or [state.get("question", "")]
    documents = []
    seen = set()
    for query in queries:
        for document in retriever.search(query):
            chunk_id = document.get("chunk_id")
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                documents.append(document)

    return {
        **state,
        "documents": documents,
        "retrieval_round": state.get("retrieval_round", 0) + 1,
    }
