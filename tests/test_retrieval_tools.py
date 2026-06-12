import importlib
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from financial_agentic_rag.retrievers.vectorstore import LocalRetriever

retrieval_tools = importlib.import_module("financial_agentic_rag.tools.retrieval")
vectorstore = importlib.import_module("financial_agentic_rag.retrievers.vectorstore")


def test_hybrid_search_deduplicates(monkeypatch) -> None:
    retriever = LocalRetriever(
        {
            "output_paths": {"chunks_jsonl": "missing.jsonl"},
            "vectorstore": {"vectorstore_dir": "missing", "docstore_dir": "missing"},
            "retrieval": {"top_k": 5},
        }
    )

    chunk = {"chunk_id": "c1", "text": "测试", "score": 0.1}
    monkeypatch.setattr(retriever, "_semantic_candidates", lambda query, top_k=None: [dict(chunk, score=0.8)])
    monkeypatch.setattr(retriever, "_keyword_candidates", lambda query, top_k=None: [dict(chunk, score=0.3)])
    monkeypatch.setattr(
        retriever,
        "_rerank_candidates",
        lambda query, chunks, top_k=None, retrieval_tool="hybrid_search": chunks[: top_k or 5],
    )

    results = retriever.hybrid_search("测试")

    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["base_score"] == 0.8
    assert results[0]["retrieval_tool"] == "hybrid_search"


def test_build_vectorstore_uses_bge_embedding_endpoint(tmp_path, monkeypatch) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    chunks = [
        {"chunk_id": "c1", "text": "危险化学品", "document_title": "法一"},
        {"chunk_id": "c2", "text": "安全管理", "document_title": "法二"},
    ]
    chunks_path.write_text("\n".join(json.dumps(chunk, ensure_ascii=False) for chunk in chunks), encoding="utf-8")
    calls = []

    def fake_embed(texts):
        calls.append(list(texts))
        return np.asarray([[1.0, 0.0], [0.0, 1.0]][: len(texts)], dtype="float32")

    monkeypatch.setattr(vectorstore, "_embed_texts_with_vllm", fake_embed)
    monkeypatch.setattr(
        vectorstore,
        "_embedding_config",
        lambda: {"provider": "vllm", "model": "bge-m3"},
    )

    stats = vectorstore.build_vectorstore(
        {
            "output_paths": {"chunks_jsonl": str(chunks_path)},
            "retrieval": {"top_k": 5},
            "vectorstore": {
                "backend": "faiss",
                "vectorstore_dir": str(tmp_path / "vectorstore"),
                "docstore_dir": str(tmp_path / "docstore"),
                "embedding_batch_size": 2,
                "normalize_embeddings": True,
            },
        }
    )

    assert stats["chunks"] == 2
    assert calls and len(calls[0]) == 2
    index_config = json.loads((tmp_path / "vectorstore" / "index_config.json").read_text(encoding="utf-8"))
    assert index_config["embedding_model"] == "bge-m3"


def test_rerank_candidates_orders_and_limits_results(monkeypatch) -> None:
    retriever = LocalRetriever(
        {
            "output_paths": {"chunks_jsonl": "missing.jsonl"},
            "retrieval": {"final_top_k": 2, "enable_reranker": True},
            "vectorstore": {"vectorstore_dir": "missing", "docstore_dir": "missing"},
        }
    )

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(
                {
                    "results": [
                        {"index": 1, "relevance_score": 0.9},
                        {"index": 0, "relevance_score": 0.5},
                    ]
                }
            ).encode("utf-8")

    monkeypatch.setattr(
        vectorstore,
        "_reranker_config",
        lambda: {"base_url": "http://127.0.0.1:8002", "api_key": "EMPTY", "model": "bge-reranker-v2-m3"},
    )
    monkeypatch.setattr(vectorstore.urlrequest, "urlopen", lambda req, timeout=60: FakeResponse())

    results = retriever._rerank_candidates(
        "query",
        [
            {"chunk_id": "c1", "text": "one", "score": 0.1},
            {"chunk_id": "c2", "text": "two", "score": 0.2},
            {"chunk_id": "c3", "text": "three", "score": 0.3},
        ],
        top_k=2,
        retrieval_tool="semantic_search",
    )

    assert [chunk["chunk_id"] for chunk in results] == ["c2", "c1"]
    assert results[0]["rerank_score"] == 0.9
    assert results[0]["score"] == 0.9


def test_hybrid_search_uses_rrf_before_rerank(monkeypatch) -> None:
    retriever = LocalRetriever(
        {
            "output_paths": {"chunks_jsonl": "missing.jsonl"},
            "retrieval": {"candidate_top_k": 20, "final_top_k": 5, "rrf_k": 60},
            "vectorstore": {"vectorstore_dir": "missing", "docstore_dir": "missing"},
        }
    )
    semantic = [
        {"chunk_id": "a", "text": "a", "score": 0.9},
        {"chunk_id": "same", "text": "same", "score": 0.8},
    ]
    keyword = [
        {"chunk_id": "same", "text": "same", "score": 0.7},
        {"chunk_id": "b", "text": "b", "score": 0.6},
    ]
    captured = {}

    monkeypatch.setattr(retriever, "_semantic_candidates", lambda query, top_k=None: semantic)
    monkeypatch.setattr(retriever, "_keyword_candidates", lambda query, top_k=None: keyword)

    def fake_rerank(query, chunks, top_k=None, retrieval_tool="hybrid_search"):
        captured["chunks"] = chunks
        return chunks[: top_k or 5]

    monkeypatch.setattr(retriever, "_rerank_candidates", fake_rerank)

    results = retriever.hybrid_search("query")

    assert results[0]["chunk_id"] == "same"
    assert results[0]["retrieval_tool"] == "hybrid_search"
    assert len({chunk["chunk_id"] for chunk in captured["chunks"]}) == 3


def test_run_retrieval_tool_reuses_cached_retriever_across_threads(monkeypatch) -> None:
    retrieval_tools.clear_retriever_cache()
    init_count = 0

    class FakeRetriever:
        def __init__(self, config):
            nonlocal init_count
            init_count += 1

        def semantic_search(self, query, top_k=None):
            return [{"chunk_id": query, "text": query}]

        def keyword_search(self, query, top_k=None):
            return [{"chunk_id": query, "text": query}]

        def hybrid_search(self, query, top_k=None):
            return [{"chunk_id": query, "text": query}]

    monkeypatch.setattr(retrieval_tools, "load_yaml", lambda path: {"retrieval": {"top_k": 1}})
    monkeypatch.setattr(retrieval_tools, "LocalRetriever", FakeRetriever)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(
                lambda index: retrieval_tools.run_retrieval_tool("semantic_search", f"q{index}"),
                range(8),
            )
        )

    assert init_count == 1
    assert len(results) == 8
    assert all(result["chunks"] for result in results)
    retrieval_tools.clear_retriever_cache()
