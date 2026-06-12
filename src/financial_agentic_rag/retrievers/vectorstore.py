from __future__ import annotations

import json
import math
import pickle
import re
import os
from pathlib import Path
from typing import Any
from urllib import request as urlrequest
from urllib.error import URLError

import numpy as np
from dotenv import load_dotenv

from financial_agentic_rag.config import load_yaml, resolve_project_path
from financial_agentic_rag.utils.io import iter_jsonl, write_json


TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+")
DEFAULT_MODEL_CONFIG = "configs/model_config.yaml"


def _load_chunks(config: dict[str, Any]) -> list[dict[str, Any]]:
    chunks_path = resolve_project_path(
        config.get("output_paths", {}).get("chunks_jsonl", "data/processed/chunks.jsonl")
    )
    return list(iter_jsonl(chunks_path) or [])


def _text_for_embedding(chunk: dict[str, Any]) -> str:
    metadata = " ".join(
        str(chunk.get(key, ""))
        for key in ("document_title", "chapter_title", "chunk_type")
    )
    return f"{metadata}\n{chunk.get('text', '')}".strip()


def _simple_score(query: str, text: str) -> float:
    query_tokens = set(TOKEN_RE.findall(query.lower()))
    text_tokens = set(TOKEN_RE.findall(text.lower()))
    if not query_tokens or not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / math.sqrt(len(text_tokens))


def _model_config() -> dict[str, Any]:
    load_dotenv()
    return load_yaml(DEFAULT_MODEL_CONFIG)


def _embedding_config() -> dict[str, Any]:
    config = _model_config().get("embedding", {})
    return {
        "provider": config.get("provider", "vllm"),
        "base_url": os.getenv("BGE_EMBEDDING_BASE_URL", config.get("base_url", "http://127.0.0.1:8001/v1")),
        "api_key": os.getenv("BGE_EMBEDDING_API_KEY", config.get("api_key", "EMPTY")),
        "model": os.getenv("BGE_EMBEDDING_MODEL", config.get("model", "bge-m3")),
    }


def _reranker_config() -> dict[str, Any]:
    config = _model_config().get("reranker", {})
    return {
        "provider": config.get("provider", "vllm"),
        "base_url": os.getenv("BGE_RERANKER_BASE_URL", config.get("base_url", "http://127.0.0.1:8002")),
        "api_key": os.getenv("BGE_RERANKER_API_KEY", config.get("api_key", "EMPTY")),
        "model": os.getenv("BGE_RERANKER_MODEL", config.get("model", "bge-reranker-v2-m3")),
    }


def _embed_texts_with_vllm(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.asarray([], dtype="float32")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("BGE embedding requires openai. Run: pip install -r requirements.txt") from exc

    config = _embedding_config()
    client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
    response = client.embeddings.create(model=config["model"], input=texts)
    ordered = sorted(response.data, key=lambda item: item.index)
    return np.asarray([item.embedding for item in ordered], dtype="float32")


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _chunk_document(chunk: dict[str, Any]) -> str:
    return _text_for_embedding(chunk)


class LocalRetriever:
    """Retriever facade: bge-m3 coarse retrieval, optional keyword/RRF, then rerank."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        vector_cfg = config.get("vectorstore", {})
        self.vectorstore_dir = resolve_project_path(vector_cfg.get("vectorstore_dir", "storage/vectorstore"))
        self.docstore_dir = resolve_project_path(vector_cfg.get("docstore_dir", "storage/docstore"))
        self.chunks = _load_chunks(config)
        self.index = None
        self.id_map: list[str] = []
        self.by_id = {chunk["chunk_id"]: chunk for chunk in self.chunks if chunk.get("chunk_id")}
        self._load_faiss_if_available(vector_cfg)

    def _load_faiss_if_available(self, vector_cfg: dict[str, Any]) -> None:
        index_path = self.vectorstore_dir / "index.faiss"
        id_map_path = self.vectorstore_dir / "id_map.json"
        if not index_path.exists() or not id_map_path.exists():
            return
        try:
            import faiss
        except Exception:
            return
        self.index = faiss.read_index(str(index_path))
        self.id_map = json.loads(id_map_path.read_text(encoding="utf-8"))

    def _candidate_top_k(self) -> int:
        retrieval_cfg = self.config.get("retrieval", {})
        return int(retrieval_cfg.get("candidate_top_k", 20))

    def _final_top_k(self, top_k: int | None = None) -> int:
        retrieval_cfg = self.config.get("retrieval", {})
        return int(top_k or retrieval_cfg.get("final_top_k") or retrieval_cfg.get("top_k", 5))

    def _reranker_enabled(self) -> bool:
        return bool(self.config.get("retrieval", {}).get("enable_reranker", True))

    def _embed_query(self, query: str) -> np.ndarray:
        embedding = _embed_texts_with_vllm([query])
        if bool(self.config.get("vectorstore", {}).get("normalize_embeddings", True)):
            embedding = _normalize_vectors(embedding)
        return np.asarray(embedding, dtype="float32")

    def _semantic_candidates(self, query: str, candidate_top_k: int | None = None) -> list[dict[str, Any]]:
        candidate_top_k = candidate_top_k or self._candidate_top_k()
        if self.index is None:
            return []
        embedding = self._embed_query(query)
        scores, indices = self.index.search(embedding, candidate_top_k)
        results: list[dict[str, Any]] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self.id_map):
                continue
            chunk = dict(self.by_id.get(self.id_map[idx], {}))
            if chunk:
                chunk["score"] = float(score)
                chunk["base_score"] = float(score)
                chunk["semantic_rank"] = rank
                chunk["retrieval_tool"] = "semantic_search"
                results.append(chunk)
        return results

    def _keyword_candidates(self, query: str, candidate_top_k: int | None = None) -> list[dict[str, Any]]:
        candidate_top_k = candidate_top_k or self._candidate_top_k()
        ranked = sorted(
            (
                (_simple_score(query, _text_for_embedding(chunk)), chunk)
                for chunk in self.chunks
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        return [
            dict(
                chunk,
                score=float(score),
                base_score=float(score),
                keyword_rank=rank,
                retrieval_tool="keyword_search",
            )
            for rank, (score, chunk) in enumerate(ranked[:candidate_top_k], start=1)
            if score > 0
        ]

    def _rrf_fuse(
        self,
        semantic_chunks: list[dict[str, Any]],
        keyword_chunks: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        rrf_k = int(self.config.get("retrieval", {}).get("rrf_k", 60))
        candidate_top_k = top_k or self._candidate_top_k()
        fused: dict[str, dict[str, Any]] = {}

        for source, chunks in (("semantic", semantic_chunks), ("keyword", keyword_chunks)):
            for rank, chunk in enumerate(chunks, start=1):
                chunk_id = chunk.get("chunk_id")
                if not chunk_id:
                    continue
                current = fused.setdefault(
                    str(chunk_id),
                    {
                        **chunk,
                        "score": 0.0,
                        "base_score": float(chunk.get("score", 0) or 0),
                        "retrieval_sources": [],
                        "retrieval_tool": "hybrid_search",
                    },
                )
                current["score"] = float(current.get("score", 0) or 0) + 1.0 / (rrf_k + rank)
                current["rrf_score"] = current["score"]
                sources = list(current.get("retrieval_sources", []))
                if source not in sources:
                    sources.append(source)
                current["retrieval_sources"] = sources
                current[f"{source}_rank"] = rank
                if float(chunk.get("score", 0) or 0) > float(current.get("base_score", 0) or 0):
                    current["base_score"] = float(chunk.get("score", 0) or 0)

        return sorted(fused.values(), key=lambda item: float(item.get("score", 0)), reverse=True)[:candidate_top_k]

    def _rerank_candidates(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int | None = None,
        retrieval_tool: str = "semantic_search",
    ) -> list[dict[str, Any]]:
        final_top_k = self._final_top_k(top_k)
        candidates = [
            dict(chunk, retrieval_tool=retrieval_tool)
            for chunk in chunks
            if chunk.get("chunk_id") and str(chunk.get("text", "")).strip()
        ]
        if not candidates:
            return []
        if not self._reranker_enabled():
            return sorted(candidates, key=lambda item: float(item.get("score", 0) or 0), reverse=True)[:final_top_k]

        config = _reranker_config()
        endpoint = config["base_url"].rstrip("/") + "/rerank"
        payload = json.dumps(
            {
                "model": config["model"],
                "query": query,
                "documents": [_chunk_document(chunk) for chunk in candidates],
                "top_n": final_top_k,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        req = urlrequest.Request(
            endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config['api_key']}",
            },
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (OSError, URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Failed to call BGE reranker endpoint {endpoint}") from exc

        reranked: list[dict[str, Any]] = []
        for item in data.get("results", []):
            index = int(item.get("index", -1))
            if index < 0 or index >= len(candidates):
                continue
            relevance_score = float(item.get("relevance_score", 0) or 0)
            chunk = dict(candidates[index])
            chunk["base_score"] = float(chunk.get("score", 0) or 0)
            chunk["rerank_score"] = relevance_score
            chunk["score"] = relevance_score
            chunk["retrieval_tool"] = retrieval_tool
            reranked.append(chunk)
        return reranked[:final_top_k]

    def semantic_search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        candidates = self._semantic_candidates(query, self._candidate_top_k())
        return self._rerank_candidates(query, candidates, top_k=top_k, retrieval_tool="semantic_search")

    def keyword_search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        candidates = self._keyword_candidates(query, self._candidate_top_k())
        return self._rerank_candidates(query, candidates, top_k=top_k, retrieval_tool="keyword_search")

    def hybrid_search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        semantic = self._semantic_candidates(query, self._candidate_top_k())
        keyword = self._keyword_candidates(query, self._candidate_top_k())
        fused = self._rrf_fuse(semantic, keyword, top_k=self._candidate_top_k())
        return self._rerank_candidates(query, fused, top_k=top_k, retrieval_tool="hybrid_search")

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        return self.hybrid_search(query, top_k=top_k)


def build_vectorstore(config: dict[str, Any]) -> dict[str, Any]:
    vector_cfg = config.get("vectorstore", {})
    backend = vector_cfg.get("backend", "faiss")
    if backend != "faiss":
        raise ValueError("Only the local FAISS backend is implemented in v1.")

    chunks = _load_chunks(config)
    if not chunks:
        raise RuntimeError("No chunks found. Run scripts/build_chunks.py first.")

    try:
        import faiss
    except Exception as exc:
        raise RuntimeError(
            "FAISS indexing requires faiss-cpu. "
            "Install project dependencies before building the index."
        ) from exc

    vectorstore_dir = resolve_project_path(vector_cfg.get("vectorstore_dir", "storage/vectorstore"))
    docstore_dir = resolve_project_path(vector_cfg.get("docstore_dir", "storage/docstore"))
    vectorstore_dir.mkdir(parents=True, exist_ok=True)
    docstore_dir.mkdir(parents=True, exist_ok=True)

    texts = [_text_for_embedding(chunk) for chunk in chunks]
    batch_size = int(vector_cfg.get("embedding_batch_size", 32))
    vectors_list: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors_list.append(_embed_texts_with_vllm(batch))
    vectors = np.vstack(vectors_list).astype("float32")
    if bool(vector_cfg.get("normalize_embeddings", True)):
        vectors = _normalize_vectors(vectors).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    id_map = [chunk["chunk_id"] for chunk in chunks]
    faiss.write_index(index, str(vectorstore_dir / "index.faiss"))
    (vectorstore_dir / "id_map.json").write_text(json.dumps(id_map, ensure_ascii=False, indent=2), encoding="utf-8")
    write_json(docstore_dir / "chunks.json", chunks)
    with (docstore_dir / "chunks.pkl").open("wb") as file:
        pickle.dump(chunks, file)
    write_json(
        vectorstore_dir / "index_config.json",
        {
            "embedding_provider": _embedding_config()["provider"],
            "embedding_model": _embedding_config()["model"],
            "normalize_embeddings": bool(vector_cfg.get("normalize_embeddings", True)),
            "dimensions": int(vectors.shape[1]),
        },
    )

    return {
        "backend": backend,
        "chunks": len(chunks),
        "vectorstore_dir": str(vectorstore_dir),
        "docstore_dir": str(docstore_dir),
    }
