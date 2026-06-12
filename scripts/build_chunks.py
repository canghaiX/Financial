"""Build RAG chunks from MinerU outputs."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from financial_agentic_rag.config import load_yaml, resolve_project_path
from financial_agentic_rag.indexing.mineru_chunks import build_all_chunks
from financial_agentic_rag.utils.io import write_jsonl


def main() -> None:
    config = load_yaml("configs/retrieval_config.yaml")
    chunks, rejected, stats = build_all_chunks(config)
    output_paths = config.get("output_paths", {})
    chunks_path = resolve_project_path(output_paths.get("chunks_jsonl", "data/processed/chunks.jsonl"))
    rejected_path = resolve_project_path(
        output_paths.get("rejected_chunks_jsonl", "data/processed/rejected_chunks.jsonl")
    )
    write_jsonl(chunks_path, (chunk.model_dump() for chunk in chunks))
    write_jsonl(rejected_path, rejected)
    print(
        f"documents={stats.documents} chunks={stats.chunks} rejected={stats.rejected} "
        f"chunks_path={chunks_path}"
    )


if __name__ == "__main__":
    main()
