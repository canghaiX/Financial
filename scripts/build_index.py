"""Build the local FAISS index from cleaned chunks."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from financial_agentic_rag.config import load_yaml
from financial_agentic_rag.retrievers.vectorstore import build_vectorstore


def main() -> None:
    config = load_yaml("configs/retrieval_config.yaml")
    stats = build_vectorstore(config)
    print(
        f"backend={stats['backend']} chunks={stats['chunks']} "
        f"vectorstore={stats['vectorstore_dir']} docstore={stats['docstore_dir']}"
    )


if __name__ == "__main__":
    main()
