"""Ingest documents: chunk -> index into pgvector -> build BM25 index."""
from __future__ import annotations

import argparse

from app.core.logging_config import configure_logging, get_logger
from app.rag.bm25_index import build_bm25
from app.rag.chunking import load_directory
from app.rag.vector_store import add_documents

logger = get_logger(__name__)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG store.")
    parser.add_argument("--path", default="data/policies", help="Directory of .txt/.md files")
    args = parser.parse_args()

    docs = load_directory(args.path)
    if not docs:
        logger.warning("No documents found in %s", args.path)
        return

    add_documents(docs)   # dense index (pgvector)
    build_bm25(docs)      # sparse index (BM25, in-memory)
    logger.info("Ingestion complete: %d chunks indexed.", len(docs))


if __name__ == "__main__":
    main()
