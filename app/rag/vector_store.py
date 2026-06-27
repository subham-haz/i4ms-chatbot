"""pgvector-backed vector store wrapper."""
from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document as LCDocument
from langchain_postgres import PGVector

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.core.schemas import Document, RetrievedChunk
from app.rag.embeddings import get_embeddings

logger = get_logger(__name__)


@lru_cache
def get_vector_store() -> PGVector:
    settings = get_settings()
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=settings.collection_name,
        connection=settings.pg_connection_string,
        use_jsonb=True,
    )


def add_documents(docs: list[Document]) -> None:
    store = get_vector_store()
    lc_docs = [
        LCDocument(page_content=d.content, metadata={**d.metadata, "doc_id": d.id})
        for d in docs
    ]
    ids = [d.id for d in docs]
    store.add_documents(lc_docs, ids=ids)
    logger.info("Indexed %d chunks into pgvector.", len(docs))


def vector_search(query: str, top_k: int) -> list[RetrievedChunk]:
    store = get_vector_store()
    results = store.similarity_search_with_relevance_scores(query, k=top_k)
    chunks: list[RetrievedChunk] = []
    for doc, score in results:
        chunks.append(
            RetrievedChunk(
                document_id=doc.metadata.get("doc_id", ""),
                content=doc.page_content,
                metadata=doc.metadata,
                vector_score=float(score),
            )
        )
    return chunks
