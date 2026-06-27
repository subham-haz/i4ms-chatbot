"""Hybrid retrieval: dense (pgvector) + sparse (BM25) fused, then reranked.

Fusion uses min-max normalized score blending controlled by `hybrid_alpha`,
with Reciprocal Rank Fusion (RRF) available as an alternative. A cross-encoder
reranker then reorders the fused candidate set for final precision.
"""
from __future__ import annotations

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.core.schemas import RetrievedChunk
from app.observability.langfuse_client import trace_span
from app.rag.bm25_index import bm25_search
from app.rag.fusion import fuse as _fuse
from app.rag.fusion import minmax as _minmax
from app.rag.vector_store import vector_search

logger = get_logger(__name__)

try:
    from sentence_transformers import CrossEncoder

    _RERANKER: CrossEncoder | None = None

    def _get_reranker() -> CrossEncoder | None:
        global _RERANKER
        if _RERANKER is None:
            _RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return _RERANKER
except ImportError:  # pragma: no cover

    def _get_reranker():  # type: ignore
        return None


def _rerank(query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
    reranker = _get_reranker()
    if reranker is None or not chunks:
        return chunks[:top_n]
    pairs = [(query, c.content) for c in chunks]
    scores = reranker.predict(pairs)
    for chunk, score in zip(chunks, scores):
        chunk.rerank_score = float(score)
    ranked = sorted(chunks, key=lambda c: c.rerank_score or 0.0, reverse=True)
    return ranked[:top_n]


def hybrid_retrieve(query: str) -> list[RetrievedChunk]:
    settings = get_settings()
    with trace_span(
        "hybrid_retrieve",
        input={"query": query},
        metadata={"top_k": settings.retrieval_top_k, "alpha": settings.hybrid_alpha},
    ) as span:
        vector_hits = vector_search(query, settings.retrieval_top_k)
        bm25_hits = bm25_search(query, settings.retrieval_top_k)
        fused = _fuse(vector_hits, bm25_hits, settings.hybrid_alpha)
        reranked = _rerank(query, fused, settings.rerank_top_n)

        if span is not None:
            span.update(
                output={
                    "n_vector": len(vector_hits),
                    "n_bm25": len(bm25_hits),
                    "n_fused": len(fused),
                    "returned": [c.document_id for c in reranked],
                }
            )
        logger.info(
            "Retrieved v=%d b=%d fused=%d -> %d",
            len(vector_hits), len(bm25_hits), len(fused), len(reranked),
        )
        return reranked
