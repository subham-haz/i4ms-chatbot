"""Pure score-fusion logic for hybrid retrieval (no heavy deps)."""
from __future__ import annotations

from app.core.schemas import RetrievedChunk


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def fuse(
    vector_hits: list[RetrievedChunk],
    bm25_hits: list[RetrievedChunk],
    alpha: float,
) -> list[RetrievedChunk]:
    """Blend min-max normalized vector and BM25 scores into one ranking."""
    pool: dict[str, RetrievedChunk] = {}

    for hit, norm in zip(vector_hits, minmax([h.vector_score for h in vector_hits])):
        hit.vector_score = norm
        pool[hit.document_id] = hit

    for hit, norm in zip(bm25_hits, minmax([h.bm25_score for h in bm25_hits])):
        if hit.document_id in pool:
            pool[hit.document_id].bm25_score = norm
        else:
            hit.bm25_score = norm
            pool[hit.document_id] = hit

    for chunk in pool.values():
        chunk.hybrid_score = alpha * chunk.vector_score + (1 - alpha) * chunk.bm25_score

    return sorted(pool.values(), key=lambda c: c.hybrid_score, reverse=True)
