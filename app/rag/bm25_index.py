"""BM25 sparse retrieval.

A lightweight in-memory BM25 index. In production this would be backed by
Postgres full-text search (tsvector) or an Elasticsearch/OpenSearch index, but
the interface stays identical so the hybrid layer is agnostic to the backend.
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from app.core.schemas import Document, RetrievedChunk

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self) -> None:
        self._docs: list[Document] = []
        self._bm25: BM25Okapi | None = None

    def build(self, docs: list[Document]) -> None:
        self._docs = docs
        corpus = [_tokenize(d.content) for d in docs]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self._bm25 is None or not self._docs:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(
            zip(self._docs, scores), key=lambda x: x[1], reverse=True
        )[:top_k]
        return [
            RetrievedChunk(
                document_id=doc.id,
                content=doc.content,
                metadata=doc.metadata,
                bm25_score=float(score),
            )
            for doc, score in ranked
        ]


# Module-level singleton; populated at ingestion time.
_index = BM25Index()


def build_bm25(docs: list[Document]) -> None:
    _index.build(docs)


def bm25_search(query: str, top_k: int) -> list[RetrievedChunk]:
    return _index.search(query, top_k)
