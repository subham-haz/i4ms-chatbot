"""Document loading and chunking."""
from __future__ import annotations

import uuid
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.core.schemas import Document


def _splitter() -> RecursiveCharacterTextSplitter:
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_text(text: str, source: str) -> list[Document]:
    chunks = _splitter().split_text(text)
    return [
        Document(
            id=str(uuid.uuid4()),
            content=chunk,
            metadata={"source": source, "chunk_index": i},
        )
        for i, chunk in enumerate(chunks)
    ]


def load_directory(path: str | Path) -> list[Document]:
    """Load .txt/.md files from a directory and chunk them."""
    base = Path(path)
    docs: list[Document] = []
    for file in sorted(base.glob("**/*")):
        if file.suffix.lower() in {".txt", ".md"}:
            text = file.read_text(encoding="utf-8")
            docs.extend(chunk_text(text, source=str(file.name)))
    return docs
