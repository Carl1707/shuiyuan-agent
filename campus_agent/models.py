from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    """An in-memory Shuiyuan topic or reply before chunking."""

    id: str
    title: str
    text: str
    uri: str
    content_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    """An in-memory Shuiyuan evidence chunk."""

    id: str
    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalResult:
    """A scored retrieval hit."""

    chunk: Chunk
    score: float
    highlights: list[str] = field(default_factory=list)
