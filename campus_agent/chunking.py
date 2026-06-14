from __future__ import annotations

import hashlib
import re

from campus_agent.models import Chunk, Document


def chunk_document(
    document: Document, max_chars: int = 1200, overlap_chars: int = 120
) -> list[Chunk]:
    """Split a document by paragraphs, falling back to character windows."""

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    overlap_chars = max(0, min(overlap_chars, max_chars // 2))
    paragraphs = _dedupe_segments(
        [part.strip() for part in re.split(r"\n\s*\n", document.text) if _is_useful_segment(part)]
    )
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs or [document.text.strip()]:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_window_text(paragraph, max_chars, overlap_chars))
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current.strip())
            prefix = current[-overlap_chars:].strip() if overlap_chars and current else ""
            current = f"{prefix}\n\n{paragraph}".strip() if prefix else paragraph

    if current:
        chunks.append(current.strip())

    result = []
    for position, text in enumerate(_dedupe_segments(chunks)):
        digest = hashlib.sha1(f"{document.id}:{position}:{text}".encode("utf-8")).hexdigest()[:12]
        metadata = {
            **document.metadata,
            "document_title": document.title,
            "document_uri": document.uri,
            "content_type": document.content_type,
            "chunk_position": position,
        }
        result.append(
            Chunk(
                id=f"{document.id}:chunk:{digest}",
                document_id=document.id,
                text=text,
                metadata=metadata,
            )
        )
    return result


def _window_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    windows: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        windows.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - overlap_chars, start + 1)
    return [window for window in windows if window]


def _is_useful_segment(text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) < 12:
        return False
    if cleaned.count("\ufffd") / max(len(cleaned), 1) > 0.02:
        return False
    if re.search(r"(new Vue\(|\$\.ajax|function\(|token\s*=|<div|</div>|javascript:)", cleaned, re.I):
        return False
    return True


def _dedupe_segments(segments: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for segment in segments:
        normalized = re.sub(r"\W+", "", segment).lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(segment)
    return deduped
