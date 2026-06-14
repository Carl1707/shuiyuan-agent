from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

from campus_agent.models import Chunk, RetrievalResult


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")


class EmbeddingModel(Protocol):
    """Small interface for pluggable text embedding providers."""

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError


class HashingEmbeddingModel:
    """Dependency-free embedding baseline using signed feature hashing."""

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions
        self._cache: dict[str, list[float]] = {}

    def embed(self, text: str) -> list[float]:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        vector = [0.0] * self.dimensions
        terms = tokenize(text)
        for term, count in Counter(terms).items():
            digest = _stable_hash(term)
            bucket = digest % self.dimensions
            sign = -1.0 if (digest >> 8) & 1 else 1.0
            vector[bucket] += sign * (1.0 + math.log(count))
        normalized = _normalize_vector(vector)
        self._cache[text] = normalized
        return normalized


DEFAULT_HASHING_EMBEDDING_MODEL = HashingEmbeddingModel()


def retrieve(
    chunks: list[Chunk],
    query: str,
    top_k: int = 5,
    *,
    mode: str = "keyword",
    keyword_weight: float = 0.65,
    vector_weight: float = 0.35,
    embedding_model: EmbeddingModel | None = None,
    metadata_filters: dict[str, object] | None = None,
) -> list[RetrievalResult]:
    """Return ranked matches from an in-memory chunk list."""
    candidate_chunks = filter_chunks(chunks, metadata_filters)

    if mode == "keyword":
        return keyword_retrieve(candidate_chunks, query, top_k)
    if mode == "vector":
        return vector_retrieve(candidate_chunks, query, top_k, embedding_model=embedding_model)
    if mode == "hybrid":
        return hybrid_retrieve(
            candidate_chunks,
            query,
            top_k,
            keyword_weight=keyword_weight,
            vector_weight=vector_weight,
            embedding_model=embedding_model,
        )
    raise ValueError(f"unknown retrieval mode: {mode}")


def filter_chunks(
    chunks: list[Chunk],
    metadata_filters: dict[str, object] | None = None,
) -> list[Chunk]:
    if not metadata_filters:
        return chunks
    return [chunk for chunk in chunks if _matches_metadata_filters(chunk, metadata_filters)]


def keyword_retrieve(
    chunks: list[Chunk], query: str, top_k: int = 5
) -> list[RetrievalResult]:
    """Return BM25-style keyword matches from an in-memory chunk list."""

    query_terms = tokenize(query)
    if not query_terms or not chunks:
        return []

    tokenized = [tokenize(chunk.text) for chunk in chunks]
    doc_freq = Counter(term for terms in tokenized for term in set(terms))
    avg_len = sum(len(terms) for terms in tokenized) / max(len(tokenized), 1)
    query_counts = Counter(query_terms)

    scored: list[RetrievalResult] = []
    for chunk, terms in zip(chunks, tokenized, strict=True):
        term_counts = Counter(terms)
        score = 0.0
        for term, query_weight in query_counts.items():
            freq = term_counts.get(term, 0)
            if freq == 0:
                continue
            idf = math.log(1 + (len(chunks) - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denom = freq + 1.2 * (1 - 0.75 + 0.75 * len(terms) / max(avg_len, 1))
            score += query_weight * idf * ((freq * 2.2) / denom)
        if score > 0:
            scored.append(RetrievalResult(chunk=chunk, score=score, highlights=_highlights(chunk.text, query_terms)))

    scored.sort(key=lambda result: result.score, reverse=True)
    return scored[:top_k]


def vector_retrieve(
    chunks: list[Chunk],
    query: str,
    top_k: int = 5,
    *,
    embedding_model: EmbeddingModel | None = None,
) -> list[RetrievalResult]:
    """Return semantic-ish matches using a pluggable embedding interface."""

    if not query.strip() or not chunks:
        return []

    model = embedding_model or DEFAULT_HASHING_EMBEDDING_MODEL
    query_vector = model.embed(query)
    scored: list[RetrievalResult] = []
    for chunk in chunks:
        score = _dot(query_vector, model.embed(chunk.text))
        if score > 0:
            scored.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    highlights=_highlights(chunk.text, tokenize(query)),
                )
            )

    scored.sort(key=lambda result: result.score, reverse=True)
    return scored[:top_k]


def hybrid_retrieve(
    chunks: list[Chunk],
    query: str,
    top_k: int = 5,
    *,
    keyword_weight: float = 0.65,
    vector_weight: float = 0.35,
    embedding_model: EmbeddingModel | None = None,
) -> list[RetrievalResult]:
    """Combine normalized BM25 and vector scores."""

    if keyword_weight < 0 or vector_weight < 0:
        raise ValueError("retrieval weights must be non-negative")
    if keyword_weight == 0 and vector_weight == 0:
        raise ValueError("at least one retrieval weight must be positive")

    keyword_results = keyword_retrieve(chunks, query, top_k=len(chunks))
    vector_results = vector_retrieve(
        chunks, query, top_k=len(chunks), embedding_model=embedding_model
    )
    keyword_scores = {result.chunk.id: result.score for result in keyword_results}
    vector_scores = {result.chunk.id: result.score for result in vector_results}
    keyword_max = max(keyword_scores.values(), default=0.0)
    vector_max = max(vector_scores.values(), default=0.0)

    combined_ids = set(keyword_scores) | set(vector_scores)
    scored: list[RetrievalResult] = []
    for chunk in chunks:
        chunk_id = chunk.id
        if chunk_id not in combined_ids:
            continue
        keyword_score = _normalize_score(keyword_scores.get(chunk_id, 0.0), keyword_max)
        vector_score = _normalize_score(vector_scores.get(chunk_id, 0.0), vector_max)
        score = keyword_weight * keyword_score + vector_weight * vector_score
        if score <= 0:
            continue
        scored.append(
            RetrievalResult(
                chunk=chunk,
                score=score,
                highlights=_highlights(chunk.text, tokenize(query)),
            )
        )

    scored.sort(key=lambda result: result.score, reverse=True)
    return scored[:top_k]


def tokenize(text: str) -> list[str]:
    terms: list[str] = []
    for match in TOKEN_PATTERN.finditer(text):
        token = match.group(0).lower()
        if _is_cjk(token):
            terms.extend(_cjk_terms(token))
        else:
            terms.append(token)
    return terms


def _is_cjk(token: str) -> bool:
    return all("\u4e00" <= char <= "\u9fff" for char in token)


def _cjk_terms(token: str) -> list[str]:
    if len(token) <= 1:
        return []
    terms: list[str] = []
    if len(token) <= 8:
        terms.append(token)
    for ngram_size in (2, 3, 4):
        if len(token) < ngram_size:
            continue
        terms.extend(token[index : index + ngram_size] for index in range(len(token) - ngram_size + 1))
    return terms


def _stable_hash(text: str) -> int:
    import hashlib

    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def _normalize_vector(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return vector
    return [value / magnitude for value in vector]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _normalize_score(score: float, max_score: float) -> float:
    if max_score <= 0:
        return 0.0
    return score / max_score


def _matches_metadata_filters(chunk: Chunk, metadata_filters: dict[str, object]) -> bool:
    metadata = chunk.metadata
    for key, expected in metadata_filters.items():
        if key.endswith("_gte"):
            field = key[:-4]
            actual = str(metadata.get(field, "") or "")
            if not actual or actual < str(expected):
                return False
            continue
        if key.endswith("_lte"):
            field = key[:-4]
            actual = str(metadata.get(field, "") or "")
            if not actual or actual > str(expected):
                return False
            continue
        actual = metadata.get(key)
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        else:
            if actual != expected:
                return False
    return True


def _highlights(text: str, terms: list[str], limit: int = 3) -> list[str]:
    lowered = text.lower()
    snippets: list[str] = []
    seen: set[str] = set()
    for term in terms:
        pos = lowered.find(term.lower())
        if pos < 0:
            continue
        start = max(0, pos - 40)
        end = min(len(text), pos + len(term) + 80)
        snippet = re.sub(r"\s+", " ", text[start:end]).strip()
        if snippet and snippet not in seen:
            snippets.append(snippet)
            seen.add(snippet)
        if len(snippets) >= limit:
            break
    return snippets
