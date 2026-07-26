from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Protocol

from .models import Domain, RetrievalRecord


class Retriever(Protocol):
    def search(self, query: str, *, limit: int = 3) -> list[RetrievalRecord]: ...

    def relevance_ack(self, query: str) -> float: ...


class SharedKnowledgeBase:
    """Baseline retriever that intentionally searches every domain."""

    def __init__(self, records: Iterable[RetrievalRecord]) -> None:
        self._records = list(records)

    def search(self, query: str, *, limit: int = 3) -> list[RetrievalRecord]:
        return _search_records(self._records, query, limit=limit)

    def relevance_ack(self, query: str) -> float:
        return _relevance_ack(self, query)


class IsolatedKnowledgeBase:
    """Deterministic retriever that indexes exactly one domain."""

    def __init__(self, domain: Domain, records: Iterable[RetrievalRecord]) -> None:
        self.domain = domain
        self._records = [record for record in records if record.domain == domain]

    def search(self, query: str, *, limit: int = 3) -> list[RetrievalRecord]:
        return _search_records(self._records, query, limit=limit)

    def relevance_ack(self, query: str) -> float:
        return _relevance_ack(self, query)


def _search_records(
    records: Iterable[RetrievalRecord], query: str, *, limit: int
) -> list[RetrievalRecord]:
    query_terms = _terms(query)
    scored: list[tuple[int, RetrievalRecord]] = []
    for record in records:
        score = len(query_terms & _terms(record.text))
        if score:
            scored.append((score, record))
    scored.sort(key=lambda item: (-item[0], item[1].record_id))
    return [record for _, record in scored[:limit]]


def _relevance_ack(retriever: Retriever, query: str) -> float:
    matches = retriever.search(query, limit=1)
    if not matches:
        return 0.0
    overlap = len(_terms(query) & _terms(matches[0].text))
    return min(1.0, overlap / max(1, len(_terms(query))))


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Zа-яА-Я0-9_%-]+", text.casefold()))
