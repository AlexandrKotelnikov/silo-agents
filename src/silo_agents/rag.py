from __future__ import annotations

import re
from collections.abc import Iterable

from .models import Domain, RetrievalRecord


class IsolatedKnowledgeBase:
    """Small deterministic retriever used for security experiments and tests."""

    def __init__(self, domain: Domain, records: Iterable[RetrievalRecord]) -> None:
        self.domain = domain
        self._records = [record for record in records if record.domain == domain]

    def search(self, query: str, *, limit: int = 3) -> list[RetrievalRecord]:
        query_terms = _terms(query)
        scored: list[tuple[int, RetrievalRecord]] = []
        for record in self._records:
            score = len(query_terms & _terms(record.text))
            if score:
                scored.append((score, record))
        scored.sort(key=lambda item: (-item[0], item[1].record_id))
        return [record for _, record in scored[:limit]]

    def relevance_ack(self, query: str) -> float:
        matches = self.search(query, limit=1)
        if not matches:
            return 0.0
        overlap = len(_terms(query) & _terms(matches[0].text))
        return min(1.0, overlap / max(1, len(_terms(query))))


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Zа-яА-Я0-9_%-]+", text.casefold()))
