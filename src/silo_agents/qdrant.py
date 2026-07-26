from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, cast
from uuid import NAMESPACE_URL, uuid5

import httpx

from .embeddings import Embedder
from .models import Domain, RetrievalRecord
from .security import RetrievalPrincipal


class QdrantRestClient:
    """Minimal Qdrant REST adapter with explicit request bodies for auditability."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        headers = {"api-key": api_key} if api_key else {}
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def ensure_collection(self, collection_name: str, vector_size: int) -> None:
        response = self._client.get(f"/collections/{collection_name}")
        if response.status_code == 200:
            return
        if response.status_code != 404:
            response.raise_for_status()
        created = self._client.put(
            f"/collections/{collection_name}",
            json={"vectors": {"size": vector_size, "distance": "Cosine"}},
        )
        created.raise_for_status()

    def upsert_records(
        self,
        collection_name: str,
        records: Iterable[RetrievalRecord],
        embedder: Embedder,
    ) -> None:
        points: list[dict[str, Any]] = [
            {
                "id": str(uuid5(NAMESPACE_URL, record.record_id)),
                "vector": embedder.embed(record.text),
                "payload": record.model_dump(mode="json"),
            }
            for record in records
        ]
        response = self._client.put(
            f"/collections/{collection_name}/points",
            params={"wait": "true"},
            json={"points": points},
        )
        response.raise_for_status()

    def query(
        self,
        collection_name: str,
        *,
        vector: list[float],
        query_filter: dict[str, Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        response = self._client.post(
            f"/collections/{collection_name}/points/query",
            json={
                "query": vector,
                "filter": query_filter,
                "limit": limit,
                "with_payload": True,
            },
        )
        response.raise_for_status()
        raw_payload: Any = response.json()
        if not isinstance(raw_payload, dict):
            raise ValueError("Qdrant returned a non-object response")
        payload = cast(dict[str, Any], raw_payload)
        result = payload.get("result", {})
        if not isinstance(result, dict):
            raise ValueError("Qdrant returned an invalid result object")
        points = result.get("points", [])
        if not isinstance(points, list):
            raise ValueError("Qdrant returned an invalid points response")
        return [cast(dict[str, Any], point) for point in points if isinstance(point, dict)]


class QdrantRetriever:
    """Retriever that binds one agent identity to one authorized domain."""

    def __init__(
        self,
        client: QdrantRestClient,
        collection_name: str,
        domain: Domain,
        principal: RetrievalPrincipal,
        embedder: Embedder,
    ) -> None:
        principal.assert_domain(domain)
        self.client = client
        self.collection_name = collection_name
        self.domain = domain
        self.principal = principal
        self.embedder = embedder

    def search(self, query: str, *, limit: int = 3) -> list[RetrievalRecord]:
        points = self.client.query(
            self.collection_name,
            vector=self.embedder.embed(query),
            query_filter=self._authorization_filter(),
            limit=limit,
        )
        records: list[RetrievalRecord] = []
        for point in points:
            payload = point.get("payload")
            if not isinstance(payload, dict):
                raise PermissionError("Qdrant result omitted the authorization payload")
            record = RetrievalRecord.model_validate(payload)
            if not self.principal.allows(record.domain, record.classification, record.record_id):
                raise PermissionError("Qdrant returned a record outside the principal scope")
            if record.domain != self.domain:
                raise PermissionError("Qdrant returned a cross-domain record")
            records.append(record)
        return records

    def relevance_ack(self, query: str) -> float:
        records = self.search(query, limit=1)
        if not records:
            return 0.0
        query_terms = _terms(query)
        overlap = len(query_terms & _terms(records[0].text))
        return min(1.0, overlap / max(1, len(query_terms)))

    def _authorization_filter(self) -> dict[str, Any]:
        conditions: list[dict[str, Any]] = [
            {"key": "domain", "match": {"value": self.domain.value}},
            {
                "key": "classification",
                "match": {
                    "any": [value.value for value in self.principal.allowed_classifications()]
                },
            },
        ]
        if self.principal.allowed_record_ids is not None:
            conditions.append(
                {
                    "key": "record_id",
                    "match": {"any": sorted(self.principal.allowed_record_ids)},
                }
            )
        return {"must": conditions}


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Zа-яА-Я0-9_%-]+", text.casefold()))
