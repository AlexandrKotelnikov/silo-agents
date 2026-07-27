from __future__ import annotations

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
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport
        )

    def close(self) -> None:
        self._client.close()

    def health(self) -> bool:
        response = self._client.get("/collections")
        response.raise_for_status()
        return True

    def ensure_collection(self, collection_name: str, vector_size: int) -> None:
        response = self._client.get(f"/collections/{collection_name}")
        if response.status_code == 200:
            existing = self._extract_vector_size(response.json())
            if existing != vector_size:
                raise ValueError(
                    f"Collection {collection_name!r} uses {existing} dimensions; "
                    f"the embedder returned {vector_size}. Delete/recreate the collection."
                )
            return
        if response.status_code != 404:
            response.raise_for_status()
        created = self._client.put(
            f"/collections/{collection_name}",
            json={"vectors": {"size": vector_size, "distance": "Cosine"}},
        )
        created.raise_for_status()

    @staticmethod
    def _extract_vector_size(raw: Any) -> int:
        if not isinstance(raw, dict):
            raise ValueError("Qdrant collection response must be an object")
        result = raw.get("result")
        if not isinstance(result, dict):
            raise ValueError("Qdrant collection response omitted result")
        config = result.get("config")
        if not isinstance(config, dict):
            raise ValueError("Qdrant collection response omitted config")
        params = config.get("params")
        if not isinstance(params, dict):
            raise ValueError("Qdrant collection response omitted params")
        vectors = params.get("vectors")
        if not isinstance(vectors, dict) or not isinstance(vectors.get("size"), int):
            raise ValueError("Only a single unnamed Qdrant vector is supported")
        return int(vectors["size"])

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
        points = self._query_points(query, limit=limit)
        return [self._validate_point(point) for point in points]

    def relevance_ack(self, query: str) -> float:
        points = self._query_points(query, limit=1)
        if not points:
            return 0.0
        self._validate_point(points[0])
        raw_score = points[0].get("score")
        if not isinstance(raw_score, int | float):
            raise ValueError("Qdrant result omitted the semantic relevance score")
        return max(0.0, min(1.0, float(raw_score)))

    def _query_points(self, query: str, *, limit: int) -> list[dict[str, Any]]:
        return self.client.query(
            self.collection_name,
            vector=self.embedder.embed(query),
            query_filter=self._authorization_filter(),
            limit=limit,
        )

    def _validate_point(self, point: dict[str, Any]) -> RetrievalRecord:
        payload = point.get("payload")
        if not isinstance(payload, dict):
            raise PermissionError("Qdrant result omitted the authorization payload")
        record = RetrievalRecord.model_validate(payload)
        if not self.principal.allows(record.domain, record.classification, record.record_id):
            raise PermissionError("Qdrant returned a record outside the principal scope")
        if record.domain != self.domain:
            raise PermissionError("Qdrant returned a cross-domain record")
        return record

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
