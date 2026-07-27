import os
import uuid

import pytest

from silo_agents.embeddings import HashingEmbedder
from silo_agents.models import Domain, RetrievalRecord
from silo_agents.qdrant import QdrantRestClient, QdrantRetriever
from silo_agents.security import RetrievalPrincipal


@pytest.mark.integration
def test_real_qdrant_round_trip() -> None:
    url = os.getenv("QDRANT_TEST_URL")
    if not url:
        pytest.skip("QDRANT_TEST_URL is not configured")
    collection = f"silo_ci_{uuid.uuid4().hex}"
    client = QdrantRestClient(url)
    embedder = HashingEmbedder(32)
    record = RetrievalRecord(
        record_id="PROC-CI", domain=Domain.PROCESS, text="reactor cooling capacity"
    )
    try:
        client.ensure_collection(collection, embedder.dimensions)
        client.upsert_records(collection, [record], embedder)
        retriever = QdrantRetriever(
            client,
            collection,
            Domain.PROCESS,
            RetrievalPrincipal(principal_id="ci", allowed_domains={Domain.PROCESS}),
            embedder,
        )
        assert retriever.search("reactor cooling")[0].record_id == "PROC-CI"
    finally:
        client.close()
