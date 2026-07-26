from __future__ import annotations

from .agents import LLMDomainAgent
from .embeddings import Embedder
from .llm import GroundedLLM
from .models import Classification, Domain, RetrievalRecord
from .orchestrator import BlindOrchestrator
from .policy import PolicyGateway
from .qdrant import QdrantRestClient, QdrantRetriever
from .security import RetrievalPrincipal


def ingest_qdrant(
    client: QdrantRestClient,
    collection_name: str,
    records: list[RetrievalRecord],
    embedder: Embedder,
) -> None:
    client.ensure_collection(collection_name, embedder.dimensions)
    client.upsert_records(collection_name, records, embedder)


def build_qdrant_llm_system(
    client: QdrantRestClient,
    collection_name: str,
    embedder: Embedder,
    llm: GroundedLLM,
    *,
    max_classification: Classification = Classification.INTERNAL,
) -> BlindOrchestrator:
    domains = (Domain.PROCESS, Domain.MAINTENANCE, Domain.ECONOMICS)
    agents = {}
    for domain in domains:
        principal = RetrievalPrincipal(
            principal_id=f"{domain.value}-agent",
            allowed_domains={domain},
            max_classification=max_classification,
        )
        retriever = QdrantRetriever(client, collection_name, domain, principal, embedder)
        agents[domain] = LLMDomainAgent(domain, retriever, llm)
    routes = {domain: {Domain.ORCHESTRATOR} for domain in domains}
    return BlindOrchestrator(agents, PolicyGateway(routes))
