from __future__ import annotations

from .agents import DomainAgent, LLMDomainAgent
from .embeddings import Embedder
from .llm import GroundedLLM
from .models import AgentId, Classification, RetrievalRecord
from .orchestrator import BlindOrchestrator
from .policy import PolicyGateway
from .project import AgentRegistry, ProjectSpec
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
    safe_context: bool = True,
    project: ProjectSpec | None = None,
) -> BlindOrchestrator:
    if project is not None:
        return AgentRegistry(project).build_qdrant_system(
            client,
            collection_name,
            embedder,
            llm,
            safe_context=safe_context,
        )

    domains = (AgentId.PROCESS, AgentId.MAINTENANCE, AgentId.ECONOMICS)
    agents: dict[AgentId, DomainAgent] = {}
    for domain in domains:
        principal = RetrievalPrincipal(
            principal_id=f"{domain.value}-agent",
            allowed_domains={domain},
            max_classification=max_classification,
        )
        retriever = QdrantRetriever(client, collection_name, domain, principal, embedder)
        agents[domain] = LLMDomainAgent(
            domain,
            retriever,
            llm,
            safe_context=safe_context,
        )
    routes = {domain: {AgentId.ORCHESTRATOR} for domain in domains}
    return BlindOrchestrator(agents, PolicyGateway(routes))
