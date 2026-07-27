from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from .agents import DomainAgent, LLMDomainAgent
from .embeddings import Embedder
from .llm import GroundedLLM
from .models import AgentId, Classification
from .orchestrator import BlindOrchestrator
from .policy import PolicyGateway
from .qdrant import QdrantRestClient, QdrantRetriever
from .security import RetrievalPrincipal


class RoutingSpec(BaseModel):
    description: str = ""
    terms: set[str] = Field(default_factory=set)
    aliases: dict[str, str] = Field(default_factory=dict)


class AgentSpec(BaseModel):
    id: AgentId
    name: str
    description: str = ""
    knowledge_namespace: str | None = None
    max_classification: Classification = Classification.INTERNAL
    routing: RoutingSpec = Field(default_factory=RoutingSpec)

    @model_validator(mode="after")
    def default_namespace(self) -> AgentSpec:
        if self.id == AgentId.ORCHESTRATOR:
            raise ValueError("orchestrator is reserved and cannot be declared as an agent")
        if self.knowledge_namespace is None:
            self.knowledge_namespace = self.id.value
        # Reuse AgentId validation for namespace names.
        AgentId(self.knowledge_namespace)
        return self

    @property
    def namespace_id(self) -> AgentId:
        if self.knowledge_namespace is None:
            raise ValueError("knowledge_namespace was not initialized")
        return AgentId(self.knowledge_namespace)


class OrchestratorSpec(BaseModel):
    id: AgentId = AgentId.ORCHESTRATOR
    max_agents_per_query: int = Field(default=8, ge=1, le=1000)
    relative_threshold: float = Field(default=0.5, gt=0, le=1)


class ProjectPathsSpec(BaseModel):
    corpus: str = "corpus/records.jsonl"
    cases: str = "benchmarks/tasks.jsonl"
    reports: str = "reports"


class PolicySpec(BaseModel):
    default: str = "deny"
    routes: dict[AgentId, set[AgentId]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_fail_closed_default(self) -> PolicySpec:
        if self.default != "deny":
            raise ValueError("Only fail-closed policy.default='deny' is supported")
        return self


class ProjectSpec(BaseModel):
    schema_version: int = 1
    name: str
    orchestrator: OrchestratorSpec = Field(default_factory=OrchestratorSpec)
    paths: ProjectPathsSpec = Field(default_factory=ProjectPathsSpec)
    agents: list[AgentSpec]
    policy: PolicySpec = Field(default_factory=PolicySpec)

    @model_validator(mode="after")
    def validate_registry(self) -> ProjectSpec:
        ids = [agent.id for agent in self.agents]
        if not ids:
            raise ValueError("A project must define at least one agent")
        if len(ids) != len(set(ids)):
            raise ValueError("Agent IDs must be unique")
        namespaces = [agent.namespace_id for agent in self.agents]
        if len(namespaces) != len(set(namespaces)):
            raise ValueError("Knowledge namespaces must be unique")
        known = set(ids) | {self.orchestrator.id}
        for sender, recipients in self.policy.routes.items():
            if sender not in known:
                raise ValueError(f"Unknown policy sender: {sender.value}")
            unknown = recipients - known
            if unknown:
                names = ", ".join(sorted(value.value for value in unknown))
                raise ValueError(f"Unknown policy recipients: {names}")
        return self

    @classmethod
    def load(cls, path: str | Path) -> ProjectSpec:
        location = Path(path)
        raw_text = location.read_text(encoding="utf-8")
        if location.suffix.casefold() == ".json":
            raw: Any = json.loads(raw_text)
        else:
            raw = yaml.safe_load(raw_text)
        if not isinstance(raw, dict):
            raise ValueError("Project configuration must be an object")
        return cls.model_validate(raw)

    def write(self, path: str | Path) -> Path:
        location = Path(path)
        location.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json", exclude_none=True)
        if location.suffix.casefold() == ".json":
            location.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            location.write_text(
                yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
            )
        return location

    def allowed_routes(self) -> dict[AgentId, set[AgentId]]:
        if self.policy.routes:
            return {sender: set(recipients) for sender, recipients in self.policy.routes.items()}
        return {agent.id: {self.orchestrator.id} for agent in self.agents}

    def namespace_to_agent(self) -> dict[AgentId, AgentId]:
        return {agent.namespace_id: agent.id for agent in self.agents}


class AgentRegistry:
    """Build an arbitrary number of isolated agents from a validated project file."""

    def __init__(self, project: ProjectSpec) -> None:
        self.project = project

    @classmethod
    def from_file(cls, path: str | Path) -> AgentRegistry:
        return cls(ProjectSpec.load(path))

    @property
    def ids(self) -> tuple[AgentId, ...]:
        return tuple(agent.id for agent in self.project.agents)

    def build_qdrant_system(
        self,
        client: QdrantRestClient,
        collection_name: str,
        embedder: Embedder,
        llm: GroundedLLM,
        *,
        safe_context: bool = True,
    ) -> BlindOrchestrator:
        agents: dict[AgentId, DomainAgent] = {}
        for spec in self.project.agents:
            namespace = spec.namespace_id
            principal = RetrievalPrincipal(
                principal_id=f"{spec.id.value}-agent",
                allowed_domains={namespace},
                max_classification=spec.max_classification,
            )
            retriever = QdrantRetriever(
                client,
                collection_name,
                namespace,
                principal,
                embedder,
                routing_terms=spec.routing.terms,
                routing_aliases=spec.routing.aliases,
            )
            agents[spec.id] = LLMDomainAgent(
                spec.id,
                retriever,
                llm,
                safe_context=safe_context,
            )
        return BlindOrchestrator(
            agents,
            PolicyGateway(self.project.allowed_routes()),
            orchestrator_id=self.project.orchestrator.id,
            max_agents=self.project.orchestrator.max_agents_per_query,
            relative_threshold=self.project.orchestrator.relative_threshold,
        )
