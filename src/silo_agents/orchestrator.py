from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import uuid4

from .agents import DomainAgent
from .models import AgentMessage, Domain, PolicyDecision
from .policy import PolicyGateway


class ExperimentMode(StrEnum):
    SHARED_RAG = "shared_rag"
    ISOLATED_RAG = "isolated_rag"
    POLICY_GATED = "policy_gated"


@dataclass(frozen=True)
class OrchestrationResult:
    selected_agent: Domain
    raw_message: AgentMessage
    policy_decision: PolicyDecision | None


class BlindOrchestrator:
    """Routes through relevance ACKs and never reads domain documents directly."""

    def __init__(self, agents: dict[Domain, DomainAgent], gateway: PolicyGateway) -> None:
        self.agents = agents
        self.gateway = gateway

    def route(self, query: str) -> Domain:
        scored = [(agent.relevance_ack(query), domain) for domain, agent in self.agents.items()]
        score, domain = max(scored, key=lambda item: (item[0], item[1].value))
        if score == 0:
            raise ValueError("No agent reported relevant private knowledge")
        return domain

    def run(self, query: str, mode: ExperimentMode = ExperimentMode.POLICY_GATED) -> OrchestrationResult:
        selected = self.route(query)
        message = self.agents[selected].answer(str(uuid4()), query, Domain.ORCHESTRATOR)
        decision = self.gateway.evaluate(message) if mode == ExperimentMode.POLICY_GATED else None
        return OrchestrationResult(selected, message, decision)
