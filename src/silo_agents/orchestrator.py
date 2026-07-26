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


@dataclass(frozen=True)
class MultiOrchestrationResult:
    selected_agents: tuple[Domain, ...]
    raw_messages: tuple[AgentMessage, ...]
    policy_decisions: tuple[PolicyDecision, ...]
    abstained: bool = False

    @property
    def delivered_messages(self) -> tuple[AgentMessage, ...]:
        if not self.policy_decisions:
            return self.raw_messages
        return tuple(
            decision.sanitized_message
            for decision in self.policy_decisions
            if decision.allowed and decision.sanitized_message is not None
        )


class BlindOrchestrator:
    """Routes through relevance ACKs and never reads domain documents directly."""

    def __init__(self, agents: dict[Domain, DomainAgent], gateway: PolicyGateway) -> None:
        self.agents = agents
        self.gateway = gateway

    def route(self, query: str) -> Domain:
        selected = self.route_many(query, max_agents=1)
        if not selected:
            raise ValueError("No agent reported relevant private knowledge")
        return selected[0]

    def route_many(
        self,
        query: str,
        *,
        relative_threshold: float = 0.5,
        max_agents: int = 3,
    ) -> tuple[Domain, ...]:
        if not 0 < relative_threshold <= 1:
            raise ValueError("relative_threshold must be in (0, 1]")
        scored = sorted(
            ((agent.relevance_ack(query), domain) for domain, agent in self.agents.items()),
            key=lambda item: (-item[0], item[1].value),
        )
        if not scored or scored[0][0] == 0:
            return ()
        cutoff = scored[0][0] * relative_threshold
        return tuple(domain for score, domain in scored if score >= cutoff and score > 0)[:max_agents]

    def run(
        self, query: str, mode: ExperimentMode = ExperimentMode.POLICY_GATED
    ) -> OrchestrationResult:
        selected = self.route(query)
        message = self.agents[selected].answer(str(uuid4()), query, Domain.ORCHESTRATOR)
        decision = self.gateway.evaluate(message) if mode == ExperimentMode.POLICY_GATED else None
        return OrchestrationResult(selected, message, decision)

    def run_many(
        self, query: str, mode: ExperimentMode = ExperimentMode.POLICY_GATED
    ) -> MultiOrchestrationResult:
        selected = self.route_many(query)
        if not selected:
            return MultiOrchestrationResult((), (), (), abstained=True)
        task_id = str(uuid4())
        messages = tuple(
            self.agents[domain].answer(task_id, query, Domain.ORCHESTRATOR)
            for domain in selected
        )
        decisions = (
            tuple(self.gateway.evaluate(message) for message in messages)
            if mode == ExperimentMode.POLICY_GATED
            else ()
        )
        return MultiOrchestrationResult(selected, messages, decisions)
