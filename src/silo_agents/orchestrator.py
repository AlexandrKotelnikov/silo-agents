from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import uuid4

from .agents import DomainAgent
from .models import AgentId, AgentMessage, PolicyDecision
from .policy import PolicyGateway
from .routing import split_query_clauses


class ExperimentMode(StrEnum):
    SHARED_RAG = "shared_rag"
    ISOLATED_RAG = "isolated_rag"
    POLICY_GATED = "policy_gated"


@dataclass(frozen=True)
class OrchestrationResult:
    selected_agent: AgentId
    raw_message: AgentMessage
    policy_decision: PolicyDecision | None


@dataclass(frozen=True)
class MultiOrchestrationResult:
    selected_agents: tuple[AgentId, ...]
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
    """Routes through relevance ACKs and never reads agent documents directly."""

    def __init__(
        self,
        agents: dict[AgentId, DomainAgent],
        gateway: PolicyGateway,
        *,
        orchestrator_id: AgentId = AgentId.ORCHESTRATOR,
        max_agents: int = 8,
        relative_threshold: float = 0.5,
    ) -> None:
        if not agents:
            raise ValueError("At least one agent is required")
        if max_agents < 1:
            raise ValueError("max_agents must be at least 1")
        if not 0 < relative_threshold <= 1:
            raise ValueError("relative_threshold must be in (0, 1]")
        self.agents = agents
        self.gateway = gateway
        self.orchestrator_id = orchestrator_id
        self.max_agents = max_agents
        self.relative_threshold = relative_threshold

    def route(self, query: str) -> AgentId:
        selected = self.route_many(query, max_agents=1)
        if not selected:
            raise ValueError("No agent reported relevant private knowledge")
        return selected[0]

    def route_many(
        self,
        query: str,
        *,
        relative_threshold: float | None = None,
        max_agents: int | None = None,
    ) -> tuple[AgentId, ...]:
        threshold = self.relative_threshold if relative_threshold is None else relative_threshold
        limit = self.max_agents if max_agents is None else max_agents
        if not 0 < threshold <= 1:
            raise ValueError("relative_threshold must be in (0, 1]")
        if limit < 1:
            raise ValueError("max_agents must be at least 1")

        selected_scores: dict[AgentId, float] = {}
        for clause in split_query_clauses(query):
            scored = sorted(
                ((agent.relevance_ack(clause), agent_id) for agent_id, agent in self.agents.items()),
                key=lambda item: (-item[0], item[1].value),
            )
            if not scored or scored[0][0] == 0:
                continue
            cutoff = scored[0][0] * threshold
            for score, agent_id in scored:
                if score >= cutoff and score > 0:
                    selected_scores[agent_id] = max(selected_scores.get(agent_id, 0.0), score)

        return tuple(
            agent_id
            for agent_id, _ in sorted(
                selected_scores.items(), key=lambda item: (-item[1], item[0].value)
            )[:limit]
        )

    def run(
        self, query: str, mode: ExperimentMode = ExperimentMode.POLICY_GATED
    ) -> OrchestrationResult:
        selected = self.route(query)
        message = self.agents[selected].answer(str(uuid4()), query, self.orchestrator_id)
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
            self.agents[agent_id].answer(task_id, query, self.orchestrator_id)
            for agent_id in selected
        )
        decisions = (
            tuple(self.gateway.evaluate(message) for message in messages)
            if mode == ExperimentMode.POLICY_GATED
            else ()
        )
        return MultiOrchestrationResult(selected, messages, decisions)
