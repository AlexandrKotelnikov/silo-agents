from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

from .llm import GroundedLLM
from .models import AgentMessage, Domain, Evidence
from .rag import Retriever
from .routing import safe_llm_record, safe_shareable


class DomainAgent:
    def __init__(self, domain: Domain, knowledge_base: Retriever) -> None:
        self.domain = domain
        self.knowledge_base = knowledge_base

    def relevance_ack(self, query: str) -> float:
        return self.knowledge_base.relevance_ack(query)

    def answer(self, task_id: str, query: str, recipient: Domain) -> AgentMessage:
        records = self.knowledge_base.search(query)
        if not records:
            return AgentMessage(
                message_id=str(uuid4()),
                task_id=task_id,
                sender=self.domain,
                recipient=recipient,
                purpose="domain_response",
                share_scope={recipient},
                conclusion={"status": "insufficient_domain_evidence"},
                missing_information=["relevant domain evidence"],
            )
        record = records[0]
        shareable = cast(dict[str, Any], record.metadata.get("shareable", {}))
        restricted_fields = {
            str(value) for value in cast(list[object], record.metadata.get("restricted_fields", []))
        }
        sensitive_values = {
            str(shareable[field_name])
            for field_name in restricted_fields
            if field_name in shareable
        }
        return AgentMessage(
            message_id=str(uuid4()),
            task_id=task_id,
            sender=self.domain,
            recipient=recipient,
            purpose="domain_response",
            classification=record.classification,
            share_scope={recipient},
            conclusion={
                "status": "grounded",
                "summary": record.metadata.get("summary", record.text),
                **shareable,
            },
            evidence=[Evidence(source_id=record.record_id, fragment_id=f"{record.record_id}:0")],
            restricted_fields=restricted_fields,
            sensitive_values=sensitive_values,
        )


class LLMDomainAgent(DomainAgent):
    """Domain agent that shares one LLM client but keeps a private retriever."""

    def __init__(
        self,
        domain: Domain,
        knowledge_base: Retriever,
        llm: GroundedLLM,
        *,
        safe_context: bool = False,
    ) -> None:
        super().__init__(domain, knowledge_base)
        self.llm = llm
        self.safe_context = safe_context

    def answer(self, task_id: str, query: str, recipient: Domain) -> AgentMessage:
        records = self.knowledge_base.search(query)
        if not records:
            return super().answer(task_id, query, recipient)
        synthesis_records = [safe_llm_record(record) for record in records] if self.safe_context else records
        synthesis = self.llm.synthesize(query, self.domain, synthesis_records)
        restricted_fields: set[str] = set()
        sensitive_values: set[str] = set()
        deterministic_facts: dict[str, Any] = {}
        for record in records:
            shareable = cast(dict[str, Any], record.metadata.get("shareable", {}))
            current_fields = {
                str(value)
                for value in cast(list[object], record.metadata.get("restricted_fields", []))
            }
            restricted_fields.update(current_fields)
            sensitive_values.update(
                str(shareable[field_name])
                for field_name in current_fields
                if field_name in shareable
            )
            if self.safe_context:
                deterministic_facts.update(safe_shareable(record))
        classification_order = {"public": 0, "internal": 1, "restricted": 2}
        facts = deterministic_facts if self.safe_context else synthesis.facts
        return AgentMessage(
            message_id=str(uuid4()),
            task_id=task_id,
            sender=self.domain,
            recipient=recipient,
            purpose="llm_domain_response",
            classification=max(
                (record.classification for record in records),
                key=lambda value: classification_order[value.value],
            ),
            share_scope={recipient},
            conclusion={"status": "grounded", "summary": synthesis.summary, **facts},
            evidence=[
                Evidence(source_id=record.record_id, fragment_id=f"{record.record_id}:0")
                for record in records
            ],
            restricted_fields=restricted_fields,
            sensitive_values=sensitive_values,
            telemetry={
                "model": synthesis.model,
                "prompt_tokens": synthesis.usage.prompt_tokens,
                "completion_tokens": synthesis.usage.completion_tokens,
                "total_tokens": synthesis.usage.total_tokens,
                "llm_latency_ms": synthesis.latency_ms,
                "safe_context": str(self.safe_context).lower(),
            },
        )
