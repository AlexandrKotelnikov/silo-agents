from __future__ import annotations

from uuid import uuid4

from .models import AgentMessage, Domain, Evidence
from .rag import Retriever


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
                **record.metadata.get("shareable", {}),
            },
            evidence=[Evidence(source_id=record.record_id, fragment_id=f"{record.record_id}:0")],
            restricted_fields=set(record.metadata.get("restricted_fields", [])),
        )
