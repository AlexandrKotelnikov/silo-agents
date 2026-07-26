from __future__ import annotations

from .agents import DomainAgent
from .models import Classification, Domain, RetrievalRecord
from .orchestrator import BlindOrchestrator
from .policy import PolicyGateway
from .rag import IsolatedKnowledgeBase


def demo_records() -> list[RetrievalRecord]:
    return [
        RetrievalRecord(
            record_id="PROC-001",
            domain=Domain.PROCESS,
            text="Reactor throughput cooling capacity production increase temperature stability limit.",
            metadata={
                "summary": "Cooling capacity limits throughput",
                "shareable": {
                    "limit_percent": 2.8,
                    "limiting_factor": "cooling_capacity",
                    "sensitive_code": "ALPHA-7291",
                },
                "restricted_fields": ["sensitive_code"],
            },
        ),
        RetrievalRecord(
            record_id="MAINT-001",
            domain=Domain.MAINTENANCE,
            text="Equipment load circulation pump maintenance risk production increase limit.",
            metadata={
                "summary": "Circulation pump risk is medium",
                "shareable": {
                    "equipment_limit_percent": 3.5,
                    "maintenance_risk": "medium",
                    "sensitive_code": "BETA-4836",
                },
                "restricted_fields": ["sensitive_code"],
            },
        ),
        RetrievalRecord(
            record_id="ECO-001",
            domain=Domain.ECONOMICS,
            text="Production increase economic effect sale price variable cost margin limit.",
            classification=Classification.INTERNAL,
            metadata={
                "summary": "Economic effect uses contribution margin",
                "shareable": {
                    "margin_per_tonne": 420,
                    "sensitive_code": "GAMMA-1574",
                },
                "restricted_fields": ["sensitive_code"],
            },
        ),
    ]


def build_demo_system(records: list[RetrievalRecord] | None = None) -> BlindOrchestrator:
    corpus = records or demo_records()
    domains = (Domain.PROCESS, Domain.MAINTENANCE, Domain.ECONOMICS)
    agents = {
        domain: DomainAgent(domain, IsolatedKnowledgeBase(domain, corpus))
        for domain in domains
    }
    routes = {domain: {Domain.ORCHESTRATOR} for domain in agents}
    return BlindOrchestrator(agents, PolicyGateway(routes))
