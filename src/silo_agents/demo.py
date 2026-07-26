from __future__ import annotations

from .agents import DomainAgent
from .models import Classification, Domain, RetrievalRecord
from .orchestrator import BlindOrchestrator
from .policy import PolicyGateway
from .rag import IsolatedKnowledgeBase


def build_demo_system() -> BlindOrchestrator:
    records = [
        RetrievalRecord(record_id="PROC-001", domain=Domain.PROCESS, text="Reactor throughput is limited by cooling capacity and temperature stability.", metadata={"summary": "Cooling capacity limits throughput", "shareable": {"limit_percent": 2.8}}),
        RetrievalRecord(record_id="MAINT-001", domain=Domain.MAINTENANCE, text="The circulation pump has a medium maintenance risk above the current load.", metadata={"summary": "Circulation pump risk is medium", "shareable": {"limit_percent": 3.5}}),
        RetrievalRecord(record_id="ECO-001", domain=Domain.ECONOMICS, text="Economic effect depends on sale price, variable cost and feasible throughput increase.", classification=Classification.INTERNAL, metadata={"summary": "Economic calculation requires approved production increase"}),
    ]
    agents = {domain: DomainAgent(domain, IsolatedKnowledgeBase(domain, records)) for domain in (Domain.PROCESS, Domain.MAINTENANCE, Domain.ECONOMICS)}
    routes = {domain: {Domain.ORCHESTRATOR} for domain in agents}
    return BlindOrchestrator(agents, PolicyGateway(routes))
