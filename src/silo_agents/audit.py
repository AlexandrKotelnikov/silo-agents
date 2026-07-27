from __future__ import annotations

from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from .benchmark import CaseKind
from .datasets import load_cases, load_records
from .project import ProjectSpec


class AuditSection(BaseModel):
    score: int = Field(ge=0, le=100)
    findings: list[str] = Field(default_factory=list)


class ProjectAudit(BaseModel):
    status: str
    overall_score: int = Field(ge=0, le=100)
    blockers: list[str]
    warnings: list[str]
    sections: dict[str, AuditSection]

    def render_markdown(self) -> str:
        lines = [
            "# SiloAgents project audit",
            "",
            f"- Status: **{self.status}**",
            f"- Overall score: **{self.overall_score}/100**",
            "",
            "## Section scores",
            "",
            "| Section | Score |",
            "|---|---:|",
        ]
        for name, section in self.sections.items():
            lines.append(f"| {name.replace('_', ' ').title()} | {section.score}/100 |")
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {item}" for item in self.blockers or ["None"])
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {item}" for item in self.warnings or ["None"])
        lines.extend(["", "## Findings", ""])
        for name, section in self.sections.items():
            lines.append(f"### {name.replace('_', ' ').title()}")
            lines.extend(f"- {item}" for item in section.findings or ["No findings"])
            lines.append("")
        return "\n".join(lines)


def audit_project(project_path: Path) -> ProjectAudit:
    project = ProjectSpec.load(project_path)
    root = project_path.parent
    corpus_path = root / project.paths.corpus
    cases_path = root / project.paths.cases
    blockers: list[str] = []
    warnings: list[str] = []

    records = load_records(corpus_path) if corpus_path.exists() else []
    cases = load_cases(cases_path) if cases_path.exists() else []
    if not records:
        blockers.append("Corpus is missing or empty")
    if not cases:
        blockers.append("Benchmark case set is missing or empty")

    namespaces = {agent.namespace_id for agent in project.agents}
    record_counts = Counter(record.domain for record in records)
    missing_data = namespaces - set(record_counts)
    if missing_data:
        blockers.append(
            "Agents without corpus records: " + ", ".join(sorted(x.value for x in missing_data))
        )

    routing_findings: list[str] = []
    term_owners: dict[str, list[str]] = {}
    for agent in project.agents:
        if len(agent.routing.terms) < 3:
            warnings.append(f"Agent {agent.id.value} has fewer than three routing terms")
        for term in agent.routing.terms:
            term_owners.setdefault(term.casefold(), []).append(agent.id.value)
    overlaps = {term: owners for term, owners in term_owners.items() if len(owners) > 1}
    if overlaps:
        warnings.append("Routing vocabulary overlaps across agents")
        routing_findings.extend(
            f"{term}: {', '.join(sorted(owners))}" for term, owners in sorted(overlaps.items())
        )

    kinds = Counter(case.kind for case in cases)
    has_collaboration = any(len(case.expected_domains) > 1 for case in cases)
    attack_canaries = set().union(
        *(case.forbidden_canaries for case in cases if case.kind == CaseKind.ATTACK)
    ) if cases else set()
    security_findings: list[str] = []
    if kinds[CaseKind.ATTACK] == 0:
        blockers.append("No attack benchmark cases are defined")
    if not attack_canaries:
        blockers.append("Attack cases do not define forbidden canaries")
    if kinds[CaseKind.ABSTENTION] == 0:
        blockers.append("No abstention case is defined")
    if len(project.agents) > 1 and not has_collaboration:
        blockers.append("No multi-agent collaboration case is defined")
    security_findings.append(f"Attack cases: {kinds[CaseKind.ATTACK]}")
    security_findings.append(f"Abstention cases: {kinds[CaseKind.ABSTENTION]}")
    security_findings.append(f"Forbidden canaries: {len(attack_canaries)}")

    restricted_records = sum(
        bool(record.metadata.get("restricted_fields")) for record in records
    )
    if records and restricted_records == 0:
        warnings.append("No corpus record declares restricted_fields")

    configuration_score = 100
    data_score = max(0, 100 - 30 * len(missing_data)) if project.agents else 0
    routing_score = max(0, 100 - 15 * len(overlaps) - 10 * sum(
        len(agent.routing.terms) < 3 for agent in project.agents
    ))
    security_penalties = 25 * (kinds[CaseKind.ATTACK] == 0) + 25 * (not attack_canaries)
    security_penalties += 20 * (kinds[CaseKind.ABSTENTION] == 0)
    security_penalties += 20 * (len(project.agents) > 1 and not has_collaboration)
    security_score = max(0, 100 - security_penalties)
    readiness_score = 100 if records and cases else 20

    sections = {
        "configuration": AuditSection(
            score=configuration_score,
            findings=[f"Configured agents: {len(project.agents)}", "Policy default is deny"],
        ),
        "data_coverage": AuditSection(
            score=data_score,
            findings=[
                f"Corpus records: {len(records)}",
                *(
                    f"{namespace.value}: {record_counts[namespace]} records"
                    for namespace in sorted(namespaces, key=lambda x: x.value)
                ),
            ],
        ),
        "routing_quality": AuditSection(
            score=routing_score,
            findings=routing_findings or ["No routing-term overlap detected"],
        ),
        "security_tests": AuditSection(score=security_score, findings=security_findings),
        "operational_readiness": AuditSection(
            score=readiness_score,
            findings=[f"Benchmark cases: {len(cases)}", f"Restricted-field records: {restricted_records}"],
        ),
    }
    overall = round(sum(section.score for section in sections.values()) / len(sections))
    status = "ready" if not blockers and overall >= 80 else "needs_work"
    return ProjectAudit(
        status=status,
        overall_score=overall,
        blockers=blockers,
        warnings=warnings,
        sections=sections,
    )
