from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from .agents import DomainAgent
from .models import AgentMessage, Classification, Domain, Evidence, RetrievalRecord
from .orchestrator import BlindOrchestrator, ExperimentMode
from .policy import PolicyGateway
from .rag import IsolatedKnowledgeBase, SharedKnowledgeBase


class CaseKind(StrEnum):
    NORMAL = "normal"
    ATTACK = "attack"
    ABSTENTION = "abstention"


class BenchmarkCase(BaseModel):
    case_id: str
    kind: CaseKind
    query: str
    expected_domains: set[Domain] = Field(default_factory=set)
    expected_facts: dict[str, Any] = Field(default_factory=dict)
    forbidden_canaries: set[str] = Field(default_factory=set)
    expect_abstain: bool = False


class CaseResult(BaseModel):
    case_id: str
    mode: ExperimentMode
    kind: CaseKind
    selected_domains: set[Domain]
    routing_correct: bool | None
    task_correct: bool | None
    abstained: bool
    abstention_correct: bool | None
    leaked_canaries: set[str]
    cross_domain_contamination: bool
    provenance_present: bool | None
    delivered_message_count: int
    delivered_payload_characters: int
    latency_ms: float


class ModeMetrics(BaseModel):
    mode: ExperimentMode
    total_cases: int
    routing_accuracy: float | None
    task_accuracy: float | None
    leakage_rate: float
    cross_domain_contamination_rate: float
    abstention_accuracy: float | None
    provenance_coverage: float | None
    mean_latency_ms: float
    mean_delivered_payload_characters: float


class BenchmarkReport(BaseModel):
    schema_version: str = "1.0"
    case_count: int
    modes: list[ModeMetrics]
    results: list[CaseResult]

    def write(self, output_dir: str | Path) -> tuple[Path, Path]:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        json_path = destination / "report.json"
        markdown_path = destination / "report.md"
        json_path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        markdown_path.write_text(render_markdown(self), encoding="utf-8")
        return json_path, markdown_path


@dataclass(frozen=True)
class LeakageCase:
    case_id: str
    forbidden_canaries: set[str]


@dataclass(frozen=True)
class LeakageResult:
    case_id: str
    leaked: bool
    matched_canaries: set[str]


def evaluate_leakage(case: LeakageCase, message: AgentMessage) -> LeakageResult:
    payload = message.model_dump_json().casefold()
    matches = {canary for canary in case.forbidden_canaries if canary.casefold() in payload}
    return LeakageResult(case.case_id, bool(matches), matches)


class ExperimentHarness:
    """Runs the same cases against three knowledge-isolation architectures."""

    def __init__(self, records: list[RetrievalRecord]) -> None:
        self.records = records
        domains = (Domain.PROCESS, Domain.MAINTENANCE, Domain.ECONOMICS)
        agents = {
            domain: DomainAgent(domain, IsolatedKnowledgeBase(domain, records))
            for domain in domains
        }
        routes = {domain: {Domain.ORCHESTRATOR} for domain in domains}
        self.orchestrator = BlindOrchestrator(agents, PolicyGateway(routes))
        self.shared = SharedKnowledgeBase(records)

    def run_case(self, case: BenchmarkCase, mode: ExperimentMode) -> CaseResult:
        started = time.perf_counter()
        if mode == ExperimentMode.SHARED_RAG:
            selected, delivered = self._run_shared(case)
            abstained = not delivered
        else:
            result = self.orchestrator.run_many(case.query, mode)
            selected = set(result.selected_agents)
            delivered = list(result.delivered_messages)
            abstained = result.abstained or not delivered

        latency_ms = (time.perf_counter() - started) * 1000
        payload = "\n".join(message.model_dump_json() for message in delivered)
        leaked = {
            canary for canary in case.forbidden_canaries if canary.casefold() in payload.casefold()
        }
        routing_correct = selected == case.expected_domains if case.expected_domains else None
        task_correct = (
            _contains_expected_facts(delivered, case.expected_facts)
            if case.kind == CaseKind.NORMAL
            else None
        )
        abstention_correct = abstained if case.expect_abstain else None
        contamination = bool(selected - case.expected_domains) if case.expected_domains else False
        provenance = all(message.evidence for message in delivered) if delivered else None
        return CaseResult(
            case_id=case.case_id,
            mode=mode,
            kind=case.kind,
            selected_domains=selected,
            routing_correct=routing_correct,
            task_correct=task_correct,
            abstained=abstained,
            abstention_correct=abstention_correct,
            leaked_canaries=leaked,
            cross_domain_contamination=contamination,
            provenance_present=provenance,
            delivered_message_count=len(delivered),
            delivered_payload_characters=len(payload),
            latency_ms=latency_ms,
        )

    def run(self, cases: list[BenchmarkCase]) -> BenchmarkReport:
        results = [
            self.run_case(case, mode)
            for mode in ExperimentMode
            for case in cases
        ]
        metrics = [_aggregate_mode(mode, results) for mode in ExperimentMode]
        return BenchmarkReport(case_count=len(cases), modes=metrics, results=results)

    def _run_shared(self, case: BenchmarkCase) -> tuple[set[Domain], list[AgentMessage]]:
        records = self.shared.search(case.query, limit=3)
        if not records:
            return set(), []
        selected = {record.domain for record in records}
        results: list[dict[str, Any]] = []
        evidence: list[Evidence] = []
        for record in records:
            results.append(
                {
                    "domain": record.domain.value,
                    "summary": record.metadata.get("summary", record.text),
                    **record.metadata.get("shareable", {}),
                }
            )
            evidence.append(
                Evidence(source_id=record.record_id, fragment_id=f"{record.record_id}:0")
            )
        message = AgentMessage(
            message_id=str(uuid4()),
            task_id=case.case_id,
            sender=Domain.ORCHESTRATOR,
            recipient=Domain.ORCHESTRATOR,
            purpose="shared_rag_response",
            classification=_highest_classification(records),
            share_scope={Domain.ORCHESTRATOR},
            conclusion={"status": "grounded", "results": results},
            evidence=evidence,
        )
        return selected, [message]


def _highest_classification(records: list[RetrievalRecord]) -> Classification:
    order = {
        Classification.PUBLIC: 0,
        Classification.INTERNAL: 1,
        Classification.RESTRICTED: 2,
    }
    return max((record.classification for record in records), key=order.__getitem__)


def _contains_expected_facts(
    messages: list[AgentMessage], expected_facts: dict[str, Any]
) -> bool:
    if not expected_facts:
        return True
    found: dict[str, list[Any]] = {}
    for message in messages:
        _collect_values(message.conclusion, found)
    return all(value in found.get(key, []) for key, value in expected_facts.items())


def _collect_values(value: Any, found: dict[str, list[Any]]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            found.setdefault(str(key), []).append(child)
            _collect_values(child, found)
    elif isinstance(value, list):
        for child in value:
            _collect_values(child, found)


def _aggregate_mode(mode: ExperimentMode, results: list[CaseResult]) -> ModeMetrics:
    subset = [result for result in results if result.mode == mode]
    routing = [result.routing_correct for result in subset if result.routing_correct is not None]
    tasks = [result.task_correct for result in subset if result.task_correct is not None]
    abstentions = [
        result.abstention_correct
        for result in subset
        if result.abstention_correct is not None
    ]
    provenance = [
        result.provenance_present
        for result in subset
        if result.provenance_present is not None
    ]
    return ModeMetrics(
        mode=mode,
        total_cases=len(subset),
        routing_accuracy=_ratio(routing),
        task_accuracy=_ratio(tasks),
        leakage_rate=_ratio([bool(result.leaked_canaries) for result in subset]) or 0.0,
        cross_domain_contamination_rate=(
            _ratio([result.cross_domain_contamination for result in subset]) or 0.0
        ),
        abstention_accuracy=_ratio(abstentions),
        provenance_coverage=_ratio(provenance),
        mean_latency_ms=statistics.fmean(result.latency_ms for result in subset),
        mean_delivered_payload_characters=statistics.fmean(
            result.delivered_payload_characters for result in subset
        ),
    )


def _ratio(values: list[bool]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def render_markdown(report: BenchmarkReport) -> str:
    lines = [
        "# SiloAgents benchmark report",
        "",
        f"Cases: **{report.case_count}**",
        "",
        "| Mode | Routing | Task | Leakage | Contamination | Abstention | Provenance | Mean payload chars |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in report.modes:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{metric.mode.value}`",
                    _percent(metric.routing_accuracy),
                    _percent(metric.task_accuracy),
                    _percent(metric.leakage_rate),
                    _percent(metric.cross_domain_contamination_rate),
                    _percent(metric.abstention_accuracy),
                    _percent(metric.provenance_coverage),
                    f"{metric.mean_delivered_payload_characters:.0f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Latency is measured for the deterministic local harness only. Token cost is intentionally not reported until a real LLM is connected.",
            "",
        ]
    )
    return "\n".join(lines)


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"
