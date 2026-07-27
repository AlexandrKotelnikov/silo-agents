from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from .benchmark import BenchmarkCase, CaseKind
from .config import LiveSettings
from .datasets import load_cases
from .embeddings import OllamaEmbedder
from .llm import OpenAICompatibleGroundedLLM
from .models import AgentMessage, Classification, Domain, Evidence, RetrievalRecord
from .orchestrator import ExperimentMode
from .qdrant import QdrantRestClient
from .runtime import build_qdrant_llm_system


class ComparativeCaseResult(BaseModel):
    mode: ExperimentMode
    case_id: str
    repeat: int
    expected_domains: set[Domain]
    selected_domains: set[Domain]
    routing_correct: bool | None
    task_correct: bool | None
    abstained: bool
    abstention_correct: bool | None
    leaked_canaries: set[str]
    contamination: bool
    provenance_present: bool | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    delivered_messages: int


class ComparativeModeMetrics(BaseModel):
    mode: ExperimentMode
    runs: int
    routing_accuracy: float | None
    task_accuracy: float | None
    leakage_rate: float
    contamination_rate: float
    abstention_accuracy: float | None
    provenance_coverage: float | None
    mean_tokens: float
    median_latency_ms: float
    p95_latency_ms: float


class ComparativeReport(BaseModel):
    schema_version: str = "2.1-live-comparison"
    model: str
    embedding_model: str
    case_count: int
    repeats: int
    metrics: list[ComparativeModeMetrics]
    results: list[ComparativeCaseResult]

    def write(self, output: Path) -> tuple[Path, Path, Path]:
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "report.json"
        markdown_path = output / "report.md"
        csv_path = output / "results.csv"
        json_path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        markdown_path.write_text(render_markdown(self), encoding="utf-8")
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(_csv_row(self.results[0]).keys()))
            writer.writeheader()
            writer.writerows(_csv_row(result) for result in self.results)
        return json_path, markdown_path, csv_path


def _ratio(values: list[bool]) -> float | None:
    return None if not values else sum(values) / len(values)


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _contains_expected(messages: list[AgentMessage], expected: dict[str, Any]) -> bool:
    if not expected:
        return True
    payload = json.dumps([message.conclusion for message in messages], ensure_ascii=False)
    return all(json.dumps(value, ensure_ascii=False) in payload for value in expected.values())


def _highest_classification(records: list[RetrievalRecord]) -> Classification:
    rank = {Classification.PUBLIC: 0, Classification.INTERNAL: 1, Classification.RESTRICTED: 2}
    return max((record.classification for record in records), key=rank.__getitem__)


def _shared_run(
    client: QdrantRestClient,
    collection: str,
    embedder: OllamaEmbedder,
    llm: OpenAICompatibleGroundedLLM,
    case: BenchmarkCase,
) -> tuple[set[Domain], list[AgentMessage], bool]:
    points = client.query(
        collection,
        vector=embedder.embed(case.query),
        query_filter={},
        limit=6,
    )
    records: list[RetrievalRecord] = []
    for point in points:
        payload = point.get("payload")
        if isinstance(payload, dict):
            records.append(RetrievalRecord.model_validate(payload))
    query_terms = {term.casefold() for term in case.query.split()}
    records = [
        record
        for record in records
        if query_terms & {term.casefold() for term in record.text.split()}
    ][:3]
    if not records:
        return set(), [], True
    synthesis = llm.synthesize(case.query, Domain.ORCHESTRATOR, records)
    message = AgentMessage(
        message_id=str(uuid4()),
        task_id=case.case_id,
        sender=Domain.ORCHESTRATOR,
        recipient=Domain.ORCHESTRATOR,
        purpose="shared_rag_llm_response",
        classification=_highest_classification(records),
        share_scope={Domain.ORCHESTRATOR},
        conclusion={"status": "grounded", "summary": synthesis.summary, **synthesis.facts},
        evidence=[
            Evidence(source_id=record.record_id, fragment_id=f"{record.record_id}:0")
            for record in records
        ],
        telemetry={
            "model": synthesis.model,
            "prompt_tokens": synthesis.usage.prompt_tokens,
            "completion_tokens": synthesis.usage.completion_tokens,
            "total_tokens": synthesis.usage.total_tokens,
            "llm_latency_ms": synthesis.latency_ms,
        },
    )
    return {record.domain for record in records}, [message], False


def run_comparison(
    settings: LiveSettings,
    cases: list[BenchmarkCase],
    repeats: int,
    *,
    show_progress: bool = True,
) -> ComparativeReport:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    client = QdrantRestClient(settings.qdrant_url, api_key=settings.qdrant_api_key)
    embedder = OllamaEmbedder(settings.embedding_base_url, settings.embedding_model)
    llm = OpenAICompatibleGroundedLLM(
        settings.llm_base_url,
        settings.llm_model,
        api_key=settings.llm_api_key,
        timeout=180.0,
    )
    isolated_system = build_qdrant_llm_system(
        client,
        settings.qdrant_collection,
        embedder,
        llm,
        safe_context=False,
    )
    policy_system = build_qdrant_llm_system(
        client,
        settings.qdrant_collection,
        embedder,
        llm,
        safe_context=True,
    )
    modes = (ExperimentMode.SHARED_RAG, ExperimentMode.ISOLATED_RAG, ExperimentMode.POLICY_GATED)
    total = repeats * len(cases) * len(modes)
    current = 0
    results: list[ComparativeCaseResult] = []
    try:
        for repeat in range(1, repeats + 1):
            for mode in modes:
                for case in cases:
                    current += 1
                    if show_progress:
                        print(f"[{current}/{total}] {mode.value} :: {case.case_id}", flush=True)
                    started = time.perf_counter()
                    if mode == ExperimentMode.SHARED_RAG:
                        selected, messages, abstained = _shared_run(
                            client, settings.qdrant_collection, embedder, llm, case
                        )
                    else:
                        system = (
                            policy_system
                            if mode == ExperimentMode.POLICY_GATED
                            else isolated_system
                        )
                        orchestration = system.run_many(case.query, mode)
                        selected = set(orchestration.selected_agents)
                        messages = list(orchestration.delivered_messages)
                        abstained = orchestration.abstained or not messages
                    latency_ms = (time.perf_counter() - started) * 1000
                    payload = "\n".join(message.model_dump_json() for message in messages)
                    prompt_tokens = sum(
                        int(message.telemetry.get("prompt_tokens", 0)) for message in messages
                    )
                    completion_tokens = sum(
                        int(message.telemetry.get("completion_tokens", 0)) for message in messages
                    )
                    results.append(
                        ComparativeCaseResult(
                            mode=mode,
                            case_id=case.case_id,
                            repeat=repeat,
                            expected_domains=case.expected_domains,
                            selected_domains=selected,
                            routing_correct=(selected == case.expected_domains)
                            if case.expected_domains
                            else None,
                            task_correct=_contains_expected(messages, case.expected_facts)
                            if case.kind == CaseKind.NORMAL
                            else None,
                            abstained=abstained,
                            abstention_correct=abstained if case.expect_abstain else None,
                            leaked_canaries={
                                canary
                                for canary in case.forbidden_canaries
                                if canary.casefold() in payload.casefold()
                            },
                            contamination=bool(selected - case.expected_domains)
                            if case.expected_domains
                            else bool(selected),
                            provenance_present=all(message.evidence for message in messages)
                            if messages
                            else None,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                            latency_ms=latency_ms,
                            delivered_messages=len(messages),
                        )
                    )
    finally:
        llm.close()
        embedder.close()
        client.close()
    metrics = [_aggregate(mode, results) for mode in modes]
    return ComparativeReport(
        model=settings.llm_model,
        embedding_model=settings.embedding_model,
        case_count=len(cases),
        repeats=repeats,
        metrics=metrics,
        results=results,
    )


def _aggregate(
    mode: ExperimentMode, results: list[ComparativeCaseResult]
) -> ComparativeModeMetrics:
    subset = [result for result in results if result.mode == mode]
    latencies = sorted(result.latency_ms for result in subset)
    p95_index = max(0, min(len(latencies) - 1, round(0.95 * len(latencies) + 0.5) - 1))
    return ComparativeModeMetrics(
        mode=mode,
        runs=len(subset),
        routing_accuracy=_ratio(
            [result.routing_correct for result in subset if result.routing_correct is not None]
        ),
        task_accuracy=_ratio(
            [result.task_correct for result in subset if result.task_correct is not None]
        ),
        leakage_rate=_ratio([bool(result.leaked_canaries) for result in subset]) or 0.0,
        contamination_rate=_ratio([result.contamination for result in subset]) or 0.0,
        abstention_accuracy=_ratio(
            [result.abstention_correct for result in subset if result.abstention_correct is not None]
        ),
        provenance_coverage=_ratio(
            [result.provenance_present for result in subset if result.provenance_present is not None]
        ),
        mean_tokens=statistics.fmean(result.total_tokens for result in subset),
        median_latency_ms=statistics.median(latencies),
        p95_latency_ms=latencies[p95_index],
    )


def _csv_row(result: ComparativeCaseResult) -> dict[str, str | int | float | bool | None]:
    return {
        "mode": result.mode.value,
        "case_id": result.case_id,
        "repeat": result.repeat,
        "expected_domains": ",".join(sorted(domain.value for domain in result.expected_domains)),
        "selected_domains": ",".join(sorted(domain.value for domain in result.selected_domains)),
        "routing_correct": result.routing_correct,
        "task_correct": result.task_correct,
        "abstained": result.abstained,
        "abstention_correct": result.abstention_correct,
        "leaked_canaries": ",".join(sorted(result.leaked_canaries)),
        "contamination": result.contamination,
        "provenance_present": result.provenance_present,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "latency_ms": round(result.latency_ms, 2),
        "delivered_messages": result.delivered_messages,
    }


def render_markdown(report: ComparativeReport) -> str:
    lines = [
        "# SiloAgents comparative live benchmark",
        "",
        f"- LLM: `{report.model}`",
        f"- Embeddings: `{report.embedding_model}`",
        f"- Cases: **{report.case_count}**",
        f"- Repeats: **{report.repeats}**",
        "",
        "| Mode | Routing | Task | Leakage | Contamination | Abstention | Provenance | Mean tokens | Median latency | P95 latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in report.metrics:
        lines.append(
            f"| `{metric.mode.value}` | {_percent(metric.routing_accuracy)} | "
            f"{_percent(metric.task_accuracy)} | {_percent(metric.leakage_rate)} | "
            f"{_percent(metric.contamination_rate)} | {_percent(metric.abstention_accuracy)} | "
            f"{_percent(metric.provenance_coverage)} | {metric.mean_tokens:.1f} | "
            f"{metric.median_latency_ms:.0f} ms | {metric.p95_latency_ms:.0f} ms |"
        )
    lines.extend(["", "## Failed cases", ""])
    failed = [
        result
        for result in report.results
        if result.routing_correct is False
        or result.task_correct is False
        or result.abstention_correct is False
        or result.leaked_canaries
    ]
    if not failed:
        lines.append("No failures in this run.")
    else:
        lines.extend(
            [
                "| Mode | Case | Repeat | Expected | Selected | Route | Task | Abstention | Leaks |",
                "|---|---|---:|---|---|---|---|---|---|",
            ]
        )
        for result in failed:
            lines.append(
                f"| `{result.mode.value}` | `{result.case_id}` | {result.repeat} | "
                f"{','.join(sorted(x.value for x in result.expected_domains)) or '-'} | "
                f"{','.join(sorted(x.value for x in result.selected_domains)) or '-'} | "
                f"{result.routing_correct} | {result.task_correct} | "
                f"{result.abstention_correct} | {','.join(sorted(result.leaked_canaries)) or '-'} |"
            )
    lines.extend(
        [
            "",
            "The three modes share the same local embedding and language models. This is an experimental comparison, not a production security certification.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare three live SiloAgents architectures")
    parser.add_argument("--cases", type=Path, default=Path("benchmarks/tasks_extended.jsonl"))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("reports/live-comparison"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    report = run_comparison(
        LiveSettings.from_env(), load_cases(args.cases), args.repeats, show_progress=not args.quiet
    )
    json_path, markdown_path, csv_path = report.write(args.output)
    print(markdown_path.read_text(encoding="utf-8"))
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    print(f"CSV results: {csv_path}")


if __name__ == "__main__":
    main()
