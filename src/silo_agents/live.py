from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

from .benchmark import BenchmarkCase, CaseKind
from .config import LiveSettings
from .datasets import load_cases, load_records
from .embeddings import OllamaEmbedder
from .llm import OpenAICompatibleGroundedLLM
from .models import AgentMessage, Domain
from .orchestrator import ExperimentMode
from .qdrant import QdrantRestClient
from .runtime import build_qdrant_llm_system, ingest_qdrant


class HealthCheck(BaseModel):
    name: str
    ok: bool
    detail: str


class LiveCaseResult(BaseModel):
    case_id: str
    repeat: int
    selected_domains: set[Domain]
    routing_correct: bool | None
    task_correct: bool | None
    abstention_correct: bool | None
    leaked_canaries: set[str]
    provenance_present: bool | None
    total_tokens: int
    latency_ms: float


class LiveBenchmarkReport(BaseModel):
    schema_version: str = "1.0-live"
    model: str
    embedding_model: str
    repeats: int
    case_count: int
    results: list[LiveCaseResult]
    routing_accuracy: float | None
    task_accuracy: float | None
    leakage_rate: float
    abstention_accuracy: float | None
    provenance_coverage: float | None
    mean_total_tokens: float
    mean_latency_ms: float

    def write(self, output_dir: Path) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "report.json"
        markdown_path = output_dir / "report.md"
        json_path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        markdown_path.write_text(render_live_markdown(self), encoding="utf-8")
        return json_path, markdown_path


def _contains_expected(messages: list[AgentMessage], expected: dict[str, Any]) -> bool:
    if not expected:
        return True
    payload = json.dumps([message.conclusion for message in messages], ensure_ascii=False)
    return all(json.dumps(value, ensure_ascii=False) in payload for value in expected.values())


def _ratio(values: list[bool]) -> float | None:
    return None if not values else sum(values) / len(values)


def run_health(settings: LiveSettings) -> list[HealthCheck]:
    checks: list[HealthCheck] = []
    try:
        response = httpx.get(f"{settings.qdrant_url.rstrip('/')}/collections", timeout=10.0)
        response.raise_for_status()
        checks.append(HealthCheck(name="qdrant", ok=True, detail="REST API reachable"))
    except Exception as exc:
        checks.append(HealthCheck(name="qdrant", ok=False, detail=str(exc)))

    try:
        response = httpx.get(
            f"{settings.embedding_base_url.rstrip('/')}/api/tags", timeout=10.0
        )
        response.raise_for_status()
        raw_models: Any = response.json()
        models = raw_models.get("models", []) if isinstance(raw_models, dict) else []
        names = {
            str(item.get("name", "")).split(":")[0]
            for item in models
            if isinstance(item, dict)
        }
        required = {settings.llm_model.split(":")[0], settings.embedding_model.split(":")[0]}
        missing = sorted(required - names)
        checks.append(
            HealthCheck(
                name="ollama-models",
                ok=not missing,
                detail="models available" if not missing else f"missing: {', '.join(missing)}",
            )
        )
    except Exception as exc:
        checks.append(HealthCheck(name="ollama-models", ok=False, detail=str(exc)))

    embedder = OllamaEmbedder(settings.embedding_base_url, settings.embedding_model)
    try:
        checks.append(
            HealthCheck(name="embedding", ok=True, detail=f"{embedder.dimensions} dimensions")
        )
    except Exception as exc:
        checks.append(HealthCheck(name="embedding", ok=False, detail=str(exc)))
    finally:
        embedder.close()

    try:
        response = httpx.post(
            f"{settings.llm_base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            json={
                "model": settings.llm_model,
                "messages": [
                    {
                        "role": "user",
                        "content": 'Return exactly {"summary":"READY","facts":{}}',
                    }
                ],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
            timeout=120.0,
        )
        response.raise_for_status()
        checks.append(HealthCheck(name="llm", ok=True, detail="chat completion succeeded"))
    except Exception as exc:
        checks.append(HealthCheck(name="llm", ok=False, detail=str(exc)))
    return checks


def build_live_components(
    settings: LiveSettings,
) -> tuple[QdrantRestClient, OllamaEmbedder, OpenAICompatibleGroundedLLM]:
    client = QdrantRestClient(settings.qdrant_url, api_key=settings.qdrant_api_key)
    embedder = OllamaEmbedder(settings.embedding_base_url, settings.embedding_model)
    llm = OpenAICompatibleGroundedLLM(
        settings.llm_base_url,
        settings.llm_model,
        api_key=settings.llm_api_key,
        timeout=180.0,
    )
    return client, embedder, llm


def ingest(settings: LiveSettings, corpus: Path) -> int:
    records = load_records(corpus)
    client, embedder, llm = build_live_components(settings)
    try:
        ingest_qdrant(client, settings.qdrant_collection, records, embedder)
    finally:
        llm.close()
        embedder.close()
        client.close()
    return len(records)


def live_run(settings: LiveSettings, query: str) -> dict[str, Any]:
    client, embedder, llm = build_live_components(settings)
    try:
        system = build_qdrant_llm_system(client, settings.qdrant_collection, embedder, llm)
        result = system.run_many(query, ExperimentMode.POLICY_GATED)
        return {
            "query": query,
            "selected_agents": [domain.value for domain in result.selected_agents],
            "abstained": result.abstained,
            "messages": [
                message.model_dump(mode="json") for message in result.delivered_messages
            ],
            "policy_decisions": [
                decision.model_dump(mode="json") for decision in result.policy_decisions
            ],
        }
    finally:
        llm.close()
        embedder.close()
        client.close()


def live_benchmark(
    settings: LiveSettings, cases: list[BenchmarkCase], repeats: int
) -> LiveBenchmarkReport:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    client, embedder, llm = build_live_components(settings)
    results: list[LiveCaseResult] = []
    try:
        system = build_qdrant_llm_system(client, settings.qdrant_collection, embedder, llm)
        for repeat in range(1, repeats + 1):
            for case in cases:
                started = time.perf_counter()
                orchestration = system.run_many(case.query, ExperimentMode.POLICY_GATED)
                latency_ms = (time.perf_counter() - started) * 1000
                messages = list(orchestration.delivered_messages)
                payload = "\n".join(message.model_dump_json() for message in messages)
                selected = set(orchestration.selected_agents)
                tokens = sum(
                    int(message.telemetry.get("total_tokens", 0)) for message in messages
                )
                results.append(
                    LiveCaseResult(
                        case_id=case.case_id,
                        repeat=repeat,
                        selected_domains=selected,
                        routing_correct=(selected == case.expected_domains)
                        if case.expected_domains
                        else None,
                        task_correct=_contains_expected(messages, case.expected_facts)
                        if case.kind == CaseKind.NORMAL
                        else None,
                        abstention_correct=(orchestration.abstained or not messages)
                        if case.expect_abstain
                        else None,
                        leaked_canaries={
                            value
                            for value in case.forbidden_canaries
                            if value.casefold() in payload.casefold()
                        },
                        provenance_present=all(message.evidence for message in messages)
                        if messages
                        else None,
                        total_tokens=tokens,
                        latency_ms=latency_ms,
                    )
                )
    finally:
        llm.close()
        embedder.close()
        client.close()

    routing = [item.routing_correct for item in results if item.routing_correct is not None]
    task = [item.task_correct for item in results if item.task_correct is not None]
    abstention = [
        item.abstention_correct for item in results if item.abstention_correct is not None
    ]
    provenance = [
        item.provenance_present for item in results if item.provenance_present is not None
    ]
    return LiveBenchmarkReport(
        model=settings.llm_model,
        embedding_model=settings.embedding_model,
        repeats=repeats,
        case_count=len(cases),
        results=results,
        routing_accuracy=_ratio(routing),
        task_accuracy=_ratio(task),
        leakage_rate=_ratio([bool(item.leaked_canaries) for item in results]) or 0.0,
        abstention_accuracy=_ratio(abstention),
        provenance_coverage=_ratio(provenance),
        mean_total_tokens=statistics.fmean(item.total_tokens for item in results),
        mean_latency_ms=statistics.fmean(item.latency_ms for item in results),
    )


def render_live_markdown(report: LiveBenchmarkReport) -> str:
    def percent(value: float | None) -> str:
        return "n/a" if value is None else f"{value * 100:.1f}%"

    return "\n".join(
        [
            "# SiloAgents live benchmark",
            "",
            f"- LLM: `{report.model}`",
            f"- Embeddings: `{report.embedding_model}`",
            f"- Cases: **{report.case_count}**",
            f"- Repeats: **{report.repeats}**",
            "",
            "| Routing | Task | Leakage | Abstention | Provenance | Mean tokens | Mean latency |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            f"| {percent(report.routing_accuracy)} | {percent(report.task_accuracy)} | "
            f"{percent(report.leakage_rate)} | {percent(report.abstention_accuracy)} | "
            f"{percent(report.provenance_coverage)} | {report.mean_total_tokens:.1f} | "
            f"{report.mean_latency_ms:.0f} ms |",
            "",
            "This report describes a local model-backed experiment, not a production security guarantee.",
            "",
        ]
    )


def health_main() -> None:
    settings = LiveSettings.from_env()
    checks = run_health(settings)
    for check in checks:
        print(f"[{'OK' if check.ok else 'FAIL'}] {check.name}: {check.detail}")
    if not all(check.ok for check in checks):
        raise SystemExit(1)


def ingest_main() -> None:
    parser = argparse.ArgumentParser(description="Load a JSONL corpus into Qdrant")
    parser.add_argument("--corpus", type=Path, default=Path("benchmarks/corpus.jsonl"))
    args = parser.parse_args()
    count = ingest(LiveSettings.from_env(), args.corpus)
    print(f"Loaded {count} records into Qdrant")


def run_main() -> None:
    parser = argparse.ArgumentParser(description="Run one live SiloAgents query")
    parser.add_argument("query")
    args = parser.parse_args()
    print(json.dumps(live_run(LiveSettings.from_env(), args.query), ensure_ascii=False, indent=2))


def benchmark_main() -> None:
    parser = argparse.ArgumentParser(description="Run the live Ollama + Qdrant benchmark")
    parser.add_argument("--cases", type=Path, default=Path("benchmarks/tasks.jsonl"))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("reports/live"))
    args = parser.parse_args()
    report = live_benchmark(LiveSettings.from_env(), load_cases(args.cases), args.repeats)
    json_path, markdown_path = report.write(args.output)
    print(markdown_path.read_text(encoding="utf-8"))
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
