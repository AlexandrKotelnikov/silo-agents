from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from .answer_utility import UtilityCaseResult, UtilityModeMetrics, UtilityReport
from .benchmark import CaseKind
from .config import LiveSettings
from .datasets import load_cases
from .embeddings import OllamaEmbedder
from .live_compare import (
    ComparativeCaseResult,
    ComparativeModeMetrics,
    ComparativeReport,
    _shared_run,
)
from .llm import OpenAICompatibleGroundedLLM
from .models import AgentMessage
from .orchestrator import ExperimentMode
from .project import AgentRegistry, ProjectSpec
from .qdrant import QdrantRestClient

_MODES = (ExperimentMode.SHARED_RAG, ExperimentMode.ISOLATED_RAG, ExperimentMode.POLICY_GATED)


def resolve_project_paths(project_path: Path) -> tuple[ProjectSpec, Path, Path, Path]:
    project = ProjectSpec.load(project_path)
    root = project_path.parent
    return (
        project,
        root / project.paths.corpus,
        root / project.paths.cases,
        root / project.paths.reports,
    )


def run_project_comparison(
    settings: LiveSettings,
    project_path: Path,
    *,
    repeats: int = 1,
    show_progress: bool = True,
) -> ComparativeReport:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    project, _, cases_path, _ = resolve_project_paths(project_path)
    cases = load_cases(cases_path)
    namespace_to_agent = project.namespace_to_agent()
    client = QdrantRestClient(settings.qdrant_url, api_key=settings.qdrant_api_key)
    embedder = OllamaEmbedder(settings.embedding_base_url, settings.embedding_model)
    llm = OpenAICompatibleGroundedLLM(
        settings.llm_base_url,
        settings.llm_model,
        api_key=settings.llm_api_key,
        timeout=180.0,
    )
    registry = AgentRegistry(project)
    isolated = registry.build_qdrant_system(
        client, settings.qdrant_collection, embedder, llm, safe_context=False
    )
    governed = registry.build_qdrant_system(
        client, settings.qdrant_collection, embedder, llm, safe_context=True
    )
    results: list[ComparativeCaseResult] = []
    total = repeats * len(cases) * len(_MODES)
    current = 0
    try:
        for repeat in range(1, repeats + 1):
            for mode in _MODES:
                for case in cases:
                    current += 1
                    if show_progress:
                        print(f"[{current}/{total}] {mode.value} :: {case.case_id}", flush=True)
                    started = time.perf_counter()
                    if mode == ExperimentMode.SHARED_RAG:
                        namespaces, messages, abstained = _shared_run(
                            client, settings.qdrant_collection, embedder, llm, case
                        )
                        selected = {
                            namespace_to_agent[namespace]
                            for namespace in namespaces
                            if namespace in namespace_to_agent
                        }
                    else:
                        system = governed if mode == ExperimentMode.POLICY_GATED else isolated
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
                                value
                                for value in case.forbidden_canaries
                                if value.casefold() in payload.casefold()
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
    return ComparativeReport(
        model=settings.llm_model,
        embedding_model=settings.embedding_model,
        case_count=len(cases),
        repeats=repeats,
        metrics=[_comparison_metrics(mode, results) for mode in _MODES],
        results=results,
    )


def run_project_utility(
    settings: LiveSettings,
    project_path: Path,
    *,
    show_progress: bool = True,
) -> UtilityReport:
    project, _, cases_path, _ = resolve_project_paths(project_path)
    cases = [case for case in load_cases(cases_path) if case.kind == CaseKind.NORMAL]
    namespace_to_agent = project.namespace_to_agent()
    client = QdrantRestClient(settings.qdrant_url, api_key=settings.qdrant_api_key)
    embedder = OllamaEmbedder(settings.embedding_base_url, settings.embedding_model)
    llm = OpenAICompatibleGroundedLLM(
        settings.llm_base_url,
        settings.llm_model,
        api_key=settings.llm_api_key,
        timeout=180.0,
    )
    registry = AgentRegistry(project)
    isolated = registry.build_qdrant_system(
        client, settings.qdrant_collection, embedder, llm, safe_context=False
    )
    governed = registry.build_qdrant_system(
        client, settings.qdrant_collection, embedder, llm, safe_context=True
    )
    results: list[UtilityCaseResult] = []
    current = 0
    total = len(cases) * len(_MODES)
    try:
        for mode in _MODES:
            for case in cases:
                current += 1
                if show_progress:
                    print(f"[{current}/{total}] {mode.value} :: {case.case_id}", flush=True)
                started = time.perf_counter()
                if mode == ExperimentMode.SHARED_RAG:
                    namespaces, messages, _ = _shared_run(
                        client, settings.qdrant_collection, embedder, llm, case
                    )
                    selected = {
                        namespace_to_agent[namespace]
                        for namespace in namespaces
                        if namespace in namespace_to_agent
                    }
                else:
                    system = governed if mode == ExperimentMode.POLICY_GATED else isolated
                    orchestration = system.run_many(case.query, mode)
                    selected = set(orchestration.selected_agents)
                    messages = list(orchestration.delivered_messages)
                latency_ms = (time.perf_counter() - started) * 1000
                payload = json.dumps(
                    [message.conclusion for message in messages], ensure_ascii=False
                )
                matched = sum(
                    json.dumps(value, ensure_ascii=False) in payload
                    for value in case.expected_facts.values()
                )
                serialized = "\n".join(message.model_dump_json() for message in messages)
                tokens = sum(
                    int(message.telemetry.get("total_tokens", 0)) for message in messages
                )
                expected_count = len(case.expected_facts)
                results.append(
                    UtilityCaseResult(
                        mode=mode,
                        case_id=case.case_id,
                        query=case.query,
                        expected_domains=case.expected_domains,
                        selected_domains=selected,
                        expected_fact_count=expected_count,
                        matched_fact_count=matched,
                        fact_coverage=(matched / expected_count) if expected_count else 1.0,
                        routing_correct=selected == case.expected_domains,
                        leaked_canaries={
                            value
                            for value in case.forbidden_canaries
                            if value.casefold() in serialized.casefold()
                        },
                        provenance_present=bool(messages)
                        and all(message.evidence for message in messages),
                        total_tokens=tokens,
                        latency_ms=latency_ms,
                        answer=[message.conclusion for message in messages],
                    )
                )
    finally:
        llm.close()
        embedder.close()
        client.close()
    return UtilityReport(
        model=settings.llm_model,
        embedding_model=settings.embedding_model,
        case_count=len(cases),
        metrics=[_utility_metrics(mode, results) for mode in _MODES],
        results=results,
    )


def _contains_expected(messages: list[AgentMessage], expected: dict[str, Any]) -> bool:
    if not expected:
        return True
    payload = json.dumps([message.conclusion for message in messages], ensure_ascii=False)
    return all(json.dumps(value, ensure_ascii=False) in payload for value in expected.values())


def _ratio(values: list[bool]) -> float | None:
    return None if not values else sum(values) / len(values)


def _comparison_metrics(
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


def _utility_metrics(
    mode: ExperimentMode, results: list[UtilityCaseResult]
) -> UtilityModeMetrics:
    subset = [result for result in results if result.mode == mode]
    total_tokens = sum(result.total_tokens for result in subset)
    matched_facts = sum(result.matched_fact_count for result in subset)
    return UtilityModeMetrics(
        mode=mode,
        cases=len(subset),
        mean_fact_coverage=statistics.fmean(result.fact_coverage for result in subset),
        full_answer_rate=sum(result.fact_coverage == 1.0 for result in subset) / len(subset),
        safe_success_rate=sum(result.safe_success for result in subset) / len(subset),
        leakage_rate=sum(bool(result.leaked_canaries) for result in subset) / len(subset),
        mean_tokens=statistics.fmean(result.total_tokens for result in subset),
        median_latency_ms=statistics.median(result.latency_ms for result in subset),
        useful_facts_per_1000_tokens=(matched_facts * 1000 / total_tokens)
        if total_tokens
        else 0.0,
    )
