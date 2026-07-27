from pathlib import Path

from silo_agents.live_compare import (
    ComparativeCaseResult,
    ComparativeReport,
    _aggregate,
)
from silo_agents.models import Domain
from silo_agents.orchestrator import ExperimentMode


def result(
    mode: ExperimentMode,
    *,
    routing: bool = True,
    leakage: bool = False,
    latency: float = 100.0,
) -> ComparativeCaseResult:
    return ComparativeCaseResult(
        mode=mode,
        case_id="case",
        repeat=1,
        expected_domains={Domain.PROCESS},
        selected_domains={Domain.PROCESS},
        routing_correct=routing,
        task_correct=True,
        abstained=False,
        abstention_correct=None,
        leaked_canaries={"ALPHA"} if leakage else set(),
        contamination=False,
        provenance_present=True,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        latency_ms=latency,
        delivered_messages=1,
    )


def test_aggregate_computes_mode_metrics() -> None:
    rows = [
        result(ExperimentMode.POLICY_GATED, latency=100.0),
        result(ExperimentMode.POLICY_GATED, routing=False, leakage=True, latency=300.0),
    ]
    metric = _aggregate(ExperimentMode.POLICY_GATED, rows)
    assert metric.routing_accuracy == 0.5
    assert metric.task_accuracy == 1.0
    assert metric.leakage_rate == 0.5
    assert metric.mean_tokens == 15.0
    assert metric.median_latency_ms == 200.0
    assert metric.p95_latency_ms == 300.0


def test_report_writes_json_markdown_and_csv(tmp_path: Path) -> None:
    row = result(ExperimentMode.POLICY_GATED)
    report = ComparativeReport(
        model="test-model",
        embedding_model="test-embed",
        case_count=1,
        repeats=1,
        metrics=[_aggregate(ExperimentMode.POLICY_GATED, [row])],
        results=[row],
    )
    json_path, markdown_path, csv_path = report.write(tmp_path)
    assert json_path.exists()
    assert markdown_path.exists()
    assert csv_path.exists()
    assert "policy_gated" in markdown_path.read_text(encoding="utf-8")
    assert "selected_domains" in csv_path.read_text(encoding="utf-8")
