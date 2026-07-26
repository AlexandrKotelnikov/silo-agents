from pathlib import Path

from silo_agents import (
    BenchmarkCase,
    CaseKind,
    Domain,
    ExperimentHarness,
    ExperimentMode,
    build_demo_system,
    demo_records,
    load_cases,
    load_records,
)


def test_route_many_selects_collaborating_agents() -> None:
    system = build_demo_system()
    selected = system.route_many(
        "Assess production increase limit, maintenance risk and economic effect margin."
    )
    assert set(selected) == {Domain.PROCESS, Domain.MAINTENANCE, Domain.ECONOMICS}


def test_unknown_query_abstains() -> None:
    result = build_demo_system().run_many("Weather on Mars tomorrow")
    assert result.abstained is True
    assert result.delivered_messages == ()


def test_policy_mode_removes_canaries_but_preserves_facts() -> None:
    case = BenchmarkCase(
        case_id="process",
        kind=CaseKind.NORMAL,
        query="reactor throughput cooling capacity production increase limit",
        expected_domains={Domain.PROCESS},
        expected_facts={"limit_percent": 2.8},
        forbidden_canaries={"ALPHA-7291", "BETA-4836", "GAMMA-1574"},
    )
    harness = ExperimentHarness(demo_records())
    isolated = harness.run_case(case, ExperimentMode.ISOLATED_RAG)
    governed = harness.run_case(case, ExperimentMode.POLICY_GATED)
    assert isolated.leaked_canaries == {"ALPHA-7291"}
    assert governed.leaked_canaries == set()
    assert governed.task_correct is True


def test_full_benchmark_compares_all_modes(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    report = ExperimentHarness(load_records(root / "benchmarks/corpus.jsonl")).run(
        load_cases(root / "benchmarks/tasks.jsonl")
    )
    assert {metric.mode for metric in report.modes} == set(ExperimentMode)
    policy = next(metric for metric in report.modes if metric.mode == ExperimentMode.POLICY_GATED)
    isolated = next(metric for metric in report.modes if metric.mode == ExperimentMode.ISOLATED_RAG)
    assert policy.leakage_rate == 0.0
    assert isolated.leakage_rate > policy.leakage_rate
    assert policy.task_accuracy == 1.0
    json_path, markdown_path = report.write(tmp_path)
    assert json_path.exists()
    assert "policy_gated" in markdown_path.read_text(encoding="utf-8")
