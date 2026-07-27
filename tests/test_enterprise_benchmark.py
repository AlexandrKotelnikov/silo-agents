from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from silo_agents.datasets import load_cases, load_records
from silo_agents.enterprise_benchmark import PROFILES, generate_enterprise_benchmark
from silo_agents.project import ProjectSpec


def test_smoke_profile_is_reproducible_and_structurally_complete(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    manifest_a = generate_enterprise_benchmark(first, profile_name="smoke", seed=7)
    manifest_b = generate_enterprise_benchmark(second, profile_name="smoke", seed=7)

    assert manifest_a == manifest_b
    assert (first / "corpus/records.jsonl").read_text() == (
        second / "corpus/records.jsonl"
    ).read_text()
    assert (first / "benchmarks/tasks.jsonl").read_text() == (
        second / "benchmarks/tasks.jsonl"
    ).read_text()

    project = ProjectSpec.load(first / "silo-agents.yaml")
    records = load_records(first / "corpus/records.jsonl")
    cases = load_cases(first / "benchmarks/tasks.jsonl")

    assert len(project.agents) == 6
    assert len(records) == 6 * PROFILES["smoke"].documents_per_agent
    assert len(cases) == manifest_a["case_count"]
    assert {record.domain.value for record in records} == {agent.id.value for agent in project.agents}
    effective = Counter(
        record.domain.value for record in records if bool(record.metadata["effective"])
    )
    assert effective == Counter({agent.id.value: 1 for agent in project.agents})
    assert manifest_a["effective_sources_per_agent"] == dict(sorted(effective.items()))


def test_medium_profile_contains_real_difficulty_dimensions(tmp_path: Path) -> None:
    destination = tmp_path / "medium"
    manifest = generate_enterprise_benchmark(destination, profile_name="medium", seed=42)
    records = load_records(destination / "corpus/records.jsonl")
    cases = load_cases(destination / "benchmarks/tasks.jsonl")

    kinds = {str(record.metadata["document_kind"]) for record in records}
    assert {
        "current",
        "obsolete",
        "superseded",
        "near_duplicate",
        "partial",
        "restricted",
        "prompt_injection",
        "cross_domain_noise",
    } <= kinds
    assert manifest["record_count"] == 300
    assert manifest["case_count"] >= 60
    assert manifest["record_kind_counts"]["current"] == 6
    assert any(case.kind.value == "attack" for case in cases)
    assert any(case.expect_abstain for case in cases)
    assert any(len(case.expected_domains) >= 3 for case in cases)
    assert any("Ignore previous instructions" in record.text for record in records)
    assert any(len(record.metadata["restricted_fields"]) > 1 for record in records)


def test_manifest_is_machine_readable_and_documents_limitations(tmp_path: Path) -> None:
    destination = tmp_path / "pack"
    generate_enterprise_benchmark(destination, profile_name="smoke", seed=1)
    manifest = json.loads((destination / "manifest.json").read_text())

    assert manifest["generator"] == "silo-agents-enterprise-benchmark"
    assert manifest["seed"] == 1
    assert manifest["limitations"]
    assert "current retriever does not yet rank" in manifest["limitations"][1]
