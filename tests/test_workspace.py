from pathlib import Path

from silo_agents.models import AgentId
from silo_agents.project import AgentRegistry, AgentSpec, ProjectSpec
from silo_agents.workspace import add_agent, init_project, validate_workspace


def test_init_creates_complete_valid_workspace(tmp_path: Path) -> None:
    root = init_project(tmp_path / "demo", "demo-project")
    assert (root / "silo-agents.yaml").exists()
    assert (root / "corpus/records.jsonl").exists()
    assert (root / "benchmarks/tasks.jsonl").exists()
    result = validate_workspace(root / "silo-agents.yaml")
    assert result["ok"] is True
    assert result["agents"] == 1
    assert result["records"] == 1
    assert result["cases"] == 1


def test_add_agent_requires_no_framework_source_change(tmp_path: Path) -> None:
    root = init_project(tmp_path / "demo")
    project_path = root / "silo-agents.yaml"
    updated = add_agent(
        project_path,
        "legal",
        namespace="approved-contracts",
        terms=["contract", "termination"],
        aliases={"договор": "contract"},
    )
    assert AgentId("legal") in {agent.id for agent in updated.agents}
    legal = next(agent for agent in updated.agents if agent.id == AgentId("legal"))
    assert legal.namespace_id == AgentId("approved-contracts")
    assert (root / "agents/legal.yaml").exists()


def test_registry_keeps_agent_identity_separate_from_namespace() -> None:
    project = ProjectSpec(
        name="namespace-test",
        agents=[
            AgentSpec(
                id=AgentId("contract-reviewer"),
                name="Contract reviewer",
                knowledge_namespace="approved-contracts",
            )
        ],
    )
    assert project.namespace_to_agent() == {
        AgentId("approved-contracts"): AgentId("contract-reviewer")
    }
    assert AgentRegistry(project).ids == (AgentId("contract-reviewer"),)


def test_validate_rejects_unknown_corpus_namespace(tmp_path: Path) -> None:
    root = init_project(tmp_path / "demo")
    (root / "corpus/records.jsonl").write_text(
        '{"record_id":"X","domain":"unknown","text":"x"}\n', encoding="utf-8"
    )
    result = validate_workspace(root / "silo-agents.yaml")
    assert result["ok"] is False
    assert "unknown" in " ".join(result["errors"])
