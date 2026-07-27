from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from silo_agents import (
    AgentId,
    AgentRegistry,
    DeterministicGroundedLLM,
    Domain,
    HashingEmbedder,
    ProjectSpec,
    QdrantRestClient,
)


def project_payload(count: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": f"project-{count}",
        "orchestrator": {"max_agents_per_query": max(1, count)},
        "agents": [
            {
                "id": f"agent-{index}",
                "name": f"Agent {index}",
                "description": f"Handles topic {index}",
                "routing": {
                    "terms": [f"topic-{index}"],
                    "aliases": {f"тема-{index}": f"topic-{index}"},
                },
            }
            for index in range(count)
        ],
    }


@pytest.mark.parametrize("count", [1, 10, 50])
def test_project_accepts_arbitrary_agent_counts(count: int) -> None:
    project = ProjectSpec.model_validate(project_payload(count))
    assert len(project.agents) == count
    assert project.agents[-1].id.value == f"agent-{count - 1}"
    assert len(project.allowed_routes()) == count


def test_dynamic_agent_id_preserves_legacy_domain_api() -> None:
    assert AgentId("legal").value == "legal"
    assert AgentId("quality-control").value == "quality-control"
    assert Domain.PROCESS.value == "process"
    assert AgentId("process") == Domain.PROCESS


def test_project_loads_yaml_and_custom_policy(tmp_path: Path) -> None:
    path = tmp_path / "project.yaml"
    path.write_text(
        """
schema_version: 1
name: legal-and-finance
agents:
  - id: legal
    name: Legal
  - id: finance
    name: Finance
policy:
  default: deny
  routes:
    legal: [orchestrator]
    finance: [orchestrator, legal]
""".strip(),
        encoding="utf-8",
    )
    project = ProjectSpec.load(path)
    assert {agent.id.value for agent in project.agents} == {"legal", "finance"}
    assert project.allowed_routes()[AgentId("finance")] == {
        AgentId.ORCHESTRATOR,
        AgentId("legal"),
    }


def test_registry_builds_every_configured_agent_without_network_calls() -> None:
    project = ProjectSpec.model_validate(project_payload(10))
    client = QdrantRestClient(
        "http://qdrant",
        transport=httpx.MockTransport(lambda request: httpx.Response(500)),
    )
    system = AgentRegistry(project).build_qdrant_system(
        client,
        "records",
        HashingEmbedder(),
        DeterministicGroundedLLM(),
    )
    assert {agent_id.value for agent_id in system.agents} == {
        f"agent-{index}" for index in range(10)
    }
    assert system.max_agents == 10
    client.close()


def test_duplicate_and_unknown_policy_agents_fail_closed() -> None:
    duplicate = project_payload(2)
    agents = duplicate["agents"]
    assert isinstance(agents, list)
    second = agents[1]
    assert isinstance(second, dict)
    second["id"] = "agent-0"
    with pytest.raises(ValidationError, match="Agent IDs must be unique"):
        ProjectSpec.model_validate(duplicate)

    unknown_route = project_payload(1)
    unknown_route["policy"] = {
        "default": "deny",
        "routes": {"agent-0": ["missing-agent"]},
    }
    with pytest.raises(ValidationError, match="Unknown policy recipients"):
        ProjectSpec.model_validate(unknown_route)


def test_orchestrator_identifier_is_reserved() -> None:
    payload = project_payload(1)
    agents = payload["agents"]
    assert isinstance(agents, list)
    agent = agents[0]
    assert isinstance(agent, dict)
    agent["id"] = "orchestrator"
    with pytest.raises(ValidationError, match="reserved"):
        ProjectSpec.model_validate(payload)
