from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import LiveSettings
from .embeddings import OllamaEmbedder
from .final_answer import synthesize_final_answer
from .llm import OpenAICompatibleGroundedLLM
from .orchestrator import ExperimentMode
from .project import AgentRegistry, ProjectSpec
from .qdrant import QdrantRestClient


def validate_project(path: Path) -> ProjectSpec:
    return ProjectSpec.load(path)


def run_project(
    path: Path, query: str, settings: LiveSettings, *, include_trace: bool = False
) -> dict[str, Any]:
    project = ProjectSpec.load(path)
    client = QdrantRestClient(settings.qdrant_url, api_key=settings.qdrant_api_key)
    embedder = OllamaEmbedder(settings.embedding_base_url, settings.embedding_model)
    llm = OpenAICompatibleGroundedLLM(
        settings.llm_base_url,
        settings.llm_model,
        api_key=settings.llm_api_key,
        timeout=180.0,
    )
    try:
        system = AgentRegistry(project).build_qdrant_system(
            client,
            settings.qdrant_collection,
            embedder,
            llm,
            safe_context=True,
        )
        result = system.run_many(query, ExperimentMode.POLICY_GATED)
        delivered = result.delivered_messages
        payload: dict[str, Any] = {
            "project": project.name,
            "query": query,
            "selected_agents": [agent_id.value for agent_id in result.selected_agents],
            "abstained": result.abstained,
            "final_answer": synthesize_final_answer(query, delivered).model_dump(mode="json"),
        }
        if include_trace:
            payload["trace"] = {
                "messages": [message.model_dump(mode="json") for message in delivered],
                "policy_decisions": [
                    decision.model_dump(mode="json") for decision in result.policy_decisions
                ],
            }
        return payload
    finally:
        llm.close()
        embedder.close()
        client.close()


def validate_main() -> None:
    parser = argparse.ArgumentParser(description="Validate a SiloAgents project file")
    parser.add_argument("project", type=Path)
    args = parser.parse_args()
    project = validate_project(args.project)
    print(f"[OK] {project.name}: {len(project.agents)} agents")
    for agent in project.agents:
        print(f"- {agent.id.value}: {agent.name}")


def run_main() -> None:
    parser = argparse.ArgumentParser(description="Run a configured SiloAgents project")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--trace", action="store_true", help="Include internal approved messages")
    parser.add_argument("query")
    args = parser.parse_args()
    result = run_project(
        args.project, args.query, LiveSettings.from_env(), include_trace=args.trace
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
