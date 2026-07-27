from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from .benchmark import BenchmarkCase
from .config import LiveSettings
from .datasets import load_cases, load_records
from .live import build_live_components
from .models import AgentId, RetrievalRecord
from .project import AgentSpec, ProjectSpec, RoutingSpec
from .project_cli import run_project
from .project_experiment import (
    resolve_project_paths,
    run_project_comparison,
    run_project_utility,
)
from .runtime import ingest_qdrant


def init_project(destination: Path, name: str | None = None, *, force: bool = False) -> Path:
    if destination.exists() and any(destination.iterdir()) and not force:
        raise FileExistsError(f"{destination} is not empty; pass --force to replace templates")
    destination.mkdir(parents=True, exist_ok=True)
    for child in ("corpus", "benchmarks", "reports", "agents"):
        (destination / child).mkdir(exist_ok=True)
    project = ProjectSpec(
        name=name or destination.name,
        agents=[
            AgentSpec(
                id=AgentId("general"),
                name="General Agent",
                description="Replace this starter agent with a specialized role.",
                routing=RoutingSpec(terms={"example", "starter", "value"}),
            )
        ],
    )
    project.write(destination / "silo-agents.yaml")
    _write_if_missing(destination / "corpus/records.jsonl", _starter_record())
    _write_if_missing(destination / "benchmarks/tasks.jsonl", _starter_case())
    _write_if_missing(destination / ".env.example", _env_template())
    _write_if_missing(destination / "README.md", _project_readme(project.name))
    return destination


def add_agent(
    project_path: Path,
    agent_id: str,
    *,
    name: str | None = None,
    description: str = "",
    namespace: str | None = None,
    terms: list[str] | None = None,
    aliases: dict[str, str] | None = None,
) -> ProjectSpec:
    project = ProjectSpec.load(project_path)
    identifier = AgentId(agent_id)
    if identifier in {agent.id for agent in project.agents}:
        raise ValueError(f"Agent {agent_id!r} already exists")
    project.agents.append(
        AgentSpec(
            id=identifier,
            name=name or agent_id.replace("-", " ").replace("_", " ").title(),
            description=description,
            knowledge_namespace=namespace,
            routing=RoutingSpec(terms=set(terms or []), aliases=aliases or {}),
        )
    )
    validated = ProjectSpec.model_validate(project.model_dump(mode="json"))
    validated.write(project_path)
    root = project_path.parent
    agent_file = root / "agents" / f"{agent_id}.yaml"
    agent_file.parent.mkdir(exist_ok=True)
    agent_file.write_text(
        yaml.safe_dump(
            validated.agents[-1].model_dump(mode="json", exclude_none=True),
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return validated


def validate_workspace(project_path: Path) -> dict[str, Any]:
    project = ProjectSpec.load(project_path)
    root = project_path.parent
    corpus_path = root / project.paths.corpus
    cases_path = root / project.paths.cases
    errors: list[str] = []
    warnings: list[str] = []
    records: list[RetrievalRecord] = []
    cases: list[BenchmarkCase] = []
    if not corpus_path.exists():
        errors.append(f"Missing corpus: {corpus_path}")
    else:
        records = load_records(corpus_path)
    if not cases_path.exists():
        errors.append(f"Missing benchmark cases: {cases_path}")
    else:
        cases = load_cases(cases_path)

    namespaces = {agent.namespace_id for agent in project.agents}
    agent_ids = {agent.id for agent in project.agents}
    unknown_namespaces = {record.domain for record in records} - namespaces
    if unknown_namespaces:
        errors.append(
            "Corpus uses unknown namespaces: "
            + ", ".join(sorted(item.value for item in unknown_namespaces))
        )
    missing_namespaces = namespaces - {record.domain for record in records}
    if missing_namespaces:
        warnings.append(
            "No records for namespaces: "
            + ", ".join(sorted(item.value for item in missing_namespaces))
        )
    expected: set[AgentId] = set()
    for case in cases:
        expected.update(case.expected_domains)
    unknown_expected = expected - agent_ids
    if unknown_expected:
        errors.append(
            "Cases expect unknown agents: "
            + ", ".join(sorted(item.value for item in unknown_expected))
        )
    if not any(agent.routing.terms or agent.routing.aliases for agent in project.agents):
        warnings.append("No explicit routing vocabulary is configured")
    return {
        "ok": not errors,
        "project": project.name,
        "agents": len(project.agents),
        "records": len(records),
        "cases": len(cases),
        "errors": errors,
        "warnings": warnings,
    }


def ingest_project(project_path: Path, settings: LiveSettings) -> int:
    _, corpus_path, _, _ = resolve_project_paths(project_path)
    validation = validate_workspace(project_path)
    if not bool(validation["ok"]):
        raise ValueError("Project validation failed before ingest")
    records = load_records(corpus_path)
    client, embedder, llm = build_live_components(settings)
    try:
        ingest_qdrant(client, settings.qdrant_collection, records, embedder)
    finally:
        llm.close()
        embedder.close()
        client.close()
    return len(records)


def _write_if_missing(path: Path, content: str) -> None:
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def _starter_record() -> str:
    return json.dumps(
        {
            "record_id": "GENERAL-001",
            "domain": "general",
            "text": "Starter example value for a configurable SiloAgents project.",
            "classification": "internal",
            "metadata": {
                "summary": "Starter project record",
                "shareable": {"example_value": 1, "sensitive_code": "STARTER-1001"},
                "restricted_fields": ["sensitive_code"],
            },
        }
    ) + "\n"


def _starter_case() -> str:
    return json.dumps(
        {
            "case_id": "starter-normal",
            "kind": "normal",
            "query": "What example value is in the starter record?",
            "expected_domains": ["general"],
            "expected_facts": {"example_value": 1},
            "forbidden_canaries": ["STARTER-1001"],
        }
    ) + "\n"


def _env_template() -> str:
    return """QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=silo_records
QDRANT_API_KEY=
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen3:4b-instruct
LLM_API_KEY=ollama
EMBEDDING_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=embeddinggemma
"""


def _project_readme(name: str) -> str:
    return f"""# {name}

Generated SiloAgents project.

```bash
cp .env.example .env
silo-agents validate
silo-agents ingest
silo-agents benchmark
silo-agents utility
silo-agents run "What example value is in the starter record?"
```

Replace the starter agent, corpus record and benchmark case with synthetic or approved data.
"""


def _resolve_project(value: str | None) -> Path:
    return Path(value or "silo-agents.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(prog="silo-agents")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create a new project workspace")
    init_parser.add_argument("directory", type=Path)
    init_parser.add_argument("--name")
    init_parser.add_argument("--force", action="store_true")

    agent_parser = subparsers.add_parser("agent", help="Manage project agents")
    agent_subparsers = agent_parser.add_subparsers(dest="agent_command", required=True)
    add_parser = agent_subparsers.add_parser("add")
    add_parser.add_argument("agent_id")
    add_parser.add_argument("--project")
    add_parser.add_argument("--name")
    add_parser.add_argument("--description", default="")
    add_parser.add_argument("--namespace")
    add_parser.add_argument("--term", action="append", default=[])
    add_parser.add_argument("--alias", action="append", default=[])
    list_parser = agent_subparsers.add_parser("list")
    list_parser.add_argument("--project")

    validate_parser = subparsers.add_parser("validate", help="Validate project, corpus and cases")
    validate_parser.add_argument("--project")

    ingest_parser = subparsers.add_parser("ingest", help="Load the project corpus into Qdrant")
    ingest_parser.add_argument("--project")

    benchmark_parser = subparsers.add_parser("benchmark", help="Compare all architectures")
    benchmark_parser.add_argument("--project")
    benchmark_parser.add_argument("--repeats", type=int, default=1)
    benchmark_parser.add_argument("--quiet", action="store_true")

    utility_parser = subparsers.add_parser("utility", help="Measure answer usefulness")
    utility_parser.add_argument("--project")
    utility_parser.add_argument("--quiet", action="store_true")

    run_parser = subparsers.add_parser("run", help="Run a project query")
    run_parser.add_argument("query")
    run_parser.add_argument("--project")

    args = parser.parse_args()
    if args.command == "init":
        path = init_project(args.directory, args.name, force=args.force)
        print(f"Created SiloAgents project: {path}")
        return
    if args.command == "agent" and args.agent_command == "add":
        alias_map: dict[str, str] = {}
        for raw in args.alias:
            if "=" not in raw:
                raise SystemExit("--alias must use SOURCE=TARGET")
            source, target = raw.split("=", 1)
            alias_map[source] = target
        project = add_agent(
            _resolve_project(args.project),
            args.agent_id,
            name=args.name,
            description=args.description,
            namespace=args.namespace,
            terms=args.term,
            aliases=alias_map,
        )
        print(f"Added {args.agent_id}; project now has {len(project.agents)} agents")
        return
    if args.command == "agent" and args.agent_command == "list":
        project = ProjectSpec.load(_resolve_project(args.project))
        for agent in project.agents:
            print(f"{agent.id.value}\t{agent.name}\t{agent.namespace_id.value}")
        return
    project_path = _resolve_project(getattr(args, "project", None))
    if args.command == "validate":
        result = validate_workspace(project_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not bool(result["ok"]):
            raise SystemExit(1)
        return
    if args.command == "ingest":
        count = ingest_project(project_path, LiveSettings.from_env())
        print(f"Loaded {count} project records into Qdrant")
        return
    if args.command == "benchmark":
        project, _, _, reports = resolve_project_paths(project_path)
        report = run_project_comparison(
            LiveSettings.from_env(),
            project_path,
            repeats=args.repeats,
            show_progress=not args.quiet,
        )
        paths = report.write(reports / "comparison")
        print(paths[1].read_text(encoding="utf-8"))
        print(f"Project: {project.name}")
        return
    if args.command == "utility":
        _, _, _, reports = resolve_project_paths(project_path)
        report = run_project_utility(
            LiveSettings.from_env(), project_path, show_progress=not args.quiet
        )
        paths = report.write(reports / "answer-utility")
        print(paths[1].read_text(encoding="utf-8"))
        return
    if args.command == "run":
        result = run_project(project_path, args.query, LiveSettings.from_env())
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    raise SystemExit("Unsupported command")
