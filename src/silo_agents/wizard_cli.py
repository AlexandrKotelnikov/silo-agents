from __future__ import annotations

import argparse
import json
import shutil
import urllib.error
import urllib.request
from pathlib import Path

from .audit import audit_project
from .wizard import WizardBlueprint, generate_project, run_interactive_wizard


def wizard_main() -> None:
    parser = argparse.ArgumentParser(description="Design a reviewable SiloAgents project")
    parser.add_argument("--blueprint", type=Path)
    parser.add_argument("--save-blueprint", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    blueprint = (
        WizardBlueprint.load(args.blueprint) if args.blueprint else run_interactive_wizard()
    )
    if args.save_blueprint:
        blueprint.write(args.save_blueprint)
    project_path, audit = generate_project(blueprint, force=args.force)
    print(f"Created project: {project_path}")
    print(audit.render_markdown())
    if audit.status != "ready":
        print("The generated project needs review before ingestion or real use.")


def audit_main() -> None:
    parser = argparse.ArgumentParser(description="Audit SiloAgents project readiness")
    parser.add_argument("--project", type=Path, default=Path("silo-agents.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    audit = audit_project(args.project)
    markdown = audit.render_markdown()
    print(markdown)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "project-audit.json").write_text(
            audit.model_dump_json(indent=2), encoding="utf-8"
        )
        (args.output / "project-audit.md").write_text(markdown, encoding="utf-8")
    if args.require_ready and audit.status != "ready":
        raise SystemExit(1)


def doctor_main() -> None:
    parser = argparse.ArgumentParser(description="Check local SiloAgents runtime prerequisites")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    checks = {
        "python": {"ok": True, "detail": "current interpreter is running"},
        "docker": _command_check("docker"),
        "ollama": _command_check("ollama"),
        "env": {
            "ok": Path(".env").exists(),
            "detail": ".env exists" if Path(".env").exists() else "copy .env.example to .env",
        },
        "qdrant": _url_check("http://localhost:6333/collections"),
        "ollama_api": _url_check("http://localhost:11434/api/tags"),
    }
    ready = all(bool(item["ok"]) for item in checks.values())
    payload = {"ready": ready, "checks": checks}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("SiloAgents doctor")
        for name, result in checks.items():
            marker = "OK" if result["ok"] else "MISSING"
            print(f"[{marker}] {name}: {result['detail']}")
        print("READY" if ready else "NEEDS SETUP")
    if not ready:
        raise SystemExit(1)


def _command_check(command: str) -> dict[str, object]:
    path = shutil.which(command)
    return {
        "ok": path is not None,
        "detail": path or f"{command} is not installed or not on PATH",
    }


def _url_check(url: str) -> dict[str, object]:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return {"ok": 200 <= response.status < 500, "detail": f"HTTP {response.status}"}
    except (urllib.error.URLError, TimeoutError) as exc:
        return {"ok": False, "detail": str(exc)}
