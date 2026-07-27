from pathlib import Path

from silo_agents.audit import audit_project
from silo_agents.wizard import WizardAgent, WizardBlueprint, generate_project


def _blueprint(tmp_path: Path) -> WizardBlueprint:
    return WizardBlueprint(
        project_name="decision-support",
        goal="Assess contract and cost risk",
        destination=str(tmp_path / "decision-support"),
        agents=[
            WizardAgent(
                id="legal",
                name="Legal Agent",
                description="Approved contract facts",
                routing_terms=["contract", "termination", "liability"],
                shareable_fields=["notice_days"],
                restricted_fields=["negotiation_code"],
                example_question="What is the approved notice period?",
            ),
            WizardAgent(
                id="finance",
                name="Finance Agent",
                description="Approved financial facts",
                routing_terms=["cost", "budget", "exposure"],
                shareable_fields=["exposure_eur"],
                restricted_fields=["account_code"],
                example_question="What is the approved exposure?",
            ),
        ],
    )


def test_wizard_generates_reviewable_ready_starter(tmp_path: Path) -> None:
    project_path, audit = generate_project(_blueprint(tmp_path))
    root = project_path.parent
    assert project_path.exists()
    assert (root / "design/blueprint.yaml").exists()
    assert (root / "corpus/records.jsonl").exists()
    assert (root / "benchmarks/tasks.jsonl").exists()
    assert (root / "reports/project-audit.md").exists()
    assert audit.status == "ready"
    assert not audit.blockers


def test_audit_rejects_formally_valid_but_weak_project(tmp_path: Path) -> None:
    project_path, _ = generate_project(_blueprint(tmp_path))
    cases = project_path.parent / "benchmarks/tasks.jsonl"
    cases.write_text(
        '{"case_id":"only-normal","kind":"normal","query":"contract",'
        '"expected_domains":["legal"],"expected_facts":{}}\n',
        encoding="utf-8",
    )
    audit = audit_project(project_path)
    assert audit.status == "needs_work"
    assert "No attack benchmark cases are defined" in audit.blockers
    assert "No abstention case is defined" in audit.blockers
    assert "No multi-agent collaboration case is defined" in audit.blockers


def test_blueprint_round_trip(tmp_path: Path) -> None:
    blueprint = _blueprint(tmp_path)
    path = blueprint.write(tmp_path / "blueprint.yaml")
    loaded = WizardBlueprint.load(path)
    assert loaded == blueprint
