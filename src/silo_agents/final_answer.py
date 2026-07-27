from __future__ import annotations

from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field

from .models import AgentId, AgentMessage
from .policy import _redact_values


class FinalAnswer(BaseModel):
    """One audited answer synthesized only from policy-approved messages."""

    status: str
    answer: str
    facts: dict[str, Any] = Field(default_factory=dict)
    contributing_agents: list[AgentId] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    conflicts: dict[str, list[Any]] = Field(default_factory=dict)
    missing_information: list[str] = Field(default_factory=list)


def synthesize_final_answer(query: str, messages: tuple[AgentMessage, ...]) -> FinalAnswer:
    """Create a stable final response without exposing raw documents to another LLM.

    Domain summaries are preserved as attributed statements. Structured facts are
    merged deterministically; conflicting values are surfaced instead of silently
    choosing one. The function only accepts already delivered (policy-approved)
    messages and applies a final secret-like redaction pass.
    """
    if not messages:
        return FinalAnswer(
            status="abstained",
            answer=(
                "No configured agent found enough approved evidence to answer this request. "
                "Add relevant records or refine the routing vocabulary."
            ),
        )

    values: dict[str, list[Any]] = defaultdict(list)
    sections: list[str] = []
    missing: list[str] = []
    sources: set[str] = set()
    agents: list[AgentId] = []

    for message in messages:
        agents.append(message.sender)
        conclusion = message.conclusion
        summary = conclusion.get("summary")
        if isinstance(summary, str) and summary.strip():
            sections.append(f"{message.sender.value}: {summary.strip()}")
        for key, value in conclusion.items():
            if key not in {"status", "summary"} and value not in values[key]:
                values[key].append(value)
        missing.extend(item for item in message.missing_information if item not in missing)
        sources.update(evidence.source_id for evidence in message.evidence)

    facts = {key: candidates[0] for key, candidates in values.items() if len(candidates) == 1}
    conflicts = {key: candidates for key, candidates in values.items() if len(candidates) > 1}

    lines = [f"Answer to: {query}"]
    if sections:
        lines.extend(["", "Contributions:", *(f"- {section}" for section in sections)])
    if facts:
        lines.extend(["", "Approved facts:"])
        lines.extend(f"- {key}: {_display(value)}" for key, value in sorted(facts.items()))
    if conflicts:
        lines.extend(["", "Conflicts requiring review:"])
        lines.extend(
            f"- {key}: {', '.join(_display(value) for value in candidates)}"
            for key, candidates in sorted(conflicts.items())
        )
    if missing:
        lines.extend(["", "Missing information:", *(f"- {item}" for item in missing)])

    sanitized_answer = _redact_values("\n".join(lines), set())
    sanitized_facts = _redact_values(facts, set())
    sanitized_conflicts = _redact_values(conflicts, set())
    return FinalAnswer(
        status="needs_review" if conflicts else "grounded",
        answer=str(sanitized_answer),
        facts=sanitized_facts if isinstance(sanitized_facts, dict) else {},
        contributing_agents=agents,
        sources=sorted(sources),
        conflicts=sanitized_conflicts if isinstance(sanitized_conflicts, dict) else {},
        missing_information=missing,
    )


def _display(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return str(value)
    return str(value)
