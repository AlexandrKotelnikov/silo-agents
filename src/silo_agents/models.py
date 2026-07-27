from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

_AGENT_ID = re.compile(r"[a-z][a-z0-9_-]{0,63}")


class AgentId(StrEnum):
    """Validated agent identifier with backwards-compatible built-in examples.

    The named members preserve the original public API, while ``_missing_`` allows
    projects to define arbitrary agents such as ``legal`` or ``clinical_pharmacy``
    without changing the framework source code.
    """

    PROCESS = "process"
    MAINTENANCE = "maintenance"
    ECONOMICS = "economics"
    ORCHESTRATOR = "orchestrator"

    @classmethod
    def _missing_(cls, value: object) -> AgentId | None:
        if not isinstance(value, str) or _AGENT_ID.fullmatch(value) is None:
            return None
        member = str.__new__(cls, value)
        member._name_ = f"DYNAMIC_{value.upper().replace('-', '_')}"
        member._value_ = value
        return member


# Backwards-compatible name used by the original three-agent experiment.
Domain = AgentId


class Classification(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"


class Evidence(BaseModel):
    source_id: str
    fragment_id: str
    quote_allowed: bool = False


class AgentMessage(BaseModel):
    message_id: str
    task_id: str
    sender: AgentId
    recipient: AgentId
    purpose: str
    classification: Classification = Classification.INTERNAL
    share_scope: set[AgentId] = Field(default_factory=set)
    conclusion: dict[str, Any] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    restricted_fields: set[str] = Field(default_factory=set)
    sensitive_values: set[str] = Field(default_factory=set, exclude=True, repr=False)
    missing_information: list[str] = Field(default_factory=list)
    telemetry: dict[str, int | float | str] = Field(default_factory=dict)


class RetrievalRecord(BaseModel):
    record_id: str
    domain: AgentId
    text: str
    classification: Classification = Classification.INTERNAL
    metadata: dict[str, Any] = Field(default_factory=dict)


class PolicyDecision(BaseModel):
    allowed: bool
    reason: str
    sanitized_message: AgentMessage | None = None
