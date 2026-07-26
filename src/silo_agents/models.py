from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Domain(StrEnum):
    PROCESS = "process"
    MAINTENANCE = "maintenance"
    ECONOMICS = "economics"
    ORCHESTRATOR = "orchestrator"


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
    sender: Domain
    recipient: Domain
    purpose: str
    classification: Classification = Classification.INTERNAL
    share_scope: set[Domain] = Field(default_factory=set)
    conclusion: dict[str, Any] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    restricted_fields: set[str] = Field(default_factory=set)
    sensitive_values: set[str] = Field(default_factory=set, exclude=True, repr=False)
    missing_information: list[str] = Field(default_factory=list)
    telemetry: dict[str, int | float | str] = Field(default_factory=dict)


class RetrievalRecord(BaseModel):
    record_id: str
    domain: Domain
    text: str
    classification: Classification = Classification.INTERNAL
    metadata: dict[str, Any] = Field(default_factory=dict)


class PolicyDecision(BaseModel):
    allowed: bool
    reason: str
    sanitized_message: AgentMessage | None = None
