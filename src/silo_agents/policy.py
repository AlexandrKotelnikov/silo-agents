from __future__ import annotations

from copy import deepcopy

from .models import AgentMessage, Classification, Domain, PolicyDecision


class PolicyGateway:
    """Fail-closed policy enforcement for every inter-agent message."""

    def __init__(self, allowed_routes: dict[Domain, set[Domain]]) -> None:
        self.allowed_routes = allowed_routes

    def evaluate(self, message: AgentMessage) -> PolicyDecision:
        if message.recipient not in self.allowed_routes.get(message.sender, set()):
            return PolicyDecision(allowed=False, reason="route_not_allowed")
        if message.recipient not in message.share_scope:
            return PolicyDecision(allowed=False, reason="recipient_outside_share_scope")
        if message.classification == Classification.RESTRICTED:
            return PolicyDecision(allowed=False, reason="restricted_requires_human_approval")
        if not message.evidence:
            return PolicyDecision(allowed=False, reason="missing_provenance")

        sanitized = deepcopy(message)
        for field_name in message.restricted_fields:
            sanitized.conclusion.pop(field_name, None)
        return PolicyDecision(
            allowed=True,
            reason="allowed_after_sanitization",
            sanitized_message=sanitized,
        )
