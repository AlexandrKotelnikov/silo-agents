from silo_agents.final_answer import synthesize_final_answer
from silo_agents.models import AgentId, AgentMessage, Classification, Evidence


def _message(sender: str, conclusion: dict[str, object]) -> AgentMessage:
    return AgentMessage(
        message_id=f"m-{sender}",
        task_id="task",
        sender=AgentId(sender),
        recipient=AgentId.ORCHESTRATOR,
        purpose="answer",
        classification=Classification.INTERNAL,
        share_scope={AgentId.ORCHESTRATOR},
        conclusion=conclusion,
        evidence=[Evidence(source_id=f"source-{sender}", fragment_id="0")],
    )


def test_synthesizes_one_answer_with_sources() -> None:
    answer = synthesize_final_answer(
        "Assess the decision",
        (
            _message("legal", {"summary": "Thirty days notice.", "notice_days": 30}),
            _message("finance", {"summary": "Cost is approved.", "cost_eur": 120000}),
        ),
    )

    assert answer.status == "grounded"
    assert answer.facts == {"notice_days": 30, "cost_eur": 120000}
    assert answer.sources == ["source-finance", "source-legal"]
    assert "legal: Thirty days notice." in answer.answer
    assert "finance: Cost is approved." in answer.answer


def test_surfaces_conflicts_instead_of_choosing_silently() -> None:
    answer = synthesize_final_answer(
        "What is the limit?",
        (
            _message("agent-a", {"limit": 10}),
            _message("agent-b", {"limit": 12}),
        ),
    )

    assert answer.status == "needs_review"
    assert answer.facts == {}
    assert answer.conflicts == {"limit": [10, 12]}


def test_redacts_unknown_secret_like_values_in_final_pass() -> None:
    answer = synthesize_final_answer(
        "Show approved information",
        (_message("legal", {"summary": "Internal code OMEGA-6620", "status_code": "OMEGA-6620"}),),
    )

    assert "OMEGA-6620" not in answer.answer
    assert answer.facts["status_code"] == "[REDACTED]"


def test_abstention_is_human_readable() -> None:
    answer = synthesize_final_answer("Weather?", ())
    assert answer.status == "abstained"
    assert "No configured agent" in answer.answer
