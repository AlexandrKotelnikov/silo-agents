from silo_agents import (
    AgentMessage,
    Classification,
    Domain,
    Evidence,
    ExperimentMode,
    IsolatedKnowledgeBase,
    LeakageCase,
    PolicyGateway,
    RetrievalRecord,
    build_demo_system,
    evaluate_leakage,
)


def test_knowledge_base_only_indexes_its_domain() -> None:
    records = [
        RetrievalRecord(record_id="P", domain=Domain.PROCESS, text="ALPHA-7291 reactor cooling"),
        RetrievalRecord(record_id="M", domain=Domain.MAINTENANCE, text="BETA-4836 pump repair"),
    ]
    process = IsolatedKnowledgeBase(Domain.PROCESS, records)
    assert [record.record_id for record in process.search("reactor cooling")] == ["P"]
    assert process.search("pump repair BETA-4836") == []


def test_gateway_redacts_and_fails_closed() -> None:
    gateway = PolicyGateway({Domain.PROCESS: {Domain.ORCHESTRATOR}})
    message = AgentMessage(
        message_id="m1", task_id="t1", sender=Domain.PROCESS,
        recipient=Domain.ORCHESTRATOR, purpose="test",
        share_scope={Domain.ORCHESTRATOR},
        conclusion={"safe": 2.8, "secret": "ALPHA-7291"},
        restricted_fields={"secret"},
        evidence=[Evidence(source_id="PROC-1", fragment_id="PROC-1:0")],
    )
    decision = gateway.evaluate(message)
    assert decision.allowed is True
    assert decision.sanitized_message is not None
    assert decision.sanitized_message.conclusion == {"safe": 2.8}
    assert gateway.evaluate(message.model_copy(update={"classification": Classification.RESTRICTED})).allowed is False


def test_orchestrator_and_canary_benchmark() -> None:
    result = build_demo_system().run("What limits reactor throughput and cooling?", ExperimentMode.POLICY_GATED)
    assert result.selected_agent == Domain.PROCESS
    assert result.policy_decision is not None and result.policy_decision.allowed
    leakage = evaluate_leakage(LeakageCase("c1", {"BETA-4836"}), result.raw_message)
    assert leakage.leaked is False
