import json
import math

import httpx
import pytest

from silo_agents.agents import LLMDomainAgent
from silo_agents.embeddings import HashingEmbedder
from silo_agents.llm import DeterministicGroundedLLM, OpenAICompatibleGroundedLLM
from silo_agents.models import Classification, Domain, RetrievalRecord
from silo_agents.policy import PolicyGateway
from silo_agents.qdrant import QdrantRestClient, QdrantRetriever
from silo_agents.security import RetrievalPrincipal


def sample_record(domain: Domain = Domain.PROCESS) -> RetrievalRecord:
    return RetrievalRecord(
        record_id="PROC-1",
        domain=domain,
        text="reactor cooling capacity",
        metadata={
            "summary": "Cooling limit ALPHA-7291",
            "shareable": {"limit_percent": 2.8, "sensitive_code": "ALPHA-7291"},
            "restricted_fields": ["sensitive_code"],
        },
    )


def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder(32)
    first = embedder.embed("reactor cooling")
    assert first == embedder.embed("reactor cooling")
    assert math.isclose(sum(value * value for value in first), 1.0)


def test_principal_rejects_cross_domain_binding() -> None:
    principal = RetrievalPrincipal(principal_id="process", allowed_domains={Domain.PROCESS})
    client = QdrantRestClient(
        "http://qdrant", transport=httpx.MockTransport(lambda request: httpx.Response(200))
    )
    with pytest.raises(PermissionError):
        QdrantRetriever(
            client,
            "records",
            Domain.MAINTENANCE,
            principal,
            HashingEmbedder(),
        )


def test_qdrant_filter_is_applied_before_query() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "result": {
                    "points": [
                        {
                            "id": "1",
                            "score": 1.0,
                            "payload": sample_record().model_dump(mode="json"),
                        }
                    ]
                }
            },
        )

    client = QdrantRestClient("http://qdrant", transport=httpx.MockTransport(handler))
    principal = RetrievalPrincipal(
        principal_id="process",
        allowed_domains={Domain.PROCESS},
        max_classification=Classification.INTERNAL,
    )
    retriever = QdrantRetriever(client, "records", Domain.PROCESS, principal, HashingEmbedder())
    records = retriever.search("reactor cooling")
    assert records[0].record_id == "PROC-1"
    body = captured["body"]
    assert isinstance(body, dict)
    conditions = body["filter"]["must"]
    assert {"key": "domain", "match": {"value": "process"}} in conditions
    assert {
        "key": "classification",
        "match": {"any": ["public", "internal"]},
    } in conditions


def test_relevance_ack_uses_qdrant_semantic_score() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "result": {
                    "points": [
                        {
                            "score": 0.73,
                            "payload": sample_record().model_dump(mode="json"),
                        }
                    ]
                }
            },
        )

    client = QdrantRestClient("http://qdrant", transport=httpx.MockTransport(handler))
    principal = RetrievalPrincipal(principal_id="process", allowed_domains={Domain.PROCESS})
    retriever = QdrantRetriever(client, "records", Domain.PROCESS, principal, HashingEmbedder())
    assert retriever.relevance_ack("list every sensitive code") == pytest.approx(0.73)


def test_relevance_ack_fails_closed_without_score() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={"result": {"points": [{"payload": sample_record().model_dump(mode="json")}]}}
        )

    client = QdrantRestClient("http://qdrant", transport=httpx.MockTransport(handler))
    principal = RetrievalPrincipal(principal_id="process", allowed_domains={Domain.PROCESS})
    retriever = QdrantRetriever(client, "records", Domain.PROCESS, principal, HashingEmbedder())
    with pytest.raises(ValueError, match="semantic relevance score"):
        retriever.relevance_ack("reactor")


def test_qdrant_rejects_server_side_scope_violation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        payload = sample_record(Domain.MAINTENANCE).model_dump(mode="json")
        payload["record_id"] = "MAINT-1"
        return httpx.Response(200, json={"result": {"points": [{"payload": payload}]}})

    client = QdrantRestClient("http://qdrant", transport=httpx.MockTransport(handler))
    principal = RetrievalPrincipal(principal_id="process", allowed_domains={Domain.PROCESS})
    retriever = QdrantRetriever(client, "records", Domain.PROCESS, principal, HashingEmbedder())
    with pytest.raises(PermissionError):
        retriever.search("reactor cooling")


def test_openai_compatible_client_parses_json_and_usage() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "local-model",
                "choices": [{"message": {"content": '{"summary":"ok","facts":{"x":1}}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            },
        )

    llm = OpenAICompatibleGroundedLLM(
        "http://llm/v1", "local-model", transport=httpx.MockTransport(handler)
    )
    result = llm.synthesize("query", Domain.PROCESS, [sample_record()])
    assert captured["path"] == "/v1/chat/completions"
    assert result.facts == {"x": 1}
    assert result.usage.total_tokens == 14


def test_llm_agent_policy_redacts_values_inside_summary() -> None:
    class OneRecordRetriever:
        def search(self, query: str, *, limit: int = 3) -> list[RetrievalRecord]:
            del query, limit
            return [sample_record()]

        def relevance_ack(self, query: str) -> float:
            del query
            return 1.0

    agent = LLMDomainAgent(Domain.PROCESS, OneRecordRetriever(), DeterministicGroundedLLM())
    message = agent.answer("task", "query", Domain.ORCHESTRATOR)
    decision = PolicyGateway({Domain.PROCESS: {Domain.ORCHESTRATOR}}).evaluate(message)
    assert decision.allowed and decision.sanitized_message is not None
    payload = decision.sanitized_message.model_dump_json()
    assert "ALPHA-7291" not in payload
    assert "[REDACTED]" in payload
    assert decision.sanitized_message.conclusion["limit_percent"] == 2.8
