import json
from pathlib import Path

import httpx

from silo_agents.config import LiveSettings, load_dotenv
from silo_agents.embeddings import OllamaEmbedder
from silo_agents.live import LiveBenchmarkReport, render_live_markdown, run_health


def test_load_dotenv_preserves_existing_environment(
    tmp_path: Path, monkeypatch: object
) -> None:
    env = tmp_path / ".env"
    env.write_text("LLM_MODEL=file-model\nQDRANT_URL=http://qdrant\n", encoding="utf-8")
    monkeypatch.setenv("LLM_MODEL", "existing-model")  # type: ignore[attr-defined]
    load_dotenv(env)
    assert LiveSettings.from_env(env).llm_model == "existing-model"


def test_ollama_embedder_discovers_dimensions() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embed"
        assert json.loads(request.content)["model"] == "embeddinggemma"
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3]]})

    embedder = OllamaEmbedder(
        "http://ollama", "embeddinggemma", transport=httpx.MockTransport(handler)
    )
    assert embedder.embed("text") == [0.1, 0.2, 0.3]
    assert embedder.dimensions == 3


def test_live_markdown_contains_metrics() -> None:
    report = LiveBenchmarkReport(
        model="qwen",
        embedding_model="embed",
        repeats=1,
        case_count=1,
        results=[],
        routing_accuracy=1.0,
        task_accuracy=0.5,
        leakage_rate=0.0,
        abstention_accuracy=1.0,
        provenance_coverage=1.0,
        mean_total_tokens=12.0,
        mean_latency_ms=345.0,
    )
    text = render_live_markdown(report)
    assert "100.0%" in text
    assert "345 ms" in text


def test_health_reports_backend_failures(monkeypatch: object) -> None:
    settings = LiveSettings(
        qdrant_url="http://qdrant",
        qdrant_collection="records",
        qdrant_api_key=None,
        llm_base_url="http://ollama/v1",
        llm_model="qwen3:4b-instruct",
        llm_api_key="ollama",
        embedding_base_url="http://ollama",
        embedding_model="embeddinggemma",
    )

    def fail(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "get", fail)  # type: ignore[attr-defined]
    monkeypatch.setattr(httpx, "post", fail)  # type: ignore[attr-defined]
    checks = run_health(settings)
    assert any(not check.ok for check in checks)
