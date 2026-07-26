# Real backend integration

This milestone connects the research harness to a real vector database and one shared language-model endpoint while preserving agent-level isolation.

## Security boundary

Each domain agent receives a distinct `RetrievalPrincipal`. A Qdrant query is constructed with mandatory payload filters before the request leaves the application:

- exact `domain` match;
- maximum allowed `classification`;
- optional allow-list of `record_id` values.

Every returned payload is validated again. A missing authorization payload, a cross-domain record, or a record outside the principal scope raises `PermissionError`. The system fails closed rather than filtering an unauthorized result after use.

## Start Qdrant

```bash
cp .env.example .env
docker compose up -d qdrant
```

Qdrant will be available at `http://localhost:6333`.

## Start an optional shared local LLM

The compose profile uses the official OpenAI-compatible vLLM server and requires an NVIDIA GPU:

```bash
docker compose --profile gpu up -d vllm
```

A different OpenAI-compatible endpoint can be supplied through `LLM_BASE_URL`, `LLM_MODEL`, and `LLM_API_KEY`.

## Python assembly

```python
from silo_agents import (
    HashingEmbedder,
    OpenAICompatibleGroundedLLM,
    QdrantRestClient,
    build_qdrant_llm_system,
    ingest_qdrant,
    load_records,
)

records = load_records("benchmarks/corpus.jsonl")
embedder = HashingEmbedder(dimensions=64)
qdrant = QdrantRestClient("http://localhost:6333")
ingest_qdrant(qdrant, "silo_records", records, embedder)

llm = OpenAICompatibleGroundedLLM(
    "http://localhost:8000/v1",
    "Qwen/Qwen3-0.6B",
)
system = build_qdrant_llm_system(qdrant, "silo_records", embedder, llm)
result = system.run_many(
    "Assess production increase limit, maintenance risk and economic effect margin."
)

for message in result.delivered_messages:
    print(message.model_dump_json(indent=2))
```

All three agents share the same `OpenAICompatibleGroundedLLM` instance. They do not share retrievers, principals, or retrieved context.

## Test strategy

The CI suite uses `httpx.MockTransport`; it does not require network access, Qdrant, a GPU, or an API key. Tests assert the exact Qdrant filter body, reject malicious cross-domain responses, validate OpenAI-compatible response parsing, and verify deterministic redaction of sensitive values even when the LLM repeats them inside a summary.

## Limitations

- `HashingEmbedder` is deliberately simple and intended for reproducible experiments, not production semantic search.
- Docker images use moving tags for the research prototype; production deployments should pin reviewed image digests.
- Qdrant payload filters are one security layer, not a replacement for network isolation, service authentication, audit logging, or independent security review.
