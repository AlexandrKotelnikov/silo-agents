# Comparative live benchmark

This benchmark runs the same local Ollama and Qdrant workload through three architectures:

- `shared_rag`: one shared retrieval context and one LLM response without a policy gateway;
- `isolated_rag`: domain-specific retrievers and agents without message sanitization;
- `policy_gated`: domain-specific retrievers plus deterministic policy enforcement.

## Prepare the bilingual corpus

The extended corpus adds Russian and English wording, indirect prompt injection text, and additional synthetic canaries. It uses synthetic data only.

```bash
silo-agents-ingest --corpus benchmarks/corpus_extended.jsonl
```

Upsert is idempotent because record IDs map to stable Qdrant point IDs.

## Quick comparison

On an Apple M3 with 8 GB unified memory, start with one repeat:

```bash
silo-agents-live-compare \
  --cases benchmarks/tasks_extended.jsonl \
  --repeats 1 \
  --output reports/comparison-1
```

The command prints progress for every architecture and case. The extended set contains normal single-domain tasks, multi-agent tasks, unrelated requests, direct extraction attempts, role-play attacks, and indirect instructions embedded in retrieved documents.

## Outputs

- `report.md`: aggregate comparison and failed cases;
- `report.json`: complete machine-readable report;
- `results.csv`: one row per architecture, case, and repeat.

Reported metrics include routing accuracy, task accuracy, leakage, cross-domain contamination, abstention, provenance, prompt/completion tokens, median latency, and p95 latency.

## Repeated run

```bash
silo-agents-live-compare \
  --cases benchmarks/tasks_extended.jsonl \
  --repeats 3 \
  --output reports/comparison-3
```

This is intentionally sequential to fit small Apple Silicon machines. A full three-repeat run may take tens of minutes because each multi-domain task invokes the shared local model several times.

## Interpretation

The modes share the same model, embeddings, corpus, and hardware. Differences therefore reflect retrieval and policy architecture rather than a change in the underlying model. The benchmark is a controlled experiment over synthetic data, not a production security certification.
