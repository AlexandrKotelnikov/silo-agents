# Live Ollama + Qdrant run on Apple Silicon

This profile is designed for an Apple M3 Mac with 8 GB unified memory.

## Update and install

```bash
git pull
source .venv/bin/activate
pip install -e ".[dev]"
```

If `git pull` reports a local change in `docker-compose.yml`, run:

```bash
git restore docker-compose.yml
git pull
```

The repository version already binds Qdrant only to `127.0.0.1`.

## Start and verify services

```bash
docker compose up -d qdrant
silo-agents-health
```

## Load the synthetic corpus

```bash
silo-agents-ingest --corpus benchmarks/corpus.jsonl
```

## Run one live query

```bash
silo-agents-live-run "What limits reactor throughput and cooling capacity?"
```

## Run the model-backed benchmark

Start with one repeat on an 8 GB Mac:

```bash
silo-agents-live-benchmark \
  --cases benchmarks/tasks.jsonl \
  --repeats 1 \
  --output reports/live
```

After confirming stability, use three repeats:

```bash
silo-agents-live-benchmark --repeats 3 --output reports/live-3x
```

This live report evaluates the target `policy_gated` architecture. The deterministic `silo-agents-benchmark` command remains the controlled three-mode comparison.

When changing embedding models, recreate the Qdrant volume before ingesting again:

```bash
docker compose down -v
docker compose up -d qdrant
silo-agents-ingest
```

Do not add confidential employer documents, production credentials, personal data, or real operational secrets to the benchmark corpus.
