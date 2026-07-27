#!/usr/bin/env bash
set -euo pipefail

WITH_MODELS=false
START_QDRANT=true
for arg in "$@"; do
  case "$arg" in
    --with-models) WITH_MODELS=true ;;
    --no-qdrant) START_QDRANT=false ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

command -v python3 >/dev/null || { echo "Python 3 is required" >&2; exit 1; }
python3 - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11+ is required")
PY

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

if [[ "$START_QDRANT" == true ]]; then
  command -v docker >/dev/null || { echo "Docker is required; rerun with --no-qdrant to skip" >&2; exit 1; }
  docker compose up -d qdrant
fi

if [[ "$WITH_MODELS" == true ]]; then
  command -v ollama >/dev/null || { echo "Ollama is required for --with-models" >&2; exit 1; }
  ollama pull qwen3:4b-instruct
  ollama pull embeddinggemma
fi

cat <<'EOF'

SiloAgents setup complete.

Activate the environment:
  source .venv/bin/activate

Verify services:
  silo-agents-health

Try the included experiment:
  silo-agents validate --project examples/legal-finance/project.yaml
  silo-agents ingest --project examples/legal-finance/project.yaml
  silo-agents run --project examples/legal-finance/project.yaml \
    "Assess contract termination and financial impact."
EOF
