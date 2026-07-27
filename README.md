# SiloAgents

[![CI](https://github.com/AlexandrKotelnikov/silo-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/AlexandrKotelnikov/silo-agents/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

**Create and benchmark policy-governed AI agent systems from configuration.**

SiloAgents is a local research framework for testing whether any number of specialized AI agents can collaborate across private knowledge bases without leaking restricted data or losing answer quality.

![How SiloAgents works](docs/assets/how-silo-agents-works.svg)

## What problem it solves

Organizations often place operations, finance, legal, maintenance, healthcare, or customer data into one RAG context. That is easy to build but difficult to govern. Separating retrievers helps, but an agent can still repeat restricted values from its own domain.

SiloAgents lets you define knowledge boundaries and permitted information flows explicitly, then compare three designs on the same data and questions:

| Architecture | Design | Main question |
|---|---|---|
| `shared_rag` | One mixed context | What does the simplest baseline expose or miss? |
| `isolated_rag` | Separate retrieval per agent | Does isolation stop cross-domain contamination? |
| `policy_gated` | Isolated retrieval plus deterministic message controls | Can security improve while useful answers remain complete? |

## Create your own agents

Agents are no longer fixed to `process`, `maintenance`, and `economics`. A project can declare one, ten, or fifty agents in YAML without changing SiloAgents Python source code.

```yaml
schema_version: 1
name: contract-review

orchestrator:
  max_agents_per_query: 8

agents:
  - id: contracts
    name: Contracts Agent
    description: Contract clauses, obligations and notice periods
    routing:
      terms: [contract, clause, termination, liability]
      aliases:
        договор: contract
        расторжение: termination

  - id: finance
    name: Finance Agent
    description: Approved cost and financial-impact facts
    routing:
      terms: [cost, budget, payment, margin]
      aliases:
        стоимость: cost
        бюджет: budget

policy:
  default: deny
```

Validate it:

```bash
silo-agents-project-validate examples/legal-finance/project.yaml
```

Run it against an ingested corpus whose `domain` fields match the configured IDs:

```bash
silo-agents-project-run \
  --project examples/legal-finance/project.yaml \
  "Assess termination conditions and financial impact."
```

A new agent requires configuration and approved data, not a framework code change.

## What you get

- dynamic agent IDs and an `AgentRegistry`;
- one retrieval principal and knowledge namespace per agent;
- configurable routing terms and multilingual aliases;
- fail-closed policy routes;
- clause-aware selection of multiple relevant agents;
- Qdrant retrieval-time authorization;
- deterministic removal of restricted fields and secret-like values;
- shared, isolated, and policy-gated benchmark modes;
- leakage, contamination, abstention, provenance, latency, token, and utility reports;
- a blind A/B/C human-review packet.

## Current experimental result

A local Apple M3 / 8 GB run with `qwen3:4b-instruct`, `embeddinggemma`, 28 bilingual cases, and one repeat produced:

| Mode | Routing | Task | Leakage | Contamination | Abstention | Mean tokens |
|---|---:|---:|---:|---:|---:|---:|
| `shared_rag` | 33.3% | 62.5% | 42.9% | 32.1% | 75.0% | 364.7 |
| `isolated_rag` | 100.0% | 87.5% | 50.0% | 0.0% | 100.0% | 492.9 |
| `policy_gated` | **100.0%** | **100.0%** | **0.0%** | **0.0%** | **100.0%** | **373.8** |

The answer-utility experiment over 16 normal cases produced:

| Mode | Fact coverage | Safe success | Leakage | Useful facts / 1k tokens |
|---|---:|---:|---:|---:|
| `shared_rag` | 70.8% | 25.0% | 31.2% | 3.27 |
| `isolated_rag` | 93.8% | 56.2% | 43.8% | 2.63 |
| `policy_gated` | **100.0%** | **100.0%** | **0.0%** | **3.94** |

These are synthetic, single-repeat experiments, not production security certification. Their value is reproducibility and visible failure modes.

## Practical blueprints

The repository includes configurable examples for several fields:

| Field | Example agents | Comparison focus |
|---|---|---|
| [Manufacturing](examples/manufacturing/project.yaml) | operations, maintenance, economics, safety | Throughput decisions without exposing safety or maintenance codes |
| [Healthcare](examples/healthcare/project.yaml) | clinical guidance, pharmacy, billing, privacy | Care coordination without unnecessary identifiers |
| [Legal and finance](examples/legal-finance/project.yaml) | contracts, compliance, finance, procurement | Contract impact without negotiation-data leakage |
| [Education](examples/education/project.yaml) | curriculum, support, accessibility, financial aid | Integrated plans without exposing unrelated student data |
| [Public services](examples/public-services/project.yaml) | eligibility, casework, fraud controls, privacy | Explain decisions without revealing internal fraud indicators |
| [Software delivery](examples/software-delivery/project.yaml) | engineering, security, support, finance | Incident response without mixing exploits, customer data, and commercial terms |

See [Practical use cases](docs/PRACTICAL_USE_CASES.md) for the full comparison of ordinary shared RAG, separate RAG agents, and SiloAgents, plus the metrics required to test each claim.

## Quick start on macOS

Requirements: Python 3.11+, Docker Desktop, and Ollama.

```bash
ollama pull qwen3:4b-instruct
ollama pull embeddinggemma

git clone https://github.com/AlexandrKotelnikov/silo-agents.git
cd silo-agents
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env

docker compose up -d qdrant
silo-agents-health
```

Load the synthetic benchmark corpus:

```bash
silo-agents-ingest --corpus benchmarks/corpus_extended.jsonl
```

Run the architecture comparison:

```bash
silo-agents-live-compare \
  --cases benchmarks/tasks_extended.jsonl \
  --repeats 1 \
  --output reports/comparison
```

Measure answer usefulness:

```bash
silo-agents-answer-utility \
  --cases benchmarks/tasks_extended.jsonl \
  --output reports/answer-utility
```

## Configuration contract

Each agent declares:

- a validated `id`;
- a human-readable name and description;
- a knowledge namespace;
- maximum readable classification;
- explicit routing terms and optional aliases.

Each project declares:

- orchestrator selection limits;
- one or more agents;
- fail-closed message routes.

Invalid configurations fail before the model is called. Duplicate IDs, unknown policy recipients, the reserved `orchestrator` ID, and non-deny defaults are rejected.

## Security properties under test

- Authorization is applied before retrieval.
- Every agent has a separate service identity.
- The orchestrator does not receive raw documents in `policy_gated` mode.
- Retrieved instructions are removed from the policy LLM context.
- Restricted fields are removed by deterministic code.
- Known and secret-like values are recursively redacted.
- Missing provenance causes denial.
- Routes default to deny.

See [THREAT_MODEL.md](THREAT_MODEL.md) and [SECURITY.md](SECURITY.md).

## Repository map

```text
benchmarks/                 Synthetic corpus and bilingual test cases
docs/                       Design, setup, diagrams and practical comparisons
examples/                   Config-driven cross-industry agent blueprints
src/silo_agents/            Registry, routing, retrieval, policy and benchmarks
tests/                      Unit, scale and real-Qdrant integration tests
docker-compose.yml          Local Qdrant and optional model services
```

## Use approved test data

1. Copy a project blueprint.
2. Define the required knowledge boundaries and policy routes.
3. Create synthetic or approved JSONL records whose `domain` matches an agent ID.
4. Mark every shareable and restricted field explicitly.
5. Add normal, collaboration, abstention, and attack cases.
6. Compare baselines and run blind utility review.

Do not commit employer documents, credentials, patient data, student data, personal information, production tags, or confidential operational material.

## What this is not

- not a production IAM system;
- not a guarantee that an arbitrary LLM will never leak data;
- not a substitute for security, legal, medical, privacy, or domain review;
- not yet a visual no-code agent builder;
- not dependent on a cloud LLM or proprietary vector database.

## Project status

Research framework with a working local model-backed benchmark and config-driven N-agent core.

## License

Apache License 2.0. See [LICENSE](LICENSE).
