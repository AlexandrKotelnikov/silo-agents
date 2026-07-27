# SiloAgents

[![CI](https://github.com/AlexandrKotelnikov/silo-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/AlexandrKotelnikov/silo-agents/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

**Create and benchmark policy-governed AI agent systems from configuration.**

SiloAgents is a local research framework for testing whether any number of specialized AI agents can collaborate across private knowledge bases without leaking restricted data or losing answer quality.

![How SiloAgents works](docs/assets/how-silo-agents-works.svg)

## What problem it solves

Organizations often place operations, finance, legal, maintenance, healthcare, or customer data into one RAG context. That is easy to build but difficult to govern. Separating retrievers helps, but an agent can still repeat restricted values from its own knowledge base.

SiloAgents lets you define knowledge boundaries and permitted information flows explicitly, then compare three designs on the same data and questions:

| Architecture | Design | Main question |
|---|---|---|
| `shared_rag` | One mixed context | What does the simplest baseline expose or miss? |
| `isolated_rag` | Separate retrieval per agent | Does isolation stop cross-domain contamination? |
| `policy_gated` | Isolated retrieval plus deterministic message controls | Can security improve while useful answers remain complete? |

## Start a project without writing framework code

```bash
silo-agents init my-project
cd my-project
cp .env.example .env

silo-agents agent add legal \
  --term contract \
  --term termination \
  --alias договор=contract

silo-agents agent add finance \
  --term cost \
  --term budget \
  --alias стоимость=cost

silo-agents validate
silo-agents ingest
silo-agents benchmark
silo-agents utility
silo-agents run "Assess contract termination and financial impact."
```

The generated workspace contains:

```text
my-project/
├── silo-agents.yaml
├── agents/
├── corpus/records.jsonl
├── benchmarks/tasks.jsonl
├── reports/
├── .env.example
└── README.md
```

A new agent requires configuration and approved data, not a change to SiloAgents Python source.

## Agent configuration

```yaml
schema_version: 1
name: contract-review

paths:
  corpus: corpus/records.jsonl
  cases: benchmarks/tasks.jsonl
  reports: reports

orchestrator:
  max_agents_per_query: 8

agents:
  - id: contract-reviewer
    name: Contracts Agent
    knowledge_namespace: approved-contracts
    routing:
      terms: [contract, clause, termination, liability]
      aliases:
        договор: contract
        расторжение: termination

  - id: finance
    name: Finance Agent
    routing:
      terms: [cost, budget, payment, margin]
      aliases:
        стоимость: cost
        бюджет: budget

policy:
  default: deny
```

Agent identity and data namespace are separate. A document can belong to `approved-contracts`, while the answering agent is `contract-reviewer`.

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
- a blind A/B/C human-review packet;
- project scaffolding, corpus validation, ingestion, benchmark and utility commands.

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

## Runnable cross-industry examples

| Field | Example agents | Comparison focus |
|---|---|---|
| [Manufacturing](examples/manufacturing/project.yaml) | operations, maintenance, economics, safety | Throughput decisions without exposing safety or maintenance codes |
| [Healthcare](examples/healthcare/project.yaml) | clinical guidance, pharmacy, billing, privacy | Care coordination without unnecessary identifiers |
| [Legal and finance](examples/legal-finance/project.yaml) | contracts, compliance, finance, procurement | Contract impact without negotiation-data leakage |
| [Education](examples/education/project.yaml) | curriculum, support, accessibility, financial aid | Integrated plans without exposing unrelated student data |
| [Public services](examples/public-services/project.yaml) | eligibility, casework, fraud controls, privacy | Explain decisions without revealing internal fraud indicators |
| [Software delivery](examples/software-delivery/project.yaml) | engineering, security, support, finance | Incident response without mixing exploits, customer data, and commercial terms |

The legal-finance example includes a complete corpus and benchmark:

```bash
silo-agents validate --project examples/legal-finance/project.yaml
silo-agents ingest --project examples/legal-finance/project.yaml
silo-agents benchmark --project examples/legal-finance/project.yaml
silo-agents utility --project examples/legal-finance/project.yaml
```

See [Practical use cases](docs/PRACTICAL_USE_CASES.md) for comparison with ordinary shared RAG and separate RAG agents.

## Install on macOS

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

## Validation rules

Before ingestion, `silo-agents validate` checks:

- the project file can be parsed;
- agent IDs and knowledge namespaces are unique;
- corpus namespaces are declared by agents;
- benchmark cases reference configured agent IDs;
- corpus and task files exist;
- fail-closed policy rules are valid;
- routing vocabulary is present or reported as a warning.

Invalid projects fail before an LLM call.

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
benchmarks/                 Original synthetic benchmark
examples/                   Config-driven cross-industry projects
src/silo_agents/            Registry, workspace, routing, retrieval and policy
tests/                      Unit, scale, workspace and Qdrant integration tests
docs/                       Design, setup, diagrams and practical comparisons
docker-compose.yml          Local Qdrant and optional model services
```

## Use approved test data

Do not commit employer documents, credentials, patient data, student data, personal information, production tags, or confidential operational material. Use synthetic or explicitly approved test data and mark every restricted field.

## What this is not

- not a production IAM system;
- not a guarantee that an arbitrary LLM will never leak data;
- not a substitute for security, legal, medical, privacy, or domain review;
- not yet a visual no-code agent builder;
- not dependent on a cloud LLM or proprietary vector database.

## Project status

Research framework with a working local model-backed benchmark, config-driven N-agent core, and complete project workflow.

## License

Apache License 2.0. See [LICENSE](LICENSE).
