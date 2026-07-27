# SiloAgents

<p align="center">
  <strong>Build and benchmark policy-governed multi-agent RAG systems from configuration.</strong>
</p>

<p align="center">
  <a href="README_RU.md">Русская версия</a> ·
  <a href="docs/PRACTICAL_USE_CASES.md">Use cases</a> ·
  <a href="THREAT_MODEL.md">Threat model</a> ·
  <a href="SECURITY.md">Security</a>
</p>

<p align="center">
  <a href="https://github.com/AlexandrKotelnikov/silo-agents/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/AlexandrKotelnikov/silo-agents/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue.svg">
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg">
  <img alt="Local first" src="https://img.shields.io/badge/local--first-Qdrant%20%2B%20Ollama-0A7E8C">
</p>

![How SiloAgents works](docs/assets/how-silo-agents-works.svg)

## What is this?

SiloAgents is a local framework for creating any number of specialized AI agents with separate knowledge bases, explicit information-sharing rules, and reproducible security and answer-quality benchmarks.

It lets you answer a practical question before deploying enterprise multi-agent RAG:

> Can several AI agents collaborate on private data without leaking restricted values, mixing unrelated domains, or becoming useless after security controls are applied?

## Why would I use it?

Use SiloAgents when one shared RAG context is too risky and isolated agents alone are not enough.

| Without SiloAgents | With SiloAgents |
|---|---|
| Documents from several functions enter one context | Every agent receives its own retrieval identity and namespace |
| An isolated agent may still repeat secrets from its own data | Restricted fields and secret-like values are removed deterministically |
| Multi-agent output is a collection of unrelated messages | Policy-approved messages are synthesized into one auditable answer |
| Security claims are difficult to verify | Shared, isolated, and policy-gated designs run on the same benchmark |
| Adding agents requires framework code changes | Agents, routing aliases, namespaces, and routes are configured in YAML |

Typical uses include manufacturing, healthcare, legal review, finance, education, public services, software delivery, and any workflow where several specialists must cooperate without unrestricted data sharing.

## Proven result on the included synthetic benchmark

Apple M3 / 8 GB, `qwen3:4b-instruct`, `embeddinggemma`, 28 bilingual cases, one repeat:

| Mode | Routing | Task | Leakage | Contamination | Abstention |
|---|---:|---:|---:|---:|---:|
| `shared_rag` | 33.3% | 62.5% | 42.9% | 32.1% | 75.0% |
| `isolated_rag` | 100.0% | 87.5% | 50.0% | 0.0% | 100.0% |
| `policy_gated` | **100.0%** | **100.0%** | **0.0%** | **0.0%** | **100.0%** |

These are synthetic, single-repeat results, not a production security certification. The repository is designed to make the assumptions, failures, and trade-offs reproducible.

## Install

Requirements: Python 3.11+, Docker, and Ollama.

### Automated setup

```bash
git clone https://github.com/AlexandrKotelnikov/silo-agents.git
cd silo-agents
bash scripts/setup.sh --with-models
source .venv/bin/activate
silo-agents-health
```

The script creates `.venv`, installs the package, creates `.env`, starts Qdrant, and optionally downloads the local models. Run `bash scripts/setup.sh --no-qdrant` when Qdrant is managed separately.

### Manual setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
docker compose up -d qdrant
ollama pull qwen3:4b-instruct
ollama pull embeddinggemma
silo-agents-health
```

## Try the complete example

```bash
silo-agents validate --project examples/legal-finance/project.yaml
silo-agents ingest --project examples/legal-finance/project.yaml
silo-agents benchmark --project examples/legal-finance/project.yaml
silo-agents utility --project examples/legal-finance/project.yaml
silo-agents run --project examples/legal-finance/project.yaml \
  "Assess contract termination and financial impact."
```

The final command returns one synthesized answer, approved facts, contributing agents, sources, conflicts, and missing information. Add `--trace` to inspect the approved internal messages and policy decisions.

## Create your own project

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

Generated structure:

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

A new agent requires configuration and approved data, not a modification to the framework source.

## How it works

```text
User query
  → clause-aware routing
  → N isolated retrieval identities
  → domain agents
  → deny-by-default Policy Gateway
  → deterministic final synthesis
  → one answer with provenance, conflicts, and missing information
```

Agent identity and data storage are separate:

```yaml
agents:
  - id: contract-reviewer
    knowledge_namespace: approved-contracts
    routing:
      terms: [contract, termination, liability]
      aliases:
        договор: contract
policy:
  default: deny
```

Documents use the configured namespace, while policy messages use the agent ID.

## What you get

- arbitrary N-agent projects defined in YAML;
- separate Qdrant retrieval principals and knowledge namespaces;
- multilingual routing terms and aliases;
- fail-closed agent-to-agent routes;
- deterministic removal of restricted fields and secret-like values;
- one final synthesized answer plus optional audit trace;
- shared, isolated, and policy-gated comparisons;
- leakage, contamination, routing, abstention, provenance, latency, token, and utility metrics;
- blind A/B/C human-review packets;
- project scaffolding, validation, ingestion, benchmark, and utility commands.

## Examples by field

| Field | Agents | Main comparison |
|---|---|---|
| [Manufacturing](examples/manufacturing/project.yaml) | operations, maintenance, economics, safety | Throughput decisions without exposing safety and maintenance codes |
| [Healthcare](examples/healthcare/project.yaml) | clinical guidance, pharmacy, billing, privacy | Coordination without unnecessary identifiers |
| [Legal and finance](examples/legal-finance/project.yaml) | contracts, compliance, finance, procurement | Contract impact without negotiation-data leakage |
| [Education](examples/education/project.yaml) | curriculum, support, accessibility, financial aid | Integrated support without unrelated student data |
| [Public services](examples/public-services/project.yaml) | eligibility, casework, fraud controls, privacy | Explain decisions without revealing fraud indicators |
| [Software delivery](examples/software-delivery/project.yaml) | engineering, security, support, finance | Incident response without mixing exploits, customer data, and commercial terms |

See [Practical use cases](docs/PRACTICAL_USE_CASES.md) for detailed comparisons with ordinary shared RAG and separate RAG agents.

## Safety boundaries

SiloAgents is a research framework, not a production IAM system or a guarantee that any model will never leak. Use synthetic or explicitly approved test data. Do not commit employer documents, credentials, patient data, personal information, production tags, or confidential operational material.

See [THREAT_MODEL.md](THREAT_MODEL.md) and [SECURITY.md](SECURITY.md).

## Repository map

```text
src/silo_agents/            Registry, routing, retrieval, policy, synthesis and benchmarks
examples/                   Cross-industry configurable projects
benchmarks/                 Original bilingual synthetic experiment
docs/                       Architecture, use cases and setup guidance
tests/                      Unit, scale, workspace and Qdrant integration tests
scripts/setup.sh             Automated local installation
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
