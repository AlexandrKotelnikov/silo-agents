# Project wizard, audit and doctor

The wizard converts a project idea into a reviewable SiloAgents workspace. It does not certify the resulting system for production use.

## Interactive design

```bash
silo-agents-wizard
```

The wizard asks for:

- the decision or workflow to support;
- the number of specialized agents;
- agent responsibilities and knowledge namespaces;
- at least three routing terms per agent;
- fields that may be shared;
- fields that must remain restricted;
- one representative question per agent.

It creates:

```text
project/
├── silo-agents.yaml
├── design/blueprint.yaml
├── agents/*.yaml
├── corpus/records.jsonl
├── benchmarks/tasks.jsonl
├── reports/project-audit.json
├── reports/project-audit.md
├── .env.example
└── README.md
```

## Reproducible design

Store the design as YAML and generate the same project again:

```bash
silo-agents-wizard \
  --blueprint examples/wizard-blueprint.yaml
```

Use `--save-blueprint` after an interactive session to preserve the answers.

## Readiness audit

```bash
silo-agents-audit \
  --project generated/supplier-risk-review/silo-agents.yaml \
  --output generated/supplier-risk-review/reports
```

CI can fail when the project is not ready:

```bash
silo-agents-audit --project silo-agents.yaml --require-ready
```

The audit scores:

1. configuration;
2. data coverage;
3. routing quality;
4. security-test coverage;
5. operational readiness.

A project is labelled `ready` only when no blocker remains and the overall score is at least 80/100.

Blockers include missing corpus coverage, missing attack tests, attack tests without canaries, missing abstention tests, and missing collaboration tests for multi-agent projects.

## Runtime doctor

```bash
silo-agents-doctor
```

The doctor checks command availability and local services:

- Docker;
- Ollama;
- `.env`;
- Qdrant HTTP endpoint;
- Ollama HTTP endpoint.

Machine-readable output:

```bash
silo-agents-doctor --json
```

## What is automated

- project and agent configuration;
- namespaces and deny-by-default policy;
- synthetic corpus records;
- restricted-field canaries;
- normal, collaboration, abstention and attack templates;
- initial audit reports;
- project-specific README.

## What still requires human review

- defining the correct organizational boundaries;
- deciding which fields are sensitive;
- replacing every `REPLACE_WITH_APPROVED_*` placeholder;
- validating expected benchmark facts;
- assessing domain correctness and legal/privacy requirements;
- threat modelling beyond the generated canary attacks;
- production authentication, authorization, monitoring and incident response.

The wizard reduces setup work. It cannot infer an organization's real security policy safely without accountable human review.
