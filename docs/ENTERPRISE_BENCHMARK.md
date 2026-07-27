# Enterprise-scale benchmark

The original SiloAgents benchmark is intentionally small and interpretable. This pack tests a different question:

> Can a policy-governed multi-agent system remain useful when every agent has many similar, outdated, restricted and adversarial materials?

## Generate a pack

```bash
silo-agents-enterprise-benchmark generated/enterprise-smoke --profile smoke
silo-agents-enterprise-benchmark generated/enterprise-medium --profile medium
silo-agents-enterprise-benchmark generated/enterprise-large --profile large
```

Profiles:

| Profile | Agents | Documents per agent | Total documents | Intended use |
|---|---:|---:|---:|---|
| `smoke` | 6 | 10 | 60 | Validate the workflow and estimate runtime |
| `medium` | 6 | 50 | 300 | Main local experiment |
| `large` | 6 | 200 | 1,200 | Retrieval stress test; expensive on small machines |

The default seed is `20260727`. Set `--seed` to reproduce or deliberately vary a corpus.

## Difficulty dimensions

Every agent receives a mixture of:

- a controlled current source;
- obsolete records;
- superseded versions;
- near duplicates;
- partial records;
- documents with additional restricted fields;
- documents containing prompt injection text;
- cross-domain noise.

The cases include single-agent questions, three- and four-agent collaboration, direct attacks and unrelated abstention prompts.

## Run the experiment

```bash
cd generated/enterprise-medium
cp ../../.env .env

silo-agents validate --project silo-agents.yaml
silo-agents-audit --project silo-agents.yaml
silo-agents ingest --project silo-agents.yaml
silo-agents benchmark --project silo-agents.yaml --repeats 1
silo-agents utility --project silo-agents.yaml
```

Run `smoke` first. On an Apple M3 with 8 GB RAM, the `medium` profile may already require a long live run because every architecture invokes the local model repeatedly. Use three repeats only after the one-repeat run is technically stable.

## What success would mean

The benchmark should not be judged by one aggregate percentage. Review at least:

- routing accuracy as the corpus grows;
- expected-fact coverage;
- canary leakage;
- cross-domain contamination;
- abstention accuracy;
- token and latency growth;
- failure concentration by document kind;
- human review of the final synthesized answers.

## Known limitation and likely next finding

The generated records contain `effective_date`, `authority_rank`, `document_status`, `version` and `supersedes` metadata. The current SiloAgents retriever does not yet implement a complete source-governance policy that always prefers the latest authoritative source.

Therefore this benchmark may reveal a real architectural gap rather than produce perfect scores. A failure to distinguish current and obsolete records is a useful result: it indicates that policy-safe retrieval also needs authority, validity-period and supersession rules.

This is a synthetic stress test, not production certification and not evidence that the generated business values are realistic.
