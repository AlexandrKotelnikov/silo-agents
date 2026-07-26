# Threat model

## Assets

- domain documents and embeddings;
- agent private memory;
- tool credentials;
- derived conclusions and provenance;
- policy and audit records.

## Trust boundaries

1. User to orchestrator.
2. Orchestrator to domain agent.
3. Domain agent to private retriever and tools.
4. Domain agent to policy gateway.
5. Policy gateway to orchestrator or shared conclusion store.

## Primary threats

- cross-domain retrieval;
- prompt injection requesting another agent's data;
- covert leakage through free-text fields;
- laundering restricted values through allowed fields;
- forged provenance;
- unauthorized routes;
- confused-deputy behavior by the orchestrator;
- stale permissions and cached embeddings;
- excessive context transfer;
- audit bypass.

## Required controls

- unique service identity per agent;
- authorization before retrieval;
- typed messages with allow-listed routes;
- field-level minimization and redaction;
- fail-closed policy decisions;
- provenance validation;
- canary-based leakage tests;
- immutable audit events;
- human approval for restricted transfers.

## Out of scope for milestone 1

- production identity-provider integration;
- cryptographic attestation;
- GPU side-channel isolation;
- formal policy verification;
- compromised-host protection.
